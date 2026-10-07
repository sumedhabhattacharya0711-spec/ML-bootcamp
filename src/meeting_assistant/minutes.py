"""Stage 3: LLM #2 writes the meeting record.

Following Fernández et al. 2008, the LLM does not decide whether something
was agreed. It labels evidence with exact quotes (proposal, agreement,
rejection), and Python sets each decision's status from those quotes:
  agreement quote -> "agreed", rejection quote -> "rejected", neither -> "open".
Every quote is checked against the transcript; items whose quotes don't match
are dropped. Owners and deadlines are "unspecified" unless a checked quote
supports them. Lines flagged as possible hallucinations are marked
[DOUBTFUL] for the LLM and never count as evidence.

Try it:  python -m meeting_assistant.minutes data/audio/ES2004a.wav
"""

import os
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from meeting_assistant.paths import PROMPTS_DIR
from meeting_assistant.segment import split_segments
from meeting_assistant.stt import format_timestamp

PROMPT_PATH = PROMPTS_DIR / "minutes_system.txt"
QUOTE_MIN_SCORE = 90  # rapidfuzz.partial_ratio needed for a quote to count as real
# Transcripts longer than this (estimated tokens) are split into topic parts, one
# LLM call each, then merged. Groq's free tier allows ~8K tokens per minute, so one
# call must fit the prompt + transcript + answer; a 17-minute meeting is ~3.5K.
SEGMENT_MAX_TOKENS = int(os.getenv("MINUTES_SEGMENT_TOKENS", "4000"))
SAME_ITEM_SCORE = 85  # rapidfuzz.token_set_ratio at which items from two parts are the same

# A quote that is ONLY one of these is a backchannel, not agreement.
BACKCHANNELS = {"mm hmm", "mhm", "mm", "uh huh", "hmm", "yeah", "yep", "yes", "okay", "ok",
                "right", "sure", "alright", "all right"}


# ---------- What the LLM returns (evidence, no statuses) ----------

class DecisionEvidence(BaseModel):
    text: str
    proposal_quote: str
    agreement_quotes: list[str]
    rejection_quotes: list[str]


class ActionItemEvidence(BaseModel):
    task: str
    task_quote: str
    owner: str
    owner_quote: str
    deadline: str
    deadline_quote: str
    agreement_quote: str


class OpenQuestionEvidence(BaseModel):
    question: str
    quote: str


class MinutesDraft(BaseModel):
    summary: str
    minutes: list[str]
    decisions: list[DecisionEvidence]
    action_items: list[ActionItemEvidence]
    open_questions: list[OpenQuestionEvidence]


# ---------- The final record (shown in the UI, downloaded as JSON and Markdown) ----------

class Evidence(BaseModel):
    role: str          # proposal, agreement, rejection, task, owner, deadline
    quote: str
    line: int          # transcript segment index
    start: float | None = None  # seconds into the recording


class Decision(BaseModel):
    text: str
    status: Literal["agreed", "rejected", "open"]
    quote: str  # the proposal quote
    evidence: list[Evidence] = []


class ActionItem(BaseModel):
    task: str
    owner: str = "unspecified"
    deadline: str = "unspecified"
    status: Literal["agreed", "proposed", "unassigned"] = "unassigned"
    quote: str
    evidence: list[Evidence] = []


class OpenQuestion(BaseModel):
    question: str
    quote: str
    evidence: list[Evidence] = []


class Minutes(BaseModel):
    summary: str
    minutes: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[OpenQuestion] = []


@dataclass
class MinutesResult:
    minutes: Minutes
    dropped: list[str] = field(default_factory=list)  # what Python removed or downgraded, and why
    checks: dict = field(default_factory=dict)        # counts behind the faithfulness report


# ---------- Quote checking ----------

def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


class QuoteIndex:
    """The checkable transcript (doubtful lines left out), normalised into one
    string, with a map from character positions back to segment lines."""

    def __init__(self, lines: list[tuple[int, str]], starts: list[float] | None = None):
        self.line_ids, self.offsets, parts, pos = [], [], [], 0
        for line, text in lines:
            norm = _norm(text)
            if not norm:
                continue
            self.line_ids.append(line)
            self.offsets.append(pos)
            parts.append(norm)
            pos += len(norm) + 1
        self.text = " ".join(parts)
        self.starts = starts

    def find(self, quote: str) -> int | None:
        """Character position of the quote in the transcript, or None if it
        isn't there closely enough."""
        q = _norm(quote)
        if not q:
            return None
        alignment = fuzz.partial_ratio_alignment(q, self.text)
        if alignment is None or alignment.score < QUOTE_MIN_SCORE:
            return None
        return alignment.dest_start

    def evidence(self, role: str, quote: str, pos: int) -> Evidence:
        line = self.line_ids[bisect_right(self.offsets, pos) - 1]
        start = self.starts[line] if self.starts and line < len(self.starts) else None
        return Evidence(role=role, quote=quote, line=line, start=start)


def _is_backchannel(quote: str) -> bool:
    return _norm(quote) in BACKCHANNELS


def decide_status(d: DecisionEvidence, index: QuoteIndex) -> tuple[str, list[Evidence], list[str]]:
    """Python's decision rule. Only verified, non-backchannel quotes count.
    If there is both agreement and rejection, the one said later wins.
    Returns the status, the verified agreement/rejection evidence and notes."""
    notes, evidence = [], []
    agreements, rejections = [], []
    for kind, quotes, bucket in (("agreement", d.agreement_quotes, agreements),
                                 ("rejection", d.rejection_quotes, rejections)):
        for q in quotes:
            if _is_backchannel(q):
                notes.append(f'"{d.text}": {kind} quote "{q}" is only a backchannel, ignored')
                continue
            pos = index.find(q)
            if pos is None:
                notes.append(f'"{d.text}": {kind} quote not found in transcript, ignored: "{q}"')
                continue
            bucket.append(pos)
            evidence.append(index.evidence(kind, q, pos))

    if not agreements and not rejections:
        return "open", evidence, notes
    latest_agreement, latest_rejection = max(agreements, default=-1), max(rejections, default=-1)
    return ("agreed" if latest_agreement > latest_rejection else "rejected"), evidence, notes


def _supported(value: str, quote: str, index: QuoteIndex) -> int | None:
    """Position of the quote backing an owner/deadline, or None if the quote
    isn't real or doesn't contain the value."""
    if not value or value.strip().lower() == "unspecified" or _norm(value) not in _norm(quote):
        return None
    return index.find(quote)


def verify(draft: MinutesDraft, index: QuoteIndex) -> MinutesResult:
    """Turn the LLM's evidence into the final record, checking every quote."""
    dropped = []
    checks = dict(decisions_proposed=len(draft.decisions), decisions_kept=0, decisions_agreed=0,
                  actions_proposed=len(draft.action_items), actions_kept=0, actions_agreed=0,
                  open_questions_proposed=len(draft.open_questions), open_questions_kept=0,
                  owners_removed=0, deadlines_removed=0, quotes_ignored=0)

    decisions = []
    for d in draft.decisions:
        pos = index.find(d.proposal_quote)
        if pos is None:
            dropped.append(f'Decision dropped, proposal quote not in transcript: "{d.text}"')
            continue
        status, evidence, notes = decide_status(d, index)
        dropped.extend(notes)
        checks["quotes_ignored"] += len(notes)
        evidence.insert(0, index.evidence("proposal", d.proposal_quote, pos))
        decisions.append(Decision(text=d.text, status=status, quote=d.proposal_quote,
                                  evidence=sorted(evidence, key=lambda e: e.line)))

    actions = []
    for a in draft.action_items:
        pos = index.find(a.task_quote)
        if pos is None:
            dropped.append(f'Action item dropped, quote not in transcript: "{a.task}"')
            continue
        checked, evidence = {}, [index.evidence("task", a.task_quote, pos)]
        for name, value, quote in (("owner", a.owner, a.owner_quote),
                                   ("deadline", a.deadline, a.deadline_quote)):
            found = _supported(value, quote, index)
            if found is not None:
                checked[name] = value
                evidence.append(index.evidence(name, quote, found))
                continue
            checked[name] = "unspecified"
            if value.strip().lower() != "unspecified":
                dropped.append(f'"{a.task}": {name} "{value}" not supported by a quote, set to unspecified')
                checks[f"{name}s_removed"] += 1
        status = _action_status(a, checked["owner"], index, evidence, dropped)
        actions.append(ActionItem(task=a.task, quote=a.task_quote, status=status,
                                  evidence=_unique(evidence), **checked))

    questions = []
    for q in draft.open_questions:
        pos = index.find(q.quote)
        if pos is None:
            dropped.append(f'Open question dropped, quote not in transcript: "{q.question}"')
            continue
        questions.append(OpenQuestion(question=q.question, quote=q.quote,
                                      evidence=[index.evidence("open question", q.quote, pos)]))

    checks["decisions_kept"] = len(decisions)
    checks["open_questions_kept"] = len(questions)
    checks["decisions_agreed"] = sum(d.status == "agreed" for d in decisions)
    checks["actions_kept"] = len(actions)
    checks["actions_agreed"] = sum(a.status == "agreed" for a in actions)
    minutes = Minutes(summary=draft.summary, minutes=draft.minutes,
                      decisions=decisions, action_items=actions, open_questions=questions)
    return MinutesResult(minutes, dropped, checks)


def _action_status(a: ActionItemEvidence, owner: str, index: QuoteIndex,
                   evidence: list[Evidence], dropped: list[str]) -> str:
    """Purver et al.'s agreement part of an action item: "agreed" needs a named
    owner and a verified quote where the task was accepted or volunteered."""
    if owner == "unspecified":
        return "unassigned"
    quote = a.agreement_quote.strip()
    if not quote:
        return "proposed"
    pos = None if _is_backchannel(quote) else index.find(quote)
    if pos is None:
        dropped.append(f'"{a.task}": agreement quote not verified, status set to proposed: "{quote}"')
        return "proposed"
    evidence.append(index.evidence("agreement", quote, pos))
    return "agreed"


def _unique(evidence: list[Evidence]) -> list[Evidence]:
    """One entry per (line, quote), in transcript order."""
    seen, out = set(), []
    for e in sorted(evidence, key=lambda e: e.line):
        if (e.line, _norm(e.quote)) not in seen:
            seen.add((e.line, _norm(e.quote)))
            out.append(e)
    return out


def support_rate(checks: dict) -> float | None:
    """Share of the LLM's decisions, action items and open questions whose evidence verified."""
    kinds = ("decisions", "actions", "open_questions")
    proposed = sum(checks.get(f"{k}_proposed", 0) for k in kinds)
    kept = sum(checks.get(f"{k}_kept", 0) for k in kinds)
    return round(kept / proposed, 3) if proposed else None


# ---------- Main entry point ----------

def format_transcript(segment_texts: list[str],
                      doubtful: set[int] | None = None) -> tuple[str, list[tuple[int, str]]]:
    """Returns (text for the LLM with numbered lines and [DOUBTFUL] marks,
    (line, text) pairs used to check quotes, which leave the doubtful lines out)."""
    doubtful = doubtful or set()
    for_llm, evidence = [], []
    for i, text in enumerate(segment_texts):
        mark = " [DOUBTFUL]" if i in doubtful else ""
        for_llm.append(f"[{i}]{mark} {text.strip()}")
        if i not in doubtful:
            evidence.append((i, text.strip()))
    return "\n".join(for_llm), evidence


def write_minutes(segment_texts: list[str], call_structured, doubtful: set[int] | None = None,
                  starts: list[float] | None = None, prompt_path: Path = PROMPT_PATH,
                  call_text=None, max_tokens: int | None = None) -> MinutesResult:
    """Stage 3. `call_structured` is llm.call_llm_structured (or a fake in tests);
    `starts` are the segments' start times, used to timestamp the evidence.

    Long transcripts are split into topic parts (segment.py). Each part is one
    call that also sees the summaries of the earlier parts as context (recursive
    summarization); the parts' drafts are then merged (`call_text`, e.g.
    llm.call_llm, writes the overall summary) and verified against the whole
    transcript, so a proposal in one part and its agreement in another still
    combine into one agreed decision."""
    for_llm, evidence = format_transcript(segment_texts, doubtful)
    system = prompt_path.read_text(encoding="utf-8")
    lines = for_llm.split("\n")
    parts = split_segments(lines, max_tokens or SEGMENT_MAX_TOKENS)
    if len(parts) == 1:
        draft = call_structured(system, for_llm, MinutesDraft)
    else:
        drafts = []
        for k, (a, b) in enumerate(parts):
            user = _part_message(k, len(parts), [d.summary for d in drafts], lines[a:b])
            drafts.append(call_structured(system, user, MinutesDraft))
        draft = merge_drafts(drafts, call_text)
    result = verify(draft, QuoteIndex(evidence, starts))
    result.checks["parts"] = len(parts)
    return result


def _part_message(k: int, total: int, earlier: list[str], part_lines: list[str]) -> str:
    context = "\n".join(f"- Part {i + 1}: {s}" for i, s in enumerate(earlier)) or "- (this is the first part)"
    return (f"This is part {k + 1} of {total} of a long meeting.\n\n"
            f"EARLIER PARTS (context only, to resolve references; never quote from here):\n{context}\n\n"
            f"TRANSCRIPT OF PART {k + 1}:\n" + "\n".join(part_lines) +
            "\n\nReminder: every quote must be copied from the transcript lines of this part.")


MERGE_SUMMARY_SYSTEM = (
    "You merge summaries of consecutive parts of one meeting into a single summary of 3 to 6 "
    "sentences. Use only the given summaries. Keep names, numbers and decisions exactly as "
    "stated; do not upgrade proposals into decisions. Reply with the summary only.")


def _same(a: str, b: str) -> bool:
    return fuzz.token_set_ratio(_norm(a), _norm(b)) >= SAME_ITEM_SCORE


def _union(a: list[str], b: list[str]) -> list[str]:
    out = list(a)
    for q in b:
        if not any(_norm(q) == _norm(x) for x in out):
            out.append(q)
    return out


def merge_drafts(drafts: list[MinutesDraft], call_text=None) -> MinutesDraft:
    """Reduce step: one draft from the parts' drafts. Duplicates (the same item
    seen in two parts) are merged and their quotes combined; nothing is decided
    here, verify() still sets every status from the quotes."""
    summaries = [d.summary for d in drafts if d.summary.strip()]
    summary = " ".join(summaries)
    if call_text and len(summaries) > 1:
        try:
            summary = call_text(MERGE_SUMMARY_SYSTEM,
                                "\n".join(f"Part {i + 1}: {s}" for i, s in enumerate(summaries))).strip() or summary
        except Exception:  # LLM failure: keep the parts' summaries joined, the record still works
            pass

    minutes, decisions, actions, questions = [], [], [], []
    for d in drafts:
        minutes += [m for m in d.minutes if not any(_same(m, x) for x in minutes)]
        for dec in d.decisions:
            match = next((x for x in decisions if _same(x.text, dec.text)
                          or _norm(x.proposal_quote) == _norm(dec.proposal_quote)), None)
            if match is None:
                decisions.append(dec.model_copy(deep=True))
            else:
                match.agreement_quotes = _union(match.agreement_quotes, dec.agreement_quotes)
                match.rejection_quotes = _union(match.rejection_quotes, dec.rejection_quotes)
        for act in d.action_items:
            match = next((x for x in actions if _same(x.task, act.task)), None)
            if match is None:
                actions.append(act.model_copy(deep=True))
                continue
            for field_name, quote_name in (("owner", "owner_quote"), ("deadline", "deadline_quote")):
                if getattr(match, field_name).strip().lower() == "unspecified":
                    setattr(match, field_name, getattr(act, field_name))
                    setattr(match, quote_name, getattr(act, quote_name))
            match.agreement_quote = match.agreement_quote or act.agreement_quote
        questions += [q for q in d.open_questions if not any(_same(q.question, x.question) for x in questions)]
    return MinutesDraft(summary=summary, minutes=minutes, decisions=decisions,
                        action_items=actions, open_questions=questions)


# ---------- Output: readable text and JSON from the same object ----------

def _evidence_lines(evidence: list[Evidence]) -> list[str]:
    out = []
    for e in evidence:
        when = f"[{format_timestamp(e.start)}] " if e.start is not None else f"[line {e.line}] "
        out.append(f'  - {when}{e.role}: _"{e.quote}"_')
    return out


def to_markdown(m: Minutes) -> str:
    lines = ["# Meeting record", "", "## Summary", "", m.summary, "", "## Minutes", ""]
    lines += [f"- {item}" for item in m.minutes] or ["- (none)"]
    lines += ["", "## Decisions", ""]
    for d in m.decisions:
        lines += [f"- **{d.status}**: {d.text}"] + _evidence_lines(d.evidence)
    if not m.decisions:
        lines.append("- (none)")
    lines += ["", "## Action items", ""]
    for a in m.action_items:
        lines += [f"- {a.task} (owner: {a.owner}; deadline: {a.deadline}; status: {a.status})"]
        lines += _evidence_lines(a.evidence)
    if not m.action_items:
        lines.append("- (none)")
    lines += ["", "## Open questions", ""]
    for q in m.open_questions:
        lines += [f"- {q.question}"] + _evidence_lines(q.evidence)
    if not m.open_questions:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def to_json(m: Minutes) -> str:
    return m.model_dump_json(indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m meeting_assistant.minutes <audio file> [model size]")
        sys.exit(1)
    from meeting_assistant.hallucination import doubtful_indices, score_segments
    from meeting_assistant.llm import LLMError, call_llm_structured
    from meeting_assistant.stt import AudioInputError, load_model, transcribe

    audio, size = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "medium")
    try:
        t = transcribe(audio, load_model(size), model_size=size)
    except AudioInputError as e:
        print(f"Stage 1 failed: {e}")
        sys.exit(1)
    doubtful = doubtful_indices(score_segments(t.segments, t.duration_s))
    try:
        result = write_minutes([s.text for s in t.segments], call_llm_structured, doubtful,
                               [s.start for s in t.segments])
    except LLMError as e:
        print(f"Stage 3 failed: {e}")
        sys.exit(1)
    print(to_markdown(result.minutes))
    if result.dropped:
        print("Removed or downgraded by the checks:")
        print("\n".join(f"- {d}" for d in result.dropped))
