"""Stage 3: LLM #2 writes the meeting record.

Following Fernández et al. 2008, the LLM does not decide whether something
was agreed. It labels evidence with exact quotes (proposal, agreement,
rejection), and Python sets each decision's status from those quotes:
  agreement quote -> "agreed", rejection quote -> "rejected", neither -> "open".
Every quote is checked against the transcript; items whose quotes don't match
are dropped. Owners and deadlines are "unspecified" unless a checked quote
supports them. Lines flagged as possible hallucinations are marked
[DOUBTFUL] for the LLM and never count as evidence.

When the speakers are known (speakers.py), every line is shown with its
speaker and every quote knows who said it. Then an agreement counts only if
someone other than the proposer said it, and "I'll do it" makes its speaker
the owner of the task.

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

# A quote with one of these commits its speaker to a task ("I'll send it", "let me check").
# Matched on normalised text, where "I'll" is "i ll"; "i can t" (can't) is excluded.
VOLUNTEER = re.compile(r"\b(i ll|i will|i shall|i can(?! t\b)|i m going to|i am going to|i m gonna|let me|"
                       r"leave it (to|with) me|i ll take|i ve got (it|this))\b")

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
    speaker: str | None = None  # who said it, when speakers are known


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
    participants: list[str] = []  # speaker names, when speakers are known


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

    def __init__(self, lines: list[tuple[int, str]], starts: list[float] | None = None,
                 speakers: list[str | None] | None = None):
        self.speakers = speakers  # speaker label of each segment line, or None
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
        q = _norm(self._without_label(quote))
        if not q:
            return None
        alignment = fuzz.partial_ratio_alignment(q, self.text)
        if alignment is None or alignment.score < QUOTE_MIN_SCORE:
            return None
        return alignment.dest_start

    def _without_label(self, quote: str) -> str:
        """The LLM sometimes copies the speaker label too ("Priya: yes, let's"):
        the label is not part of what was said."""
        if self.speakers:
            label, colon, rest = quote.partition(":")
            if colon and _norm(label) in {_norm(s) for s in self.speakers if s}:
                return rest
        return quote

    def line_at(self, pos: int) -> int:
        return self.line_ids[bisect_right(self.offsets, pos) - 1]

    def speaker_at(self, pos: int) -> str | None:
        line = self.line_at(pos)
        return self.speakers[line] if self.speakers and line < len(self.speakers) else None

    def evidence(self, role: str, quote: str, pos: int) -> Evidence:
        line = self.line_at(pos)
        start = self.starts[line] if self.starts and line < len(self.starts) else None
        return Evidence(role=role, quote=self._without_label(quote).strip(), line=line, start=start,
                        speaker=self.speaker_at(pos))


def _is_backchannel(quote: str) -> bool:
    return _norm(quote) in BACKCHANNELS


def _same_person(a: str | None, b: str | None) -> bool:
    return bool(a and b) and _norm(a) == _norm(b)


def decide_status(d: DecisionEvidence, index: QuoteIndex,
                  proposer: str | None = None) -> tuple[str, list[Evidence], list[str]]:
    """Python's decision rule. Only verified, non-backchannel quotes count, and
    when speakers are known, an agreement by the proposer themself does not.
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
            if kind == "agreement" and _same_person(index.speaker_at(pos), proposer):
                notes.append(f'"{d.text}": agreement quote is by the proposer ({proposer}) themself, ignored: "{q}"')
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
    quote = index._without_label(quote)  # a copied "Priya:" label is not evidence that Priya owns it
    if not value or value.strip().lower() == "unspecified" or _norm(value) not in _norm(quote):
        return None
    return index.find(quote)


def verify(draft: MinutesDraft, index: QuoteIndex) -> MinutesResult:
    """Turn the LLM's evidence into the final record, checking every quote."""
    dropped = []
    checks = dict(decisions_proposed=len(draft.decisions), decisions_kept=0, decisions_agreed=0,
                  actions_proposed=len(draft.action_items), actions_kept=0, actions_agreed=0,
                  open_questions_proposed=len(draft.open_questions), open_questions_kept=0,
                  owners_removed=0, deadlines_removed=0, quotes_ignored=0, owners_from_speaker=0)

    decisions = []
    for d in draft.decisions:
        pos = index.find(d.proposal_quote)
        if pos is None:
            dropped.append(f'Decision dropped, proposal quote not in transcript: "{d.text}"')
            continue
        status, evidence, notes = decide_status(d, index, index.speaker_at(pos))
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
            if found is None and name == "owner":
                found = _volunteered(value, quote, index)
            if found is not None:
                checked[name] = value
                evidence.append(index.evidence(name, quote, found))
                continue
            checked[name] = "unspecified"
            if value.strip().lower() != "unspecified":
                dropped.append(f'"{a.task}": {name} "{value}" not supported by a quote, set to unspecified')
                checks[f"{name}s_removed"] += 1
        if checked["owner"] == "unspecified":
            owner, found, quote = _owner_from_speaker(a, index)
            if owner:
                checked["owner"] = owner
                evidence.append(index.evidence("owner", quote, found))
                checks["owners_from_speaker"] += 1
                if a.owner.strip().lower() != "unspecified":  # it was counted as removed above
                    checks["owners_removed"] -= 1
                dropped.append(f'"{a.task}": owner set to {owner}, who volunteered: "{quote}"')
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


def _volunteered(owner: str, quote: str, index: QuoteIndex) -> int | None:
    """Position of `quote` if the owner said it themself as a commitment
    ("I'll send it"), which supports them as owner without saying their name."""
    if not index.speakers or not owner or owner.strip().lower() == "unspecified":
        return None
    if not quote or not VOLUNTEER.search(_norm(quote)):
        return None
    pos = index.find(quote)
    if pos is None or not _same_person(index.speaker_at(pos), owner):
        return None
    return pos


def _owner_from_speaker(a: ActionItemEvidence, index: QuoteIndex) -> tuple[str | None, int | None, str]:
    """When no owner is supported, the speaker of a verified "I'll do it" quote
    (the acceptance, owner or task quote) becomes the owner."""
    if not index.speakers:
        return None, None, ""
    for quote in (a.agreement_quote, a.owner_quote, a.task_quote):
        if not quote or not VOLUNTEER.search(_norm(quote)):
            continue
        pos = index.find(quote)
        speaker = index.speaker_at(pos) if pos is not None else None
        if speaker:
            return speaker, pos, quote
    return None, None, ""


def _action_status(a: ActionItemEvidence, owner: str, index: QuoteIndex,
                   evidence: list[Evidence], dropped: list[str]) -> str:
    """Purver et al.'s agreement part of an action item: "agreed" needs a named
    owner and a verified quote where the task was accepted or volunteered. When
    the owner is a known speaker, that quote must be theirs."""
    if owner == "unspecified":
        return "unassigned"
    quote = a.agreement_quote.strip()
    if not quote:
        return "proposed"
    pos = None if _is_backchannel(quote) else index.find(quote)
    if pos is None:
        dropped.append(f'"{a.task}": agreement quote not verified, status set to proposed: "{quote}"')
        return "proposed"
    said_by = index.speaker_at(pos)
    owner_is_speaker = any(_same_person(owner, s) for s in index.speakers or [])
    if owner_is_speaker and said_by and not _same_person(said_by, owner):
        dropped.append(f'"{a.task}": accepted by {said_by}, not by the owner {owner}, status set to proposed')
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

def format_transcript(segment_texts: list[str], doubtful: set[int] | None = None,
                      speakers: list[str | None] | None = None) -> tuple[str, list[tuple[int, str]]]:
    """Returns (text for the LLM with numbered lines, [DOUBTFUL] marks and the
    speaker of each line when known, (line, text) pairs used to check quotes,
    which leave the doubtful lines out)."""
    doubtful = doubtful or set()
    for_llm, evidence = [], []
    for i, text in enumerate(segment_texts):
        mark = " [DOUBTFUL]" if i in doubtful else ""
        who = f" {speakers[i]}:" if speakers and i < len(speakers) and speakers[i] else ""
        for_llm.append(f"[{i}]{mark}{who} {text.strip()}")
        if i not in doubtful:
            evidence.append((i, text.strip()))
    return "\n".join(for_llm), evidence


def write_minutes(segment_texts: list[str], call_structured, doubtful: set[int] | None = None,
                  starts: list[float] | None = None, prompt_path: Path = PROMPT_PATH,
                  call_text=None, max_tokens: int | None = None,
                  speakers: list[str | None] | None = None) -> MinutesResult:
    """Stage 3. `call_structured` is llm.call_llm_structured (or a fake in tests);
    `starts` are the segments' start times, used to timestamp the evidence;
    `speakers` the speaker label of each segment, when known.

    Long transcripts are split into topic parts (segment.py). Each part is one
    call that also sees the summaries of the earlier parts as context (recursive
    summarization); the parts' drafts are then merged (`call_text`, e.g.
    llm.call_llm, writes the overall summary) and verified against the whole
    transcript, so a proposal in one part and its agreement in another still
    combine into one agreed decision."""
    for_llm, evidence = format_transcript(segment_texts, doubtful, speakers)
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
    result = verify(draft, QuoteIndex(evidence, starts, speakers))
    if speakers:
        result.minutes.participants = list(dict.fromkeys(s for s in speakers if s))
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
        who = f"{e.speaker}, " if e.speaker else ""
        out.append(f'  - {when}{who}{e.role}: _"{e.quote}"_')
    return out


def to_markdown(m: Minutes) -> str:
    lines = ["# Meeting record", ""]
    if m.participants:
        lines += ["## Participants", "", ", ".join(m.participants), ""]
    lines += ["## Summary", "", m.summary, "", "## Minutes", ""]
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


def rename_in_minutes(m: Minutes, changes: dict[str, str], line_speakers: list[str | None]) -> None:
    """Apply edited speaker names to a finished record in place, without asking
    the LLM again. `changes` is {old label: new label} (speakers.rename_speakers);
    `line_speakers` the new label of each transcript line. Quotes stay word for
    word as spoken."""
    from meeting_assistant.speakers import replace_labels

    m.summary = replace_labels(m.summary, changes)
    m.minutes = [replace_labels(x, changes) for x in m.minutes]
    m.participants = list(dict.fromkeys(s for s in line_speakers if s))
    for d in m.decisions:
        d.text = replace_labels(d.text, changes)
    for a in m.action_items:
        a.task, a.owner = replace_labels(a.task, changes), replace_labels(a.owner, changes)
    for q in m.open_questions:
        q.question = replace_labels(q.question, changes)
    for item in [*m.decisions, *m.action_items, *m.open_questions]:
        for e in item.evidence:
            if e.speaker and e.line < len(line_speakers):
                e.speaker = line_speakers[e.line]


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
