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

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from meeting_assistant.paths import PROMPTS_DIR

PROMPT_PATH = PROMPTS_DIR / "minutes_system.txt"
QUOTE_MIN_SCORE = 90  # rapidfuzz.partial_ratio needed for a quote to count as real

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


class MinutesDraft(BaseModel):
    summary: str
    minutes: list[str]
    decisions: list[DecisionEvidence]
    action_items: list[ActionItemEvidence]


# ---------- The final record (shown in the UI, downloaded as JSON and Markdown) ----------

class Decision(BaseModel):
    text: str
    status: Literal["agreed", "rejected", "open"]
    quote: str  # the proposal quote


class ActionItem(BaseModel):
    task: str
    owner: str = "unspecified"
    deadline: str = "unspecified"
    quote: str


class Minutes(BaseModel):
    summary: str
    minutes: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]


@dataclass
class MinutesResult:
    minutes: Minutes
    dropped: list[str] = field(default_factory=list)  # what Python removed or downgraded, and why


# ---------- Quote checking ----------

def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _quote_position(quote: str, transcript_norm: str) -> int | None:
    """Where the quote is in the transcript (character index), or None if it
    isn't there closely enough."""
    q = _norm(quote)
    if not q:
        return None
    alignment = fuzz.partial_ratio_alignment(q, transcript_norm)
    if alignment is None or alignment.score < QUOTE_MIN_SCORE:
        return None
    return alignment.dest_start


def _is_backchannel(quote: str) -> bool:
    return _norm(quote) in BACKCHANNELS


def decide_status(d: DecisionEvidence, transcript_norm: str) -> tuple[str, list[str]]:
    """Python's decision rule. Only verified, non-backchannel quotes count.
    If there is both agreement and rejection, the one said later wins."""
    notes = []
    agreements, rejections = [], []
    for kind, quotes, bucket in (("agreement", d.agreement_quotes, agreements),
                                 ("rejection", d.rejection_quotes, rejections)):
        for q in quotes:
            if _is_backchannel(q):
                notes.append(f'"{d.text}": {kind} quote "{q}" is only a backchannel, ignored')
                continue
            pos = _quote_position(q, transcript_norm)
            if pos is None:
                notes.append(f'"{d.text}": {kind} quote not found in transcript, ignored: "{q}"')
                continue
            bucket.append(pos)

    if agreements and rejections:
        status = "agreed" if max(agreements) > max(rejections) else "rejected"
    elif agreements:
        status = "agreed"
    elif rejections:
        status = "rejected"
    else:
        status = "open"
    return status, notes


def _supported(value: str, quote: str, transcript_norm: str) -> bool:
    """An owner/deadline counts only if its quote is real and contains it."""
    if not value or value.strip().lower() == "unspecified":
        return False
    return _quote_position(quote, transcript_norm) is not None and _norm(value) in _norm(quote)


def verify(draft: MinutesDraft, evidence_text: str) -> MinutesResult:
    """Turn the LLM's evidence into the final record, checking every quote."""
    transcript_norm = _norm(evidence_text)
    dropped = []

    decisions = []
    for d in draft.decisions:
        if _quote_position(d.proposal_quote, transcript_norm) is None:
            dropped.append(f'Decision dropped, proposal quote not in transcript: "{d.text}"')
            continue
        status, notes = decide_status(d, transcript_norm)
        dropped.extend(notes)
        decisions.append(Decision(text=d.text, status=status, quote=d.proposal_quote))

    actions = []
    for a in draft.action_items:
        if _quote_position(a.task_quote, transcript_norm) is None:
            dropped.append(f'Action item dropped, quote not in transcript: "{a.task}"')
            continue
        owner = a.owner if _supported(a.owner, a.owner_quote, transcript_norm) else "unspecified"
        deadline = a.deadline if _supported(a.deadline, a.deadline_quote, transcript_norm) else "unspecified"
        if owner != a.owner and a.owner.strip().lower() != "unspecified":
            dropped.append(f'"{a.task}": owner "{a.owner}" not supported by a quote, set to unspecified')
        if deadline != a.deadline and a.deadline.strip().lower() != "unspecified":
            dropped.append(f'"{a.task}": deadline "{a.deadline}" not supported by a quote, set to unspecified')
        actions.append(ActionItem(task=a.task, owner=owner, deadline=deadline, quote=a.task_quote))

    minutes = Minutes(summary=draft.summary, minutes=draft.minutes,
                      decisions=decisions, action_items=actions)
    return MinutesResult(minutes, dropped)


# ---------- Main entry point ----------

def format_transcript(segment_texts: list[str], doubtful: set[int] | None = None) -> tuple[str, str]:
    """Returns (text for the LLM with numbered lines and [DOUBTFUL] marks,
    text used to check quotes, which leaves the doubtful lines out)."""
    doubtful = doubtful or set()
    for_llm, evidence = [], []
    for i, text in enumerate(segment_texts):
        mark = " [DOUBTFUL]" if i in doubtful else ""
        for_llm.append(f"[{i}]{mark} {text.strip()}")
        if i not in doubtful:
            evidence.append(text.strip())
    return "\n".join(for_llm), " ".join(evidence)


def write_minutes(segment_texts: list[str], call_structured, doubtful: set[int] | None = None,
                  prompt_path: Path = PROMPT_PATH) -> MinutesResult:
    """Stage 3. `call_structured` is llm.call_llm_structured (or a fake in tests)."""
    for_llm, evidence = format_transcript(segment_texts, doubtful)
    system = prompt_path.read_text(encoding="utf-8")
    draft = call_structured(system, for_llm, MinutesDraft)
    return verify(draft, evidence)


# ---------- Output: readable text and JSON from the same object ----------

def to_markdown(m: Minutes) -> str:
    lines = ["# Meeting record", "", "## Summary", "", m.summary, "", "## Minutes", ""]
    lines += [f"- {item}" for item in m.minutes] or ["- (none)"]
    lines += ["", "## Decisions", ""]
    lines += [f'- **{d.status}**: {d.text}  \n  _"{d.quote}"_' for d in m.decisions] or ["- (none)"]
    lines += ["", "## Action items", ""]
    lines += [f'- {a.task} (owner: {a.owner}; deadline: {a.deadline})  \n  _"{a.quote}"_'
              for a in m.action_items] or ["- (none)"]
    return "\n".join(lines) + "\n"


def to_json(m: Minutes) -> str:
    return json.dumps(m.model_dump(), indent=2, ensure_ascii=False)


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
        result = write_minutes([s.text for s in t.segments], call_llm_structured, doubtful)
    except LLMError as e:
        print(f"Stage 3 failed: {e}")
        sys.exit(1)
    print(to_markdown(result.minutes))
    if result.dropped:
        print("Removed or downgraded by the checks:")
        print("\n".join(f"- {d}" for d in result.dropped))
