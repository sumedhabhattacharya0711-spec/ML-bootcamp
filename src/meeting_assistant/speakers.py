"""Who spoke when, and what they are called.

Stage 1a  diarize, assign_speakers
          pyannote finds the speaker turns in the audio. Every Whisper word gets
          the speaker talking at that moment, and a line in which the speaker
          changes is split at the change, so every line has exactly one speaker.
Stage 2b  name_speakers
          LLM #1 points at the places where a name is revealed ("Hi, I'm Priya",
          "Priya, what do you think?"). Python checks each claim against the
          transcript and the speaker turns, and names a speaker only when the
          evidence holds up. Same rule as everywhere else in the project: the
          model proposes, plain Python verifies.

Speakers are "S1", "S2", ... in order of first appearance and are shown as
"Speaker 1", "Speaker 2", ... until they are named. Names can always be edited
afterwards (rename_speakers); that never reruns a model.

Try it:  python -m meeting_assistant.speakers data/audio/ES2004a.wav --speakers 4
"""

import os
import re
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from rapidfuzz import fuzz

from meeting_assistant.llm import LLMError, parse_json_reply
from meeting_assistant.paths import PROMPTS_DIR
from meeting_assistant.segment import split_segments

NAMES_PROMPT_PATH = PROMPTS_DIR / "speaker_names.txt"

# pyannote 4 sends usage telemetry to otel.pyannote.ai unless told not to, and
# flushes it at exit: on a slow connection the process then hangs for minutes
# after the run is saved. Off unless PYANNOTE_METRICS_ENABLED is set (read on import).
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

# ---------- Settings ----------

# pyannote's open diarization pipeline (Sept 2025). It is gated: the Hugging Face
# account behind HF_TOKEN must accept its conditions once on the model page.
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL") or "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000

NEAREST_TURN_S = 0.5  # a word outside every turn takes the speaker of a turn this close
SNAP_WORDS = 2        # a speaker change mid-sentence moves to a sentence end this close:
                      # Whisper's word times are off by ~0.1-0.3 s, so a change that
                      # lands one word into a sentence is usually timing, not a new voice
MIN_RUN_WORDS = 2     # a shorter run of one speaker inside a line is jitter, unless it
                      # is a whole sentence of its own ("Yes." between two sentences)

REPLY_LINES = 4       # how far (in lines) to look for the person someone addressed
SELF_POINTS = 3       # "I'm Priya": the speaker says it themself, strong evidence
ADDRESSED_POINTS = 1  # "Priya, what do you think?" + Priya answers: weaker, adds up
HIGH_CONFIDENCE = 2   # points needed for a "high" confidence name

# Generic words in a role ("industrial designer"): the other words tell roles apart.
ROLE_NOUNS = {"designer", "manager", "expert", "specialist", "lead", "leader", "person", "officer",
              "engineer", "developer", "analyst", "director", "head", "owner", "coordinator"}
ROLE_FILLER = {"the", "a", "an", "our", "your", "of", "for", "and", "in", "charge", "team"}

# Words people are addressed with that are not names ("thanks, everyone").
NOT_NAMES = {
    "all", "everyone", "everybody", "guys", "team", "folks", "people", "sir", "madam", "maam",
    "man", "dude", "buddy", "boss", "mate", "friend", "friends", "you", "speaker", "chair",
}


# A line that is only one of these is not an answer to being addressed.
BACKCHANNELS = {"mm hmm", "mhm", "mm", "uh huh", "hmm", "yeah", "yep", "okay", "ok", "right", "sure"}


class DiarizationError(Exception):
    """Diarization could not run. The message is shown to the user as is."""


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _ends_sentence(token: str) -> bool:
    return token.rstrip("\"')").endswith((".", "?", "!"))


# ---------- Stage 1a: speaker turns ----------

@dataclass
class Turn:
    start: float
    end: float
    speaker: str  # "S1", "S2", ...


def relabel(turns: list[tuple[float, float, str]]) -> list[Turn]:
    """Sort by time and rename pyannote's labels (SPEAKER_00, ...) to S1, S2, ...
    in order of first appearance, so Speaker 1 is whoever spoke first."""
    ids: dict[str, str] = {}
    out = []
    for start, end, label in sorted(turns, key=lambda t: (t[0], t[1])):
        if end <= start:
            continue
        ids.setdefault(label, f"S{len(ids) + 1}")
        out.append(Turn(round(float(start), 3), round(float(end), 3), ids[label]))
    return out


class Diarizer:
    """A loaded pyannote pipeline, reused across runs (like the Whisper model).
    Call it with an audio path; it returns the speaker turns."""

    def __init__(self, pipeline, device: str, model: str = DIARIZATION_MODEL):
        self.pipeline, self.device, self.model = pipeline, device, model

    def __call__(self, audio_path, num_speakers: int | None = None) -> list[Turn]:
        import torch
        from faster_whisper import decode_audio

        # Decoded with the same ffmpeg path Whisper uses and handed over in memory,
        # so pyannote's own file reader (torchcodec) is never needed.
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        options = {"num_speakers": num_speakers} if num_speakers else {}
        out = self.pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE},
                            **options)
        annotation = _annotation(out)
        return relabel([(seg.start, seg.end, label) for seg, _, label in annotation.itertracks(yield_label=True)])


def _annotation(out):
    """pyannote 4 returns an object holding an "exclusive" diarization (one
    speaker at a time, which is what a transcript needs); 3.x returns the
    annotation itself."""
    for name in ("exclusive_speaker_diarization", "speaker_diarization"):
        value = getattr(out, name, None)
        if value is not None:
            return value
    return out


def load_diarizer(model: str = DIARIZATION_MODEL) -> Diarizer:
    """Load pyannote once, on the GPU when there is one. Raises DiarizationError
    with a message the user can act on."""
    page = f"https://huggingface.co/{model}"
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise DiarizationError(f"HF_TOKEN is not set: add a Hugging Face token to .env and accept the "
                               f"model's conditions at {page}.")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(f"pyannote.audio is not installed ({e}); run pip install -r requirements.txt.") from e
    try:
        try:
            pipeline = Pipeline.from_pretrained(model, token=token)
        except TypeError:  # pyannote 3.x names the argument differently
            pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    except Exception as e:
        raise DiarizationError(f"Could not load {model} ({e}). Check HF_TOKEN, and that its account "
                               f"accepted the conditions at {page}.") from e
    if pipeline is None:  # pyannote 3.x returns None instead of raising when access is refused
        raise DiarizationError(f"Access to {model} was refused: accept its conditions at {page}.")

    device = "cpu"
    if torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
            device = "cuda"
        except Exception as e:  # out of GPU memory, missing CUDA pieces
            print(f"Diarization GPU load failed ({e}); using the CPU.", file=sys.stderr)
    return Diarizer(pipeline, device, model)


# ---------- Stage 1a: one speaker per line ----------

class _TurnIndex:
    """Finds the speaker of a time span quickly (turns sorted by start)."""

    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.longest = max((t.end - t.start for t in self.turns), default=0.0)

    def speaker(self, start: float, end: float) -> str | None:
        """The speaker overlapping [start, end] the most; else the speaker of the
        nearest turn within NEAREST_TURN_S; else None (no voice there)."""
        overlap: Counter = Counter()
        nearest_gap, nearest = NEAREST_TURN_S, None
        i = bisect_right(self.starts, end + NEAREST_TURN_S)
        while i > 0:
            i -= 1
            t = self.turns[i]
            if t.start < start - self.longest - NEAREST_TURN_S:
                break  # every earlier turn ends too early to matter
            shared = min(end, t.end) - max(start, t.start)
            if shared > 0:
                overlap[t.speaker] += shared
            elif -shared <= nearest_gap:  # -shared is the gap between the two spans
                nearest_gap, nearest = -shared, t.speaker
        if overlap:
            return overlap.most_common(1)[0][0]
        return nearest


def _runs(labels: list) -> list[tuple[int, int, object]]:
    """[(start, end, label)] for each run of equal labels."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def _fill_gaps(labels: list[str | None]) -> list[str | None]:
    """A word no turn covers takes the speaker of the word before it (or after it)."""
    out = list(labels)
    for i in range(1, len(out)):
        if out[i] is None:
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if out[i] is None:
            out[i] = out[i + 1]
    return out


def smooth_labels(tokens: list[str], labels: list[str | None]) -> list[str | None]:
    """Clean up per-word speakers inside one Whisper line:
    1. a speaker change in the middle of a sentence moves to a sentence end at
       most SNAP_WORDS words away;
    2. a run shorter than MIN_RUN_WORDS that is not a whole sentence joins the run before it."""
    labels = _fill_gaps(labels)
    n = len(labels)
    ends = [_ends_sentence(t) for t in tokens]
    for b in range(1, n):  # b = a boundary: words[:b] | words[b:]
        if labels[b] == labels[b - 1] or ends[b - 1]:
            continue
        near = [c for c in range(max(1, b - SNAP_WORDS), min(n, b + SNAP_WORDS) + 1) if ends[c - 1]]
        if not near:
            continue
        c = min(near, key=lambda c: (abs(c - b), c))
        if c > b:
            labels[b:c] = [labels[b - 1]] * (c - b)
        elif c < b:
            labels[c:b] = [labels[b]] * (b - c)

    runs = _runs(labels)
    if len(runs) > 1:
        for k, (a, z, _) in enumerate(runs):
            whole_sentence = (a == 0 or ends[a - 1]) and ends[z - 1]
            if z - a < MIN_RUN_WORDS and not whole_sentence:
                neighbour = labels[a - 1] if a > 0 else runs[k + 1][2]
                labels[a:z] = [neighbour] * (z - a)
    return labels


def assign_speakers(segments: list, turns: list[Turn]) -> list:
    """Give every line one speaker. A line in which the speaker changes is split
    at the change, one new line per speaker; a line with one speaker keeps its
    text exactly. `segments` are stt.Segment objects; new ones are returned."""
    if not turns:
        return list(segments)
    index = _TurnIndex(turns)
    out = []
    for seg in segments:
        if not seg.words:
            out.append(replace(seg, speaker=index.speaker(seg.start, seg.end)))
            continue
        labels = smooth_labels([w.text for w in seg.words],
                               [index.speaker(w.start, w.end) for w in seg.words])
        runs = _runs(labels)
        if len(runs) == 1:
            out.append(replace(seg, speaker=labels[0]))
            continue
        for a, z, label in runs:
            words = seg.words[a:z]
            out.append(replace(seg, start=seg.start if a == 0 else words[0].start,
                               end=seg.end if z == len(seg.words) else words[-1].end,
                               text=" ".join(w.text for w in words), words=list(words), speaker=label))
    return out


# ---------- Speakers and their names ----------

@dataclass
class Speaker:
    id: str               # "S1", "S2", ...
    name: str = ""        # "" = not named yet, shown as "Speaker N"
    source: str = ""      # how the name was set: "self-introduction", "addressed" or "edited"
    confidence: str = ""  # "high" or "low", for names found automatically
    note: str = ""        # why a speaker stayed unnamed, e.g. two equally likely names
    talk_s: float = 0.0
    lines: int = 0
    role: str = ""        # e.g. "industrial designer", from what the speaker said about themself
    role_source: str = "" # "self" (said it themself) or "edited"

    @property
    def label(self) -> str:
        return self.name or default_label(self.id)


def default_label(speaker_id: str) -> str:
    return f"Speaker {speaker_id.lstrip('S')}"


def summarize(segments: list) -> list[Speaker]:
    """One Speaker per diarized speaker, with talk time and line count."""
    found: dict[str, Speaker] = {}
    for s in segments:
        if s.speaker is None:
            continue
        sp = found.setdefault(s.speaker, Speaker(s.speaker))
        sp.talk_s += max(0.0, s.end - s.start)
        sp.lines += 1
    for sp in found.values():
        sp.talk_s = round(sp.talk_s, 1)
    return sorted(found.values(), key=lambda s: int(s.id.lstrip("S") or 0))


def line_labels(segments: list, speakers: list[Speaker]) -> list[str | None]:
    """The display label of each line's speaker ("Priya", "Speaker 2"), or None."""
    labels = {s.id: s.label for s in speakers}
    return [labels.get(s.speaker) if s.speaker else None for s in segments]


@dataclass
class NameEvidence:
    """One place in the transcript where the LLM saw a speaker's name."""
    speaker: str    # who the claim says it is ("S2")
    name: str
    kind: str       # "self" (they said their own name), "addressed" (someone said it to them),
                    # "role" (they said their own role; `name` then holds the role) or
                    # "assigned_role" (someone gave a named person a role: "Courtney, our marketing person")
    line: int
    quote: str
    accepted: bool = False
    reason: str = ""
    role: str = ""  # for "assigned_role": the role given to `name`


def _canonical(name: str, attendees: list[str]) -> str:
    """Spell a heard name like the matching attendee ("pria" -> "Priya Sharma");
    otherwise just capitalise it."""
    n = _norm(name)
    for a in attendees:
        if fuzz.ratio(_norm(a), n) >= 85:
            return a
    for a in attendees:  # first name only
        first = _norm(a).split()[0] if _norm(a) else ""
        if first and fuzz.ratio(first, n) >= 85:
            return a
    return " ".join(w[:1].upper() + w[1:] for w in name.split())


def _locate(quote: str, line: int, lines: list[str], doubtful: set[int]) -> int | None:
    """The line (the given one or a close neighbour) that really contains the quote."""
    q = _norm(quote)
    if not q:
        return None
    for j in (line, line - 1, line + 1, line - 2, line + 2):
        if 0 <= j < len(lines) and j not in doubtful:
            text = _norm(lines[j])
            if len(q) <= len(text) + 3 and fuzz.partial_ratio(q, text) >= 90:
                return j
    return None


def _neighbours(line: int, lines: list[str], line_speakers: list[str | None],
                doubtful: set[int]) -> dict[str, str]:
    """Who the person on `line` can be talking to: the last other speaker before
    it ("thanks, Rahul" thanks whoever just spoke) and the first other speaker
    who answers after it ("Priya, what do you think?"). A third person's
    "mm-hmm" in between is not an answer and is skipped."""
    who, found = line_speakers[line], {}
    for j in range(line - 1, max(-1, line - 1 - REPLY_LINES), -1):
        sp = line_speakers[j]
        if j not in doubtful and sp and sp != who:
            found[sp] = "spoke just before"
            break
    for j in range(line + 1, min(len(line_speakers), line + 1 + REPLY_LINES)):
        sp = line_speakers[j]
        if j in doubtful or not sp or sp == who or _norm(lines[j]) in BACKCHANNELS:
            continue
        found.setdefault(sp, "answered right after")
        break
    return found


def check_assigned_role(claim: dict, lines: list[str], doubtful: set[int], attendees: list[str]) -> NameEvidence:
    """Check "someone gave a named person a role" ("the marketing person, Courtney"):
    the quote must be in the transcript and contain both the name and the role.
    Who said it does not matter, and no voice is claimed: the role belongs to the name."""
    name = " ".join(str(claim.get("name", "")).split()).strip(" .,;:!?\"'")
    role = clean_role(str(claim.get("role", "")))
    quote = re.sub(r"^\s*S\d+\s*:\s*", "", str(claim.get("quote", "")).strip())
    try:
        line = int(claim.get("line"))
    except (TypeError, ValueError):
        line = -1
    ev = NameEvidence("", name, "assigned_role", line, quote, role=role)
    if not name or _norm(name) in NOT_NAMES:
        ev.reason = "not a person's name"
        return ev
    if not role_words(role):
        ev.reason = "not a role"
        return ev
    found = _locate(quote, line, lines, doubtful)
    if found is None:
        ev.reason = "quote not found in the transcript"
        return ev
    ev.line = found
    words = set(_norm(quote).split())
    if not any(w in words for w in _norm(name).split() if len(w) >= 2):
        ev.reason = "the quote does not contain the name"
    elif not all(w in words for w in role_words(role)):
        ev.reason = "the quote does not contain the role"
    else:
        ev.name, ev.accepted, ev.reason = _canonical(name, attendees), True, "role given by name"
    return ev


def named_roles(evidence: list[NameEvidence]) -> dict[str, str]:
    """{person's name: role} from accepted "assigned_role" claims (first one wins)."""
    out: dict[str, str] = {}
    for e in evidence:
        if e.accepted and e.kind == "assigned_role":
            out.setdefault(e.name, e.role)
    return out


def check_claim(claim: dict, lines: list[str], line_speakers: list[str | None],
                doubtful: set[int], attendees: list[str]) -> NameEvidence:
    """Python's check of one LLM claim. Returns the claim with accepted/reason set."""
    speaker = str(claim.get("speaker", "")).strip().upper()
    name = " ".join(str(claim.get("name", "")).split()).strip(" .,;:!?\"'")
    kind = str(claim.get("kind", "")).strip().lower()
    quote = re.sub(r"^\s*S\d+\s*:\s*", "", str(claim.get("quote", "")).strip())
    try:
        line = int(claim.get("line"))
    except (TypeError, ValueError):
        line = -1
    ev = NameEvidence(speaker, name, kind, line, quote)

    def reject(reason: str) -> NameEvidence:
        ev.reason = reason
        return ev

    if speaker not in set(filter(None, line_speakers)):
        return reject(f"unknown speaker label {speaker!r}")
    if kind == "role":
        name = clean_role(name)
        ev.name = name
        if not role_words(name):
            return reject("not a role")
    elif not name or _norm(name) in NOT_NAMES or not re.search(r"[A-Za-z]", name):
        return reject("not a person's name")
    if kind not in ("self", "addressed", "role"):
        return reject(f"unknown kind {kind!r}")
    found = _locate(quote, line, lines, doubtful)
    if found is None:
        return reject("quote not found in the transcript")
    ev.line = found
    quote_words = set(_norm(quote).split())
    needed = role_words(name) if kind == "role" else [w for w in _norm(name).split() if len(w) >= 2]
    if kind == "role" and not all(w in quote_words for w in needed):
        return reject("the quote does not contain the role")
    if kind != "role" and not any(w in quote_words for w in needed):
        return reject("the quote does not contain the name")

    said_by = line_speakers[found]
    if kind == "role":
        if said_by != speaker:
            return reject(f"said by {said_by}, not by {speaker}")
        ev.accepted, ev.reason = True, "said their own role"
        return ev
    if kind == "self":
        if said_by != speaker:
            return reject(f"said by {said_by}, not by {speaker}")
        ev.reason = "said their own name"
    else:
        if said_by == speaker:
            return reject("a speaker cannot address themself")
        how = _neighbours(found, lines, line_speakers, doubtful).get(speaker)
        if how is None:
            return reject(f"{speaker} neither spoke just before nor answered right after")
        ev.reason = f"addressed by {said_by}; {speaker} {how}"
    ev.name, ev.accepted = _canonical(name, attendees), True
    return ev


def clean_role(role: str) -> str:
    """ "The Industrial Designer." -> "industrial designer"."""
    words = _norm(role).split()
    while words and words[0] in ROLE_FILLER:
        words.pop(0)
    return " ".join(words)


def role_words(role: str) -> list[str]:
    """The words of a role that matter ("user interface designer" -> user, interface, designer)."""
    return [w for w in _norm(role).split() if w not in ROLE_FILLER]


def role_mentioned(role: str, text: str) -> bool:
    """True if `text` refers to this role: the whole role ("the industrial designer"),
    or its distinctive word followed by a role noun ("the marketing expert" for a
    "marketing manager"). A distinctive word alone ("the project") is not enough."""
    words = _norm(text).split()
    joined = f" {' '.join(words)} "
    if role and f" {_norm(role)} " in joined:
        return True
    distinctive = [w for w in role_words(role) if w not in ROLE_NOUNS]
    for i, w in enumerate(words):
        if w in distinctive and any(x in ROLE_NOUNS for x in words[i + 1:i + 3]):
            return True
    return False


def assign_roles(speakers: list[Speaker], evidence: list[NameEvidence]) -> None:
    """Each speaker gets the role they said most often about themself, one role
    per speaker and one speaker per role. Roles the user typed are kept."""
    points: Counter = Counter()
    for e in evidence:
        if e.accepted and e.kind == "role":
            points[(e.speaker, e.name)] += 1
    for s in speakers:
        if s.role_source != "edited":
            s.role, s.role_source = "", ""
    done = {s.id for s in speakers if s.role_source == "edited"}
    taken = {s.role for s in speakers if s.role_source == "edited"}
    by_id = {s.id: s for s in speakers}
    for (sid, role), score in sorted(points.items(), key=lambda kv: (-kv[1], kv[0])):
        if sid in done or role in taken or sid not in by_id:
            continue
        if any(p == score for (s2, r2), p in points.items() if s2 == sid and r2 != role and r2 not in taken):
            done.add(sid)  # two roles equally likely: leave it to the user
            continue
        by_id[sid].role, by_id[sid].role_source = role, "self"
        done.add(sid)
        taken.add(role)
    # A role someone else gave by name ("Courtney, our marketing person") goes to
    # the speaker with that name, if they have no role of their own.
    for name, role in named_roles(evidence).items():
        sp = next((x for x in speakers if x.name and _norm(x.name) == _norm(name)), None)
        if sp and not sp.role and role not in taken:
            sp.role, sp.role_source = role, "named by others"
            taken.add(role)


def _full_names(names: list[str]) -> dict[str, str]:
    """{normalised short name: full name} when a name is the start of exactly one
    longer name heard ("Bart" -> "Bart Beute"), so the two count as one person."""
    by_norm = {_norm(n): n for n in names if _norm(n)}
    out = {}
    for short in by_norm:
        longer = [n for n in by_norm if n != short and n.startswith(short + " ")]
        if len(longer) == 1:
            out[short] = by_norm[longer[0]]
    return out


def assign_names(speakers: list[Speaker], evidence: list[NameEvidence]) -> None:
    """Score each (speaker, name) pair from the accepted evidence, then hand out
    names one to one, best first. A speaker whose two best names score the same
    stays unnamed, with a note. Names the user typed ("edited") are kept."""
    points: Counter = Counter()
    kinds: dict[tuple[str, str], set[str]] = {}
    spelling: dict[str, str] = {}
    naming = [e for e in evidence if e.accepted and e.kind in ("self", "addressed")]
    full = _full_names([e.name for e in naming])
    for e in naming:
        e.name = full.get(_norm(e.name), e.name)
        key = (e.speaker, _norm(e.name))
        points[key] += SELF_POINTS if e.kind == "self" else ADDRESSED_POINTS
        kinds.setdefault(key, set()).add(e.kind)
        spelling.setdefault(_norm(e.name), e.name)

    by_id = {s.id: s for s in speakers}
    for s in speakers:
        if s.source != "edited":
            s.name, s.source, s.confidence, s.note = "", "", "", ""
    done = {s.id for s in speakers if s.source == "edited"}
    taken = {_norm(s.name) for s in speakers if s.source == "edited"}

    for (sid, name), score in sorted(points.items(), key=lambda kv: (-kv[1], kv[0])):
        if sid in done or name in taken or sid not in by_id:
            continue
        rivals = [p for (s2, n2), p in points.items() if s2 == sid and n2 != name and n2 not in taken]
        if rivals and max(rivals) == score:
            tied = sorted(spelling[n2] for (s2, n2), p in points.items() if s2 == sid and p == score)
            by_id[sid].note = "unclear: " + " or ".join(tied)
            done.add(sid)
            continue
        others = [s2 for (s2, n2), p in points.items() if n2 == name and s2 != sid and s2 not in done and p == score]
        if others:  # the same name fits two speakers equally well
            for s2 in [sid] + others:
                by_id[s2].note = f"unclear: {spelling[name]} could be {default_label(sid)} or " \
                                 f"{' or '.join(default_label(o) for o in others)}"
            taken.add(name)
            continue
        sp = by_id[sid]
        sp.name = spelling[name]
        sp.source = "self-introduction" if "self" in kinds[(sid, name)] else "addressed"
        sp.confidence = "high" if score >= HIGH_CONFIDENCE else "low"
        done.add(sid)
        taken.add(name)


def name_speakers(lines: list[str], line_speakers: list[str | None], speakers: list[Speaker],
                  call_llm, doubtful: set[int] | None = None, attendees: list[str] | None = None,
                  max_tokens: int = 4000, prompt_path: Path = NAMES_PROMPT_PATH) -> list[NameEvidence]:
    """Stage 2b. `lines` are the (refined) transcript lines, `line_speakers` their
    speaker ids. Names the speakers in place and returns every claim with
    Python's verdict. Raises LLMError if the LLM fails or answers badly."""
    doubtful, attendees = doubtful or set(), attendees or []
    shown = [f"[{i}] {sp}: {text.strip()}" for i, (text, sp) in enumerate(zip(lines, line_speakers))
             if sp and i not in doubtful and text.strip()]
    if not speakers or not shown:
        return []
    system = prompt_path.read_text(encoding="utf-8")
    header = (f"Known attendees (the names will usually be among these): {', '.join(attendees)}\n\n"
              if attendees else "")
    claims = []
    for a, b in split_segments(shown, max_tokens):  # long meetings: one call per part
        data = parse_json_reply(call_llm(system, header + "\n".join(shown[a:b])))
        items = data.get("names")
        if not isinstance(items, list):
            raise LLMError('LLM reply is missing the "names" list')
        claims += [c for c in items if isinstance(c, dict)]

    # A self-introduction can carry a role ("I'm Nick, the industrial designer"),
    # and a role can be said without a name: each role becomes a claim of its own.
    for claim in list(claims):
        role = str(claim.get("role") or "").strip()
        if role and str(claim.get("kind", "")).strip().lower() in ("self", "role"):
            claims.append(dict(claim, name=role, kind="role"))
    claims = [c for c in claims if str(c.get("name") or "").strip()]

    evidence, seen = [], set()
    for claim in claims:
        if str(claim.get("kind", "")).strip().lower() == "assigned_role":
            ev = check_assigned_role(claim, lines, doubtful, attendees)
        else:
            ev = check_claim(claim, lines, line_speakers, doubtful, attendees)
        key = (ev.speaker, _norm(ev.name), ev.kind, ev.line, ev.role)
        if key not in seen:
            seen.add(key)
            evidence.append(ev)
    assign_names(speakers, evidence)
    assign_roles(speakers, evidence)
    return evidence


# ---------- Editing names ----------

def rename_speakers(speakers: list[Speaker], names: dict[str, str],
                    roles: dict[str, str] | None = None) -> dict[str, str]:
    """Set the names a user typed ({speaker id: name}; "" = back to "Speaker N"),
    and optionally roles ({speaker id: role}; "" = no role).
    Returns {old label: new label} for every label that changed. Giving two
    speakers the same name is allowed: it is how a user merges one person whom
    diarization split in two."""
    for s in speakers:
        if roles and s.id in roles:
            new_role = clean_role(str(roles[s.id] or ""))
            if new_role != s.role:
                s.role, s.role_source = new_role, ("edited" if new_role else "")
    changes = {}
    for s in speakers:
        if s.id not in names:
            continue
        new = " ".join(str(names[s.id]).split())
        if new == default_label(s.id):
            new = ""
        if new == s.name:
            continue
        old = s.label
        s.name, s.source, s.confidence, s.note = new, ("edited" if new else ""), "", ""
        if s.label != old:
            changes[old] = s.label
    return changes


def replace_labels(text: str, changes: dict[str, str]) -> str:
    """Replace whole-word speaker labels in one pass, so swapping two names works
    ("Speaker 2" does not match inside "Speaker 21")."""
    if not changes or not text:
        return text
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(o) for o in sorted(changes, key=len, reverse=True))
                         + r")(?!\w)")
    return pattern.sub(lambda m: changes[m.group(1)], text)


if __name__ == "__main__":
    import argparse

    from meeting_assistant.stt import AudioInputError, format_timestamp, load_model, transcribe

    parser = argparse.ArgumentParser(description="Stage 1a (and optionally 2b) on an audio file.")
    parser.add_argument("audio")
    parser.add_argument("--speakers", type=int, default=0, help="number of speakers (0 = find out)")
    parser.add_argument("--names", action="store_true", help="also ask the LLM for the speakers' names")
    parser.add_argument("--attendees", default="", help='known names, e.g. "Priya, Rahul"')
    parser.add_argument("--model-size", default="medium")
    args = parser.parse_args()

    try:
        t = transcribe(args.audio, load_model(args.model_size), model_size=args.model_size)
        diarizer = load_diarizer()
    except (AudioInputError, DiarizationError) as e:
        print(e)
        sys.exit(1)
    turns = diarizer(args.audio, args.speakers or None)
    segments = assign_speakers(t.segments, turns)
    speakers = summarize(segments)
    if args.names:
        from meeting_assistant.glossary import parse_user_terms
        from meeting_assistant.llm import call_llm
        attendees = [term.text for term in parse_user_terms(args.attendees)]
        for e in name_speakers([s.text for s in segments], [s.speaker for s in segments], speakers,
                               call_llm, attendees=attendees):
            print(f"  {'ACCEPTED' if e.accepted else 'rejected'}: {e.speaker} = {e.name} ({e.kind}, "
                  f"line {e.line}): {e.reason}")
    labels = line_labels(segments, speakers)
    for s, label in zip(segments, labels):
        print(f"[{format_timestamp(s.start)}] {label or '?'}: {s.text}")
    print(f"\n{len(speakers)} speakers on {diarizer.device}, {len(turns)} turns, "
          f"{len(segments) - len(t.segments)} lines split at speaker changes")
    for s in speakers:
        print(f"  {s.id} {s.label:20} {s.talk_s:7.1f} s  {s.lines} lines  {s.source} {s.confidence} {s.note}")
