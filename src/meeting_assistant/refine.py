"""Stage 2: LLM #1 corrects misheard domain terms.

Following Pusateri et al. 2024 (retrieval-augmented correction of named
entities): Python proposes hints, the LLM decides by context, and a Python
guard reverts unsafe edits.

1. Candidate spans: every 1-5 word window, never crossing a sentence end.
2. Sound-alike score = average of Metaphone (sound) similarity and spelling
   similarity. Keep matches above 72, or above 65 if Whisper was unsure of a
   word. (Our adaptation: the paper's phonetic retriever is not public.)
3. Hints: best non-overlapping match per span, at most 15.
4. LLM call with only the hinted lines (and their neighbours). No hints -> no call.
5. Guard: word-level diff of raw vs corrected. Edits touching a number or a
   negation, or not producing a glossary term, are reverted.
6. "Possible error, not changed" flags from the LLM, kept only if the words
   really are in the line. Never applied.

Try it:  python -m meeting_assistant.refine data/audio/ES2004a_1min.wav --terms "Kubeflow, ONNX" --hints-only
"""

import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import jellyfish
from rapidfuzz import fuzz

from meeting_assistant.glossary import Term
from meeting_assistant.llm import LLMError, parse_json_reply
from meeting_assistant.paths import PROMPTS_DIR

PROMPT_PATH = PROMPTS_DIR / "refine_system.txt"

# ---------- Settings (starting values; eval.py tunes them) ----------

MATCH_SCORE = 72          # average score needed for a hint
MATCH_SCORE_UNSURE = 65   # ... when Whisper was unsure of a word in the span
UNSURE_PROB = 0.5         # word.probability below this = Whisper was unsure
MAX_SPAN_WORDS = 5
MAX_HINTS = 15
GUARD_TERM_SCORE = 90     # an edit must produce (nearly) exactly a glossary term
MIN_SPAN_CHARS = 4        # single words shorter than this ("cat", "app") collide with everything
MIN_SPELLING = 60         # the letters must share something too ("kind" vs kinetic: 55)

# Everyday words that are never a misheard term on their own ("could" -> CUDA).
COMMON_WORDS = {
    "a", "about", "after", "again", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "but", "by", "can", "come", "could", "did", "do", "does", "done",
    "down", "each", "even", "for", "from", "get", "give", "go", "going", "good", "got", "had",
    "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "kind", "know", "like", "look", "make", "may", "maybe", "me", "might", "more",
    "most", "much", "must", "my", "need", "now", "of", "off", "ok", "okay", "on", "one", "only",
    "or", "other", "our", "out", "over", "really", "right", "said", "same", "say", "see", "she",
    "should", "so", "some", "something", "sort", "still", "such", "take", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "thing", "things", "think", "this",
    "those", "through", "to", "too", "up", "us", "use", "very", "want", "was", "way", "we",
    "well", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with",
    "would", "yeah", "yes", "you", "your",
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "half", "quarter", "dozen",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "thirtieth",
}
NEGATION_WORDS = {
    "not", "no", "never", "nothing", "none", "nobody", "nowhere", "neither", "nor", "cannot",
}


# ---------- Small helpers ----------

def _clean(token: str) -> str:
    """'Kubeflow,' -> 'kubeflow'. Letters and digits only, lowercase."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _ends_sentence(token: str) -> bool:
    return token.rstrip("\"')").endswith((".", "?", "!"))


def _term_text(t) -> str:
    return t.text if isinstance(t, Term) else str(t)


def _seg_text(s) -> str:
    return s if isinstance(s, str) else s.text


def _word_probs(s, n_tokens: int) -> list[float] | None:
    """Whisper's per-word confidence, if the words line up with the text tokens."""
    words = getattr(s, "words", None)
    if words and len(words) == n_tokens:
        return [w.probability for w in words]
    return None


# ---------- Steps 1-3: candidate spans, scoring, hints ----------

@dataclass
class Hint:
    line: int        # segment index
    start: int       # first token of the span
    end: int         # one past the last token
    heard: str       # the words as Whisper wrote them
    term: str        # the glossary term they might be
    score: float     # average of sound and spelling
    sound: float
    spelling: float
    unsure: bool     # Whisper was unsure of a word in the span


def score_match(heard: str, term: str) -> tuple[float, float, float]:
    """(sound, spelling, average) similarity, each 0-100.
    Spaces are removed first, so "cube flow" is compared as "cubeflow"."""
    a, b = _clean(heard), _clean(term)
    if not a or not b:
        return 0.0, 0.0, 0.0
    sound = fuzz.ratio(jellyfish.metaphone(a), jellyfish.metaphone(b))
    spelling = fuzz.ratio(a, b)
    return sound, spelling, (sound + spelling) / 2


def candidate_spans(tokens: list[str]):
    """Yield (start, end) for every 1-5 word window that doesn't cross a
    sentence end."""
    for start in range(len(tokens)):
        for end in range(start + 1, min(start + MAX_SPAN_WORDS, len(tokens)) + 1):
            yield start, end
            if _ends_sentence(tokens[end - 1]):
                break  # a longer window would cross into the next sentence


def find_hints(segments, glossary, doubtful: set[int] | None = None) -> list[Hint]:
    """Score every span of every line against every glossary term and keep the
    best non-overlapping matches, at most MAX_HINTS."""
    doubtful = doubtful or set()
    terms = [t for t in (_term_text(g) for g in glossary) if _clean(t)]
    found = []
    for line, seg in enumerate(segments):
        if line in doubtful:
            continue  # don't "correct" text that may be a hallucination
        tokens = _seg_text(seg).split()
        probs = _word_probs(seg, len(tokens))
        for start, end in candidate_spans(tokens):
            heard = " ".join(tokens[start:end])
            heard_clean = _clean(heard)
            if end - start == 1 and len(heard_clean) < MIN_SPAN_CHARS:
                continue  # short single words; short multi-word spans ("on x", "G P U") are kept
            if all(_clean(tok) in COMMON_WORDS for tok in tokens[start:end]):
                continue
            unsure = probs is not None and min(probs[start:end]) < UNSURE_PROB
            threshold = MATCH_SCORE_UNSURE if unsure else MATCH_SCORE
            for term in terms:
                term_clean = _clean(term)
                if any(term_clean in _clean(tok) for tok in tokens[start:end]):
                    continue  # one word already is the term ("in Grafana."); "G D P R" still counts
                if not 0.5 <= len(heard_clean) / len(term_clean) <= 2:
                    continue  # lengths too different to be the same word
                sound, spelling, score = score_match(heard, term)
                if score > threshold and spelling >= MIN_SPELLING:
                    found.append(Hint(line, start, end, heard, term, round(score, 1),
                                      sound, spelling, unsure))

    # Best first; take a hint only if its words aren't already used by a better one.
    found.sort(key=lambda h: -h.score)
    chosen, used = [], set()
    for h in found:
        span = {(h.line, i) for i in range(h.start, h.end)}
        if span & used:
            continue
        chosen.append(h)
        used |= span
        if len(chosen) == MAX_HINTS:
            break
    return sorted(chosen, key=lambda h: (h.line, h.start))


# ---------- Step 5: the guard ----------

@dataclass
class Edit:
    line: int
    before: str
    after: str
    status: str   # "applied" or "blocked"
    reason: str


def _block_reason(before: str, after: str, terms: list[str]) -> str | None:
    """Why this edit must be reverted, or None if it is safe."""
    words = before.split() + after.split()
    cleaned = {_clean(w) for w in words}
    if any(ch.isdigit() for ch in before + after) or cleaned & NUMBER_WORDS:
        return "touches a number"
    if cleaned & NEGATION_WORDS or any(w.lower().rstrip(".,!?").endswith("n't") for w in words):
        return "touches a negation"
    if not after.strip():
        return "deletes words"
    after_clean = _clean(after)
    if not any(fuzz.ratio(after_clean, _clean(t)) >= GUARD_TERM_SCORE for t in terms):
        return "result is not a glossary term"
    return None


def guard_line(line: int, raw: str, corrected: str, terms: list[str]) -> tuple[str, list[Edit]]:
    """Diff raw vs corrected word by word. Keep safe edits, revert the rest.
    Words are compared without punctuation, so punctuation-only changes are ignored."""
    raw_toks, new_toks = raw.split(), corrected.split()
    matcher = SequenceMatcher(a=[_clean(t) for t in raw_toks],
                              b=[_clean(t) for t in new_toks], autojunk=False)
    out, edits = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out += raw_toks[i1:i2]
            continue
        before, after = " ".join(raw_toks[i1:i2]), " ".join(new_toks[j1:j2])
        reason = _block_reason(before, after, terms)
        if reason:
            out += raw_toks[i1:i2]
            edits.append(Edit(line, before, after, "blocked", reason))
        else:
            out += new_toks[j1:j2]
            edits.append(Edit(line, before, after, "applied", "matches a glossary term"))
    return " ".join(out), edits


# ---------- Step 4 + 6: the LLM call and possible-error flags ----------

@dataclass
class PossibleError:
    line: int
    text: str
    reason: str


@dataclass
class RefineResult:
    refined: list[str]                 # one corrected line per segment
    edits: list[Edit] = field(default_factory=list)
    possible_errors: list[PossibleError] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_called: bool = False

    @property
    def text(self) -> str:
        return " ".join(line for line in self.refined if line)


def build_llm_input(lines: list[str], hints: list[Hint]) -> str:
    """Only the hinted lines and their neighbours (for context), plus the hints."""
    wanted = set()
    for h in hints:
        wanted |= {h.line - 1, h.line, h.line + 1}
    shown = {str(i): lines[i] for i in sorted(wanted) if 0 <= i < len(lines)}
    payload = {
        "lines": shown,
        "hints": [{"line": h.line, "heard": h.heard, "term": h.term} for h in hints],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def refine(segments, glossary, call_llm=None, doubtful: set[int] | None = None,
           prompt_path: Path = PROMPT_PATH) -> RefineResult:
    """Stage 2. `segments` are stt.Segment objects (or plain strings);
    `glossary` is a list of glossary.Term (or strings); `call_llm` is
    llm.call_llm (or a fake in tests)."""
    lines = [_seg_text(s).strip() for s in segments]
    result = RefineResult(refined=list(lines))
    terms = [_term_text(g) for g in glossary]

    result.hints = find_hints(segments, glossary, doubtful)
    if not result.hints:
        return result  # nothing to fix: the LLM is skipped
    if call_llm is None:
        result.warnings.append("No LLM configured: hints found but not applied.")
        return result

    system = prompt_path.read_text(encoding="utf-8")
    reply = call_llm(system, build_llm_input(lines, result.hints))
    result.llm_called = True
    try:
        data = parse_json_reply(reply)
    except LLMError as e:
        result.warnings.append(f"Refinement skipped, transcript kept as is: {e}")
        return result

    changed = data.get("lines") or {}
    if not isinstance(changed, dict):
        result.warnings.append('Refinement skipped: LLM reply "lines" was not an object.')
        changed = {}
    for key, corrected in changed.items():
        try:
            i = int(key)
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(lines) or not isinstance(corrected, str):
            continue
        result.refined[i], edits = guard_line(i, lines[i], corrected, terms)
        result.edits += edits

    for item in data.get("possible_errors") or []:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        words = " ".join(_clean(w) for w in text.split())
        line_words = " ".join(_clean(w) for w in lines[i].split()) if 0 <= i < len(lines) else ""
        if words and f" {words} " in f" {line_words} ":  # really in the raw line
            result.possible_errors.append(PossibleError(i, text, str(item.get("reason", ""))))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 on an audio file.")
    parser.add_argument("audio")
    parser.add_argument("--terms", default="", help='typed glossary, e.g. "Kubeflow, ONNX"')
    parser.add_argument("--packs", default="", help='preset packs, e.g. "ml_tech"')
    parser.add_argument("--hints-only", action="store_true", help="show hints, don't call the LLM")
    parser.add_argument("--model-size", default="medium")
    args = parser.parse_args()

    from meeting_assistant.glossary import build_glossary
    from meeting_assistant.hallucination import doubtful_indices, score_segments
    from meeting_assistant.stt import AudioInputError, load_model, transcribe

    packs = [p.strip() for p in args.packs.split(",") if p.strip()]
    terms, _ = build_glossary(args.terms, packs=packs)
    try:
        t = transcribe(args.audio, load_model(args.model_size), terms, model_size=args.model_size)
    except AudioInputError as e:
        print(f"Stage 1 failed: {e}")
        sys.exit(1)
    doubtful = doubtful_indices(score_segments(t.segments, t.duration_s))

    call = None
    if not args.hints_only:
        from meeting_assistant.llm import LLMError, call_llm
        call = call_llm
    try:
        r = refine(t.segments, terms, call, doubtful)
    except Exception as e:  # LLMError: show it as a stage failure
        print(f"Stage 2 failed: {e}")
        sys.exit(1)

    print(f"{len(r.hints)} hints:")
    for h in r.hints:
        flag = " (Whisper unsure)" if h.unsure else ""
        print(f"  line {h.line}: {h.heard!r} -> {h.term}  score {h.score} "
              f"(sound {h.sound:.0f}, spelling {h.spelling:.0f}){flag}")
    for e in r.edits:
        print(f"  {e.status.upper()}: line {e.line}: {e.before!r} -> {e.after!r} ({e.reason})")
    for p in r.possible_errors:
        print(f"  POSSIBLE ERROR, not changed: line {p.line}: {p.text!r} ({p.reason})")
    for w in r.warnings:
        print(f"  WARNING: {w}")
