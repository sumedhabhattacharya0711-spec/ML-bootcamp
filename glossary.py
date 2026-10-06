"""Glossary: build the per-meeting list of domain terms.

Three sources, in order of trust (see the implementation doc, Stage 2):
  1. "user"     - terms typed into the glossary box
  2. "inferred" - terms an LLM guesses from the raw transcript (verified by Python)
  3. "pack"     - optional preset word lists in data/glossary/*.txt

Stage 1 uses to_initial_prompt() to bias Whisper's spelling.
Stage 2 (refine.py) uses the full list to propose correction hints.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).parent
PACKS_DIR = PROJECT_DIR / "data" / "glossary"
INFER_PROMPT_PATH = PROJECT_DIR / "prompts" / "glossary_infer.txt"

# Lower number = more trusted. Used for ordering and for what gets dropped first.
SOURCE_PRIORITY = {"user": 0, "inferred": 1, "pack": 2}

# call_llm(system_prompt, user_message) -> reply text. Provided by llm.py later.
CallLLM = Callable[[str, str], str]


@dataclass
class Term:
    text: str
    source: str  # "user", "inferred" or "pack"
    heard_as: str = ""  # for inferred terms: the words in the transcript it was matched to


def _norm(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces. Used only for comparisons."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


# ---------- Source 1: typed by the user ----------

def parse_user_terms(text: str | None) -> list[Term]:
    """Split the glossary box on commas, semicolons or new lines."""
    if not text:
        return []
    parts = re.split(r"[,;\n]", text)
    return [Term(p.strip(), "user") for p in parts if p.strip()]


# ---------- Source 3: preset packs ----------

def list_packs(packs_dir: Path = PACKS_DIR) -> list[str]:
    """Names of available packs, e.g. ["ml_tech"]."""
    return sorted(p.stem for p in packs_dir.glob("*.txt"))


def load_packs(names: list[str], packs_dir: Path = PACKS_DIR) -> list[Term]:
    """Read one term per line. Blank lines and lines starting with # are skipped."""
    terms = []
    for name in names:
        path = packs_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Glossary pack not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(Term(line, "pack"))
    return terms


# ---------- Source 2: inferred by the LLM ----------

class GlossaryInferError(Exception):
    pass


def _parse_json_reply(reply: str) -> dict:
    """Parse the LLM reply, tolerating ```json fences or text around the object."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise GlossaryInferError("LLM reply contained no JSON object")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise GlossaryInferError(f"LLM reply was not valid JSON: {e}") from e


def _appears_in(phrase: str, transcript_norm: str) -> bool:
    """True if the phrase (or a near-exact copy of it) is in the transcript."""
    phrase = _norm(phrase)
    if not phrase:
        return False
    if re.search(rf"\b{re.escape(phrase)}\b", transcript_norm):
        return True
    # Fuzzy fallback tolerates small spelling differences, but short phrases
    # ("on x", "api") would match almost anything fuzzily, so they must be exact.
    return len(phrase) >= 6 and fuzz.partial_ratio(phrase, transcript_norm) >= 90


def transcript_for_inference(segments: list[str], doubtful: set[int] | None = None) -> str:
    """Join Whisper's segment texts into one transcript for the glossary LLM,
    leaving out segments flagged as possible hallucinations (by index, from
    hallucination.py), so a phantom line can't add a fake term. The segments
    stay in the raw transcript shown in the UI; they are only skipped here."""
    doubtful = doubtful or set()
    return " ".join(s.strip() for i, s in enumerate(segments) if i not in doubtful and s.strip())


def infer_terms(
    raw_transcript: str,
    call_llm: CallLLM,
    max_transcript_chars: int = 15000,
    prompt_path: Path = INFER_PROMPT_PATH,
) -> list[Term]:
    """Ask the LLM which domain terms the meeting is about, then keep only the
    terms whose "heard_as" text (or the term itself) really appears in the transcript.
    The LLM suggests; Python verifies. Raises GlossaryInferError on a bad reply."""
    if not raw_transcript.strip():
        return []
    system = prompt_path.read_text(encoding="utf-8")
    reply = call_llm(system, raw_transcript[:max_transcript_chars])
    data = _parse_json_reply(reply)

    items = data.get("terms")
    if not isinstance(items, list):
        raise GlossaryInferError('LLM reply is missing the "terms" list')

    transcript_norm = _norm(raw_transcript)
    terms = []
    for item in items:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        heard_as = str(item.get("heard_as", "")).strip()
        if not term:
            continue
        if _appears_in(heard_as, transcript_norm) or _appears_in(term, transcript_norm):
            terms.append(Term(term, "inferred", heard_as))
    return terms


# ---------- Merge ----------

def merge_terms(*groups: list[Term]) -> list[Term]:
    """Combine sources, most trusted first. Case-insensitive duplicates keep the
    more trusted entry (so a user's spelling beats an inferred one)."""
    all_terms = [t for group in groups for t in group]
    all_terms.sort(key=lambda t: SOURCE_PRIORITY[t.source])  # stable sort keeps input order
    seen = set()
    merged = []
    for t in all_terms:
        key = _norm(t.text)
        if key and key not in seen:
            seen.add(key)
            merged.append(t)
    return merged


def build_glossary(
    user_text: str | None = None,
    raw_transcript: str | list[str] | None = None,
    call_llm: CallLLM | None = None,
    packs: list[str] | None = None,
    doubtful: set[int] | None = None,
) -> tuple[list[Term], list[str]]:
    """Build the per-meeting glossary.

    raw_transcript is either one string or Whisper's list of segment texts.
    With a list, segments whose index is in `doubtful` (possible hallucinations)
    are left out of the text sent to the LLM and of the check on its answer.

    Returns (terms, warnings). Inference is skipped if there is no transcript
    or no call_llm. If inference fails, the warning says why and the user and
    pack terms are still returned, so Stage 2 can continue."""
    warnings = []
    user_terms = parse_user_terms(user_text)
    pack_terms = load_packs(packs) if packs else []

    if isinstance(raw_transcript, list):
        raw_transcript = transcript_for_inference(raw_transcript, doubtful)

    inferred = []
    if raw_transcript and call_llm:
        try:
            inferred = infer_terms(raw_transcript, call_llm)
        except Exception as e:  # LLM/network errors too: glossary inference is optional
            warnings.append(f"Glossary inference failed, using typed and pack terms only: {e}")

    return merge_terms(user_terms, inferred, pack_terms), warnings


# ---------- Whisper hint ----------

def to_initial_prompt(terms: list[Term], max_terms: int = 50, max_chars: int = 600) -> str:
    """Turn the glossary into Whisper's initial_prompt, e.g. "Kubeflow, ONNX, LoRA".

    Whisper keeps only ~224 tokens of prompt and silently cuts the rest, and a
    very long list can make it "hear" terms nobody said. So we keep whole terms
    only, most trusted first, and stop at max_terms or max_chars."""
    out = []
    length = 0
    for t in merge_terms(terms):
        added = len(t.text) + (2 if out else 0)  # ", " separator
        if len(out) >= max_terms or length + added > max_chars:
            break
        out.append(t.text)
        length += added
    return ", ".join(out)
