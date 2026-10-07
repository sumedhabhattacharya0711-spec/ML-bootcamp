"""Stage 1: speech-to-text.

check_audio()  - reject empty, unsupported, unreadable, too short or silent files
                 with a clear message, before Whisper runs.
load_model()   - load faster-whisper once (GPU if available, else CPU).
transcribe()   - run Whisper with VAD and word timestamps; return the raw
                 transcript with per-segment and per-word confidences.

Try it:  python -m meeting_assistant.stt data/audio/ES2004a_1min.wav
"""

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from meeting_assistant.glossary import Term, to_hotwords

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MIN_DURATION_S = 1.0
SILENCE_MAX_DB = -50.0  # loudest moment quieter than this = silent recording


class AudioInputError(Exception):
    """A problem with the uploaded file. The message is shown to the user as is."""


# ---------- Input checks ----------

def _probe(path: Path) -> dict:
    """Ask ffprobe for the file's format and streams."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AudioInputError("The file is corrupt or unreadable as audio.")
    return json.loads(result.stdout or "{}")


def _max_volume_db(path: Path) -> float:
    """Loudest moment in the file, in dB (0 = maximum, -91 = digital silence)."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", result.stderr)
    if not match:
        raise AudioInputError("The file is corrupt or unreadable as audio.")
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def check_audio(path: str | Path) -> float:
    """Raise AudioInputError with a user-facing message if the file can't be
    transcribed. Returns the duration in seconds if it's fine."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise AudioInputError("The file is empty.")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS))
        raise AudioInputError(f"Unsupported format: {ext or 'no extension'}. Use one of: {allowed}.")

    info = _probe(path)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise AudioInputError("The file has no audio track.")

    try:
        duration = float(info["format"]["duration"])
    except (KeyError, ValueError):
        raise AudioInputError("The file is corrupt or unreadable as audio.")
    if duration < MIN_DURATION_S:
        raise AudioInputError(f"The recording is too short ({duration:.1f} s).")

    if _max_volume_db(path) < SILENCE_MAX_DB:
        raise AudioInputError("The recording appears to be silent.")
    return duration


# ---------- Model ----------

def _preload_cuda_libs() -> None:
    """faster-whisper needs cuBLAS and cuDNN. They are installed as pip packages
    (nvidia-cublas-cu12, nvidia-cudnn-cu12) inside .venv, where Linux doesn't look
    for them, so we load them by full path first."""
    import ctypes
    import importlib.util

    for package, names in [
        ("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
        ("nvidia.cudnn", ["libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"]),
    ]:
        spec = importlib.util.find_spec(package)
        if not spec or not spec.submodule_search_locations:
            continue
        lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        for name in names:
            if (lib_dir / name).exists():
                ctypes.CDLL(str(lib_dir / name), mode=ctypes.RTLD_GLOBAL)


def load_model(size: str = "medium"):
    """Load faster-whisper once. GPU with float16 if available, else CPU int8."""
    import ctranslate2
    from faster_whisper import WhisperModel

    if ctranslate2.get_cuda_device_count() > 0:
        try:
            _preload_cuda_libs()
            return WhisperModel(size, device="cuda", compute_type="float16")
        except Exception as e:  # e.g. missing CUDA libraries or out of GPU memory
            print(f"GPU load failed ({e}); falling back to CPU.", file=sys.stderr)
    return WhisperModel(size, device="cpu", compute_type="int8")


# ---------- Transcription ----------

@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    segments: list[Segment]
    language: str
    duration_s: float
    model_size: str
    transcribe_s: float
    hotwords: str

    def to_dict(self) -> dict:
        return asdict(self)


def transcribe(path: str | Path, model, glossary_terms: list[Term] | None = None,
               model_size: str = "medium") -> Transcript:
    """Check the file, then transcribe it. Raises AudioInputError for bad input."""
    duration = check_audio(path)
    hotwords = to_hotwords(glossary_terms) if glossary_terms else ""

    started = time.perf_counter()
    raw_segments, info = model.transcribe(
        str(path),
        language="en",
        vad_filter=True,                   # skip silence, where Whisper invents text
        word_timestamps=True,              # per-word times and confidences
        # hotwords, not initial_prompt: with condition_on_previous_text=False,
        # initial_prompt only reaches the first 30-second window; hotwords reach all.
        hotwords=hotwords or None,
        condition_on_previous_text=False,  # stops one chunk's text looping into the next
    )
    segments = []
    # float(): faster-whisper returns numpy floats for times and word probabilities;
    # plain floats keep comparisons and the saved JSON simple.
    for s in raw_segments:  # a generator: decoding happens while we iterate
        words = [Word(float(w.start), float(w.end), w.word.strip(), float(w.probability))
                 for w in (s.words or [])]
        segments.append(Segment(float(s.start), float(s.end), s.text.strip(), s.no_speech_prob,
                                s.avg_logprob, s.compression_ratio, words))
    elapsed = time.perf_counter() - started

    return Transcript(
        text=" ".join(s.text for s in segments if s.text),
        segments=segments,
        language=info.language,
        duration_s=duration,
        model_size=model_size,
        transcribe_s=round(elapsed, 2),
        hotwords=hotwords,
    )


def format_timestamp(t: float) -> str:
    """83.5 -> "01:23.50"."""
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m meeting_assistant.stt <audio file> [model size]")
        sys.exit(1)
    audio, size = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "medium")
    try:
        check_audio(audio)
    except AudioInputError as e:
        print(f"Stage 1 failed: {e}")
        sys.exit(1)

    t0 = time.perf_counter()
    model = load_model(size)
    print(f"Loaded {size} in {time.perf_counter() - t0:.1f} s")

    result = transcribe(audio, model, model_size=size)
    for s in result.segments:
        print(f"[{format_timestamp(s.start)} - {format_timestamp(s.end)}] {s.text}")
    print(f"\n{len(result.segments)} segments, {result.duration_s:.0f} s of audio "
          f"transcribed in {result.transcribe_s:.1f} s")
