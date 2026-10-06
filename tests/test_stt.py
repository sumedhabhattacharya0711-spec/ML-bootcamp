import subprocess
from pathlib import Path

import pytest

from meeting_assistant.stt import AudioInputError, check_audio

PROJECT_DIR = Path(__file__).parent.parent
CLIP = PROJECT_DIR / "data" / "audio" / "ES2004a_1min.wav"


def make_audio(path, source, seconds):
    """Generate a small test file with ffmpeg (lavfi = a built-in signal generator)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", source,
         "-t", str(seconds), "-ac", "1", "-ar", "16000", str(path)],
        check=True,
    )
    return path


def test_empty_file(tmp_path):
    f = tmp_path / "empty.wav"
    f.touch()
    with pytest.raises(AudioInputError, match="empty"):
        check_audio(f)


def test_missing_file(tmp_path):
    with pytest.raises(AudioInputError, match="empty"):
        check_audio(tmp_path / "nope.wav")


def test_unsupported_format(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(AudioInputError, match="Unsupported format: .txt"):
        check_audio(f)


def test_corrupt_file(tmp_path):
    f = tmp_path / "broken.wav"
    f.write_bytes(b"this is not audio" * 100)
    with pytest.raises(AudioInputError, match="corrupt"):
        check_audio(f)


def test_silent_file(tmp_path):
    f = make_audio(tmp_path / "silent.wav", "anullsrc=r=16000:cl=mono", 3)
    with pytest.raises(AudioInputError, match="silent"):
        check_audio(f)


def test_too_short(tmp_path):
    f = make_audio(tmp_path / "short.wav", "sine=frequency=440", 0.3)
    with pytest.raises(AudioInputError, match="too short"):
        check_audio(f)


def test_good_file_returns_duration(tmp_path):
    f = make_audio(tmp_path / "tone.wav", "sine=frequency=440", 2)
    assert check_audio(f) == pytest.approx(2.0, abs=0.1)


@pytest.mark.slow
@pytest.mark.skipif(not CLIP.exists(), reason="1-minute clip not in data/audio")
def test_whisper_on_real_clip():
    from meeting_assistant.stt import load_model, transcribe

    model = load_model("medium")
    result = transcribe(CLIP, model)
    assert len(result.segments) > 0
    assert len(result.text.split()) > 20
    assert all(0 <= w.probability <= 1 for s in result.segments for w in s.words)
