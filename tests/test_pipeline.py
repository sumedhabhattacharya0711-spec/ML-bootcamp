import json
import subprocess
from types import SimpleNamespace

import pytest

from meeting_assistant.llm import LLMError
from meeting_assistant.minutes import DecisionEvidence, MinutesDraft
from meeting_assistant.pipeline import run

LINES = ["We moved the training jobs to cube flow.",
         "I propose we keep it.",
         "Yes, let's keep it.",
         "Thanks for watching!"]


class FakeWhisper:
    """Stands in for the faster-whisper model: returns fixed segments."""
    def transcribe(self, path, **kwargs):
        segments = [SimpleNamespace(start=i * 2.0, end=i * 2.0 + 2, text=text, words=[],
                                    no_speech_prob=0.9 if "watching" in text else 0.05,
                                    avg_logprob=-0.3, compression_ratio=1.4)
                    for i, text in enumerate(LINES)]
        return iter(segments), SimpleNamespace(language="en")


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", "10", "-ac", "1", "-ar", "16000", str(path)], check=True)
    return path


def fake_llm(system, user):
    if '"hints"' in user:  # refine
        return json.dumps({"lines": {"0": "We moved the training jobs to Kubeflow."}, "possible_errors": []})
    return json.dumps({"terms": []})  # glossary inference


def fake_structured(system, user, schema):
    return MinutesDraft(summary="Short meeting.", minutes=["Training jobs"], action_items=[],
                        decisions=[DecisionEvidence(text="Keep it", proposal_quote="I propose we keep it",
                                                    agreement_quotes=["Yes, let's keep it"],
                                                    rejection_quotes=[])])


def failing_llm(system, user):
    raise LLMError("LLM call failed: quota exceeded")


def test_full_run_saves_everything(audio, tmp_path):
    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs")
    assert [s.status for s in r.stages] == ["done"] * 4
    assert "Kubeflow" in r.refined_text and "cube flow" in r.raw_text
    assert r.minutes.minutes.decisions[0].status == "agreed"
    for name in ["transcript_raw.txt", "transcript_refined.txt", "edit_log.json",
                 "minutes.md", "minutes.json", "run.json"]:
        assert (r.run_dir / name).exists(), name
    assert "1 of 4 segments flagged" in r.stages[1].message  # "Thanks for watching!"


def test_bad_audio_fails_stage_1_and_skips_the_rest(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    r = run(empty, FakeWhisper(), runs_dir=tmp_path / "runs")
    assert r.stages[0].status == "failed" and "empty" in r.stages[0].message
    assert [s.status for s in r.stages[1:]] == ["skipped"] * 3


def test_refine_failure_still_writes_minutes_from_raw(audio, tmp_path):
    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=failing_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs")
    assert r.stages[2].status == "failed" and "quota" in r.stages[2].message
    assert r.stages[3].status == "done"
    assert r.refined_text == r.raw_text


def test_minutes_failure_keeps_transcripts(audio, tmp_path):
    def broken(system, user, schema):
        raise LLMError("LLM answer failed validation: summary: Field required")

    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=broken,
            runs_dir=tmp_path / "runs")
    assert r.stages[3].status == "failed" and "validation" in r.stages[3].message
    assert (r.run_dir / "transcript_refined.txt").read_text().startswith("We moved the training jobs to Kubeflow.")
    assert not (r.run_dir / "minutes.md").exists()


def test_unknown_pack_fails_stage_1_instead_of_crashing(audio, tmp_path):
    r = run(audio, FakeWhisper(), packs=["no_such_pack"], runs_dir=tmp_path / "runs")
    assert r.stages[0].status == "failed"
    assert "Glossary pack not found" in r.stages[0].message and "no_such_pack" in r.stages[0].message
    assert [s.status for s in r.stages[1:]] == ["skipped"] * 3


def test_failed_run_is_saved_too(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    r = run(empty, FakeWhisper(), runs_dir=tmp_path / "runs")
    saved = json.loads((r.run_dir / "run.json").read_text())
    assert saved["stages"][0]["status"] == "failed"
