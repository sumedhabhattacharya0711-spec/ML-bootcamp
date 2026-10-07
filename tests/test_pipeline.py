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
    return MinutesDraft(summary="Short meeting.", minutes=["Training jobs"], action_items=[], open_questions=[],
                        decisions=[DecisionEvidence(text="Keep it", proposal_quote="I propose we keep it",
                                                    agreement_quotes=["Yes, let's keep it"],
                                                    rejection_quotes=[], settled_quote="")])


def failing_llm(system, user):
    raise LLMError("LLM call failed: quota exceeded")


def test_full_run_saves_everything(audio, tmp_path):
    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs")
    assert [s.status for s in r.stages] == ["done", "skipped", "done", "done", "skipped", "done"]
    assert "Kubeflow" in r.refined_text and "cube flow" in r.raw_text
    assert r.minutes.minutes.decisions[0].status == "agreed"
    for name in ["transcript_raw.txt", "transcript_refined.txt", "edit_log.json",
                 "minutes.md", "minutes.json", "run.json"]:
        assert (r.run_dir / name).exists(), name
    assert "1 of 4 segments flagged" in r.stages[2].message  # "Thanks for watching!"


def test_bad_audio_fails_stage_1_and_skips_the_rest(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    r = run(empty, FakeWhisper(), runs_dir=tmp_path / "runs")
    assert r.stages[0].status == "failed" and "empty" in r.stages[0].message
    assert [s.status for s in r.stages[1:]] == ["skipped"] * 5


def test_refine_failure_still_writes_minutes_from_raw(audio, tmp_path):
    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=failing_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs")
    assert r.stages[3].status == "failed" and "quota" in r.stages[3].message
    assert r.stages[5].status == "done"
    assert r.refined_text == r.raw_text


def test_minutes_failure_keeps_transcripts(audio, tmp_path):
    def broken(system, user, schema):
        raise LLMError("LLM answer failed validation: summary: Field required")

    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=broken,
            runs_dir=tmp_path / "runs")
    assert r.stages[5].status == "failed" and "validation" in r.stages[5].message
    assert (r.run_dir / "transcript_refined.txt").read_text().startswith("We moved the training jobs to Kubeflow.")
    assert not (r.run_dir / "minutes.md").exists()


def test_unknown_pack_fails_stage_1_instead_of_crashing(audio, tmp_path):
    r = run(audio, FakeWhisper(), packs=["no_such_pack"], runs_dir=tmp_path / "runs")
    assert r.stages[0].status == "failed"
    assert "Glossary pack not found" in r.stages[0].message and "no_such_pack" in r.stages[0].message
    assert [s.status for s in r.stages[1:]] == ["skipped"] * 5


def test_failed_run_is_saved_too(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    r = run(empty, FakeWhisper(), runs_dir=tmp_path / "runs")
    saved = json.loads((r.run_dir / "run.json").read_text())
    assert saved["stages"][0]["status"] == "failed"


def test_on_stage_reports_progress(audio, tmp_path):
    seen = []
    run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=fake_structured,
        runs_dir=tmp_path / "runs", on_stage=lambda r: seen.append([s.status for s in r.stages]))
    assert seen[0] == ["running"] + ["waiting"] * 5
    assert ["done", "skipped", "done", "running", "waiting", "waiting"] in seen
    assert seen[-1] == ["done", "skipped", "done", "done", "skipped", "done"]


# ---------- With speakers ----------

from meeting_assistant.speakers import Turn  # noqa: E402
from meeting_assistant.minutes import ActionItemEvidence  # noqa: E402

TALK = [(0.0, 3.0, "Hi, I'm Priya. Let's start."),
        (3.0, 9.0, "I propose we keep it. Yes, let's keep it."),   # two speakers in one Whisper line
        (9.0, 11.0, "Rahul, can you send the report?"),
        (11.0, 14.0, "Sure, I'll send it by Friday.")]


class TalkWhisper:
    def transcribe(self, path, **kwargs):
        segments = []
        for start, end, text in TALK:
            tokens = text.split()
            step = (end - start) / len(tokens)
            ws = [SimpleNamespace(start=start + i * step, end=start + (i + 1) * step, word=" " + t,
                                  probability=0.9) for i, t in enumerate(tokens)]
            segments.append(SimpleNamespace(start=start, end=end, text=text, words=ws, no_speech_prob=0.05,
                                            avg_logprob=-0.3, compression_ratio=1.4))
        return iter(segments), SimpleNamespace(language="en")


def talk_diarizer(path, num_speakers=None):
    return [Turn(0, 5.5, "S1"), Turn(5.5, 9, "S2"), Turn(9, 11, "S1"), Turn(11, 14, "S2")]


def talk_llm(system, user):
    if "names of the speakers" in system:
        return json.dumps({"names": [
            {"speaker": "S1", "name": "Priya", "kind": "self", "line": 0, "quote": "I'm Priya"},
            {"speaker": "S2", "name": "Rahul", "kind": "addressed", "line": 3, "quote": "Rahul, can you send"}]})
    if '"hints"' in user:
        return json.dumps({"lines": {}, "possible_errors": []})
    return json.dumps({"terms": []})


def talk_structured(system, user, schema):
    assert "[2] Speaker 2: Yes, let's keep it." not in user  # names are known by Stage 3
    assert "[2] Rahul: Yes, let's keep it." in user
    return MinutesDraft(
        summary="Rahul will send the report.", minutes=["Keeping it"], open_questions=[],
        decisions=[DecisionEvidence(text="Keep it", proposal_quote="I propose we keep it",
                                    agreement_quotes=["Yes, let's keep it"], rejection_quotes=[],
                                    settled_quote="")],
        action_items=[ActionItemEvidence(task="Send the report", task_quote="can you send the report",
                                         owner="unspecified", owner_quote="", deadline="by Friday",
                                         deadline_quote="I'll send it by Friday",
                                         agreement_quote="Sure, I'll send it by Friday")])


def test_speakers_are_found_named_and_used_in_the_record(audio, tmp_path):
    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=talk_structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    assert [s.status for s in r.stages] == ["done"] * 6
    assert "1 line split" in r.stages[1].message and "Priya, Rahul" in r.stages[4].message
    assert r.labels == ["Priya", "Priya", "Rahul", "Priya", "Rahul"]
    m = r.minutes.minutes
    assert m.decisions[0].status == "agreed" and m.participants == ["Priya", "Rahul"]
    assert (m.action_items[0].owner, m.action_items[0].status) == ("Rahul", "agreed")
    saved = (r.run_dir / "transcript_raw.txt").read_text()
    assert "[00:06.33] Rahul: Yes, let's keep it." in saved
    assert (r.run_dir / "speakers.json").exists()


def test_renaming_rewrites_the_record_and_files(audio, tmp_path):
    from meeting_assistant.pipeline import rename_speakers

    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=talk_structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    assert rename_speakers(r, {"S2": "Rahul Verma"}) == {"Rahul": "Rahul Verma"}
    assert r.minutes.minutes.action_items[0].owner == "Rahul Verma"
    assert "Rahul Verma will send the report." in (r.run_dir / "minutes.md").read_text()
    assert "Rahul Verma: Sure, I'll send it by Friday." in (r.run_dir / "transcript_refined.txt").read_text()
    saved = json.loads((r.run_dir / "speakers.json").read_text())
    assert saved["speakers"][1]["name"] == "Rahul Verma" and saved["speakers"][1]["source"] == "edited"


def test_diarization_failure_runs_without_speakers(audio, tmp_path):
    def broken(path, num_speakers=None):
        raise RuntimeError("CUDA out of memory")

    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs", diarizer=broken)
    assert r.stages[1].status == "failed" and "out of memory" in r.stages[1].message
    assert r.stages[4].status == "skipped" and r.stages[5].status == "done"
    assert r.speakers == [] and all(s.speaker is None for s in r.segments)


def test_naming_failure_keeps_numbered_speakers(audio, tmp_path):
    def no_names(system, user):
        if "names of the speakers" in system:
            raise LLMError("LLM call failed: rate limited")
        return talk_llm(system, user)

    def structured(system, user, schema):
        assert "[0] Speaker 1: Hi, I'm Priya." in user
        return fake_structured(system, user, schema)

    r = run(audio, TalkWhisper(), call_llm=no_names, call_structured=structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    assert r.stages[4].status == "failed" and "unnamed" in r.stages[4].message
    assert r.labels[0] == "Speaker 1" and r.stages[5].status == "done"


def test_role_of_a_person_without_a_voice_still_names_the_owner():
    from meeting_assistant.pipeline import PipelineResult
    from meeting_assistant.speakers import NameEvidence, Speaker

    r = PipelineResult(stages=[], speakers=[Speaker("S1", "Mandy", role="project manager", role_source="self")],
                       name_evidence=[NameEvidence("", "Courtney", "assigned_role", 3, "the marketing person, Courtney",
                                                   accepted=True, role="marketing person"),
                                      NameEvidence("", "Bob", "assigned_role", 4, "Bob, project manager",
                                                   accepted=True, role="project manager")])
    assert r.roles == {"Mandy": "project manager", "Courtney": "marketing person"}  # one person per role
