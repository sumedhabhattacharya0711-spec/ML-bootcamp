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
    return MinutesDraft(given=[], assessments=[], summary="Short meeting.", minutes=["Training jobs"], action_items=[], open_questions=[],
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
    assert json.loads((r.run_dir / "run.json").read_text())["audio"] == "tone.wav"


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
    return MinutesDraft(given=[], assessments=[],
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


def test_merging_speakers_rechecks_decisions(audio, tmp_path):
    from meeting_assistant.pipeline import rename_speakers

    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=talk_structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    assert r.minutes.minutes.decisions[0].status == "agreed"  # Rahul agreed with Priya
    rename_speakers(r, {"S2": "Priya"})  # it was all one person
    assert r.minutes.minutes.decisions[0].status == "open"  # the proposer agreeing with herself
    assert any("by the proposer (Priya) themself" in n for n in r.minutes.dropped)
    assert r.minutes.minutes.action_items[0].owner == "Priya"
    assert "**open**: Keep it" in (r.run_dir / "minutes.md").read_text()


# ---------- Deliverables ----------

DELIVERABLES = ["raw_transcript.txt", "raw_transcript.json", "refined_transcript.txt", "refined_transcript.json",
                "meeting_minutes.md", "meeting_minutes.json", "key_decisions.md", "key_decisions.json",
                "action_items.md", "action_items.json"]


def test_every_run_writes_the_ten_deliverables_and_a_zip(audio, tmp_path):
    import zipfile

    r = run(audio, FakeWhisper(), "Kubeflow", call_llm=fake_llm, call_structured=fake_structured,
            runs_dir=tmp_path / "runs")
    folder = r.run_dir / "deliverables"
    assert sorted(p.name for p in folder.iterdir()) == sorted(DELIVERABLES)
    assert sorted(zipfile.ZipFile(r.run_dir / "deliverables.zip").namelist()) == sorted(
        f"deliverables/{n}" for n in DELIVERABLES)
    assert "cube flow" in (folder / "raw_transcript.txt").read_text()
    assert "Kubeflow" in (folder / "refined_transcript.txt").read_text()
    assert json.loads((folder / "refined_transcript.json").read_text())[0]["text"] == \
        "We moved the training jobs to Kubeflow."
    decisions = json.loads((folder / "key_decisions.json").read_text())
    assert [(d["decision"], d["status"]) for d in decisions] == [("Keep it", "agreed")]
    assert "**agreed**: Keep it" in (folder / "key_decisions.md").read_text()
    assert json.loads((folder / "action_items.json").read_text()) == []  # none assigned: an empty list
    assert "No action items were assigned" in (folder / "action_items.md").read_text()
    minutes = json.loads((folder / "meeting_minutes.json").read_text())
    assert minutes["summary"] == "Short meeting." and minutes["discussion_points"] == ["Training jobs"]


def test_open_proposals_are_not_key_decisions(audio, tmp_path):
    def no_agreement(system, user, schema):
        return MinutesDraft(given=[], assessments=[], summary="Short.", minutes=[], action_items=[], open_questions=[],
                            decisions=[DecisionEvidence(text="Keep it", proposal_quote="I propose we keep it",
                                                        agreement_quotes=[], rejection_quotes=[], settled_quote="")])

    r = run(audio, FakeWhisper(), call_llm=fake_llm, call_structured=no_agreement, runs_dir=tmp_path / "runs")
    folder = r.run_dir / "deliverables"
    assert json.loads((folder / "key_decisions.json").read_text()) == []
    assert "No decisions were reached" in (folder / "key_decisions.md").read_text()
    minutes = json.loads((folder / "meeting_minutes.json").read_text())
    assert [(p["proposal"], p["status"]) for p in minutes["proposals_not_decided"]] == [("Keep it", "open")]


def test_failed_record_leaves_its_deliverables_out_and_says_why(audio, tmp_path):
    def broken(system, user, schema):
        raise LLMError("LLM call failed: rate limited")

    r = run(audio, FakeWhisper(), call_llm=fake_llm, call_structured=broken, runs_dir=tmp_path / "runs")
    names = sorted(p.name for p in (r.run_dir / "deliverables").iterdir())
    assert names == sorted(DELIVERABLES[:4] + ["STATUS.txt"])
    status = (r.run_dir / "deliverables" / "STATUS.txt").read_text()
    assert "no meeting record" in status and "rate limited" in status


def test_bad_audio_gets_only_a_status(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    r = run(empty, FakeWhisper(), runs_dir=tmp_path / "runs")
    assert [p.name for p in (r.run_dir / "deliverables").iterdir()] == ["STATUS.txt"]


def test_deliverables_carry_speakers_and_follow_renames(audio, tmp_path):
    from meeting_assistant.pipeline import rename_speakers

    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=talk_structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    folder = r.run_dir / "deliverables"
    assert "[00:06.33] Rahul: Yes, let's keep it." in (folder / "raw_transcript.txt").read_text()
    actions = json.loads((folder / "action_items.json").read_text())
    assert [(a["task"], a["owner"], a["deadline"]) for a in actions] == [("Send the report", "Rahul", "by Friday")]
    rename_speakers(r, {"S2": "Rahul Verma"})
    assert json.loads((folder / "action_items.json").read_text())[0]["owner"] == "Rahul Verma"
    assert "Rahul Verma: Yes, let's keep it." in (folder / "refined_transcript.txt").read_text()


def test_human_and_machine_files_say_the_same(audio, tmp_path):
    r = run(audio, TalkWhisper(), call_llm=talk_llm, call_structured=talk_structured,
            runs_dir=tmp_path / "runs", diarizer=talk_diarizer)
    folder = r.run_dir / "deliverables"
    for stem in ("key_decisions", "action_items"):
        md = (folder / f"{stem}.md").read_text()
        for item in json.loads((folder / f"{stem}.json").read_text()):
            text = item.get("decision") or item["task"]
            assert text in md
            assert all(e["quote"] in md for e in item["evidence"])
            if "owner" in item:
                assert f"owner: {item['owner']}" in md and f"deadline: {item['deadline']}" in md
    lines = (folder / "raw_transcript.txt").read_text().splitlines()
    rows = json.loads((folder / "raw_transcript.json").read_text())
    assert len(lines) == len(rows) and all(row["text"] in line for row, line in zip(rows, lines))
