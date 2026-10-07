"""Run all stages in order and save the results.

  Stage 1   stt.transcribe          audio -> raw transcript
  Stage 1a  speakers (diarization)  who spoke when; one speaker per line
  Stage 1b  hallucination           flag possible phantom lines
  glossary  build_glossary          typed + inferred + pack terms
  Stage 2   refine                  LLM #1 corrects domain terms
  Stage 2b  speakers (names)        LLM #1 finds names, Python checks them
  Stage 3   minutes                 LLM #2 writes the meeting record

Each stage keeps its own status and output, so a later failure never hides an
earlier result: if Stage 2 fails, Stage 3 runs on the raw transcript; if
diarization or naming fails, everything runs without (or with unnamed)
speakers. The glossary is built (with one LLM call) as part of Stage 2 and
timed with it. Every run, including failed ones, is saved to runs/<timestamp>/.
Speaker names can be edited afterwards (rename_speakers), which rewrites the
saved files without running any model again.

Try it:  python -m meeting_assistant.pipeline data/audio/ES2004a_1min.wav --terms "Real Reaction" --speakers 4
"""

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

from meeting_assistant import llm
from meeting_assistant.glossary import Term, build_glossary, load_packs, merge_terms, parse_user_terms
from meeting_assistant.hallucination import SegmentFlag, doubtful_indices, score_segments
from meeting_assistant.minutes import (MinutesResult, rename_in_minutes, support_rate, to_json, to_markdown,
                                       write_minutes)
from meeting_assistant.paths import RUNS_DIR
from meeting_assistant.refine import RefineResult, refine
from meeting_assistant.speakers import (NameEvidence, Speaker, Turn, assign_speakers, line_labels,
                                        name_speakers, summarize)
from meeting_assistant import speakers as speakers_mod
from meeting_assistant.stt import AudioInputError, Transcript, format_timestamp, transcribe

STAGES = ["Stage 1: speech-to-text", "Stage 1a: speaker diarization", "Stage 1b: hallucination flags",
          "Stage 2: refinement", "Stage 2b: speaker names", "Stage 3: meeting record"]


@dataclass
class StageStatus:
    name: str
    status: str = "waiting"   # waiting, running, done, failed or skipped
    message: str = ""
    seconds: float = 0.0


@dataclass
class PipelineResult:
    stages: list[StageStatus]
    transcript: Transcript | None = None
    flags: list[SegmentFlag] = field(default_factory=list)
    glossary: list[Term] = field(default_factory=list)
    refined: RefineResult | None = None
    minutes: MinutesResult | None = None
    run_dir: Path | None = None
    turns: list[Turn] = field(default_factory=list)            # diarization output
    speakers: list[Speaker] = field(default_factory=list)      # empty = speakers unknown
    name_evidence: list[NameEvidence] = field(default_factory=list)
    llm_usage: list = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return self.transcript.text if self.transcript else ""

    @property
    def refined_text(self) -> str:
        return self.refined.text if self.refined else self.raw_text

    @property
    def segments(self) -> list:
        return self.transcript.segments if self.transcript else []

    @property
    def roles(self) -> dict[str, str]:
        """{speaker label: stated role} for speakers with a role."""
        return {s.label: s.role for s in self.speakers if s.role}

    @property
    def labels(self) -> list[str | None]:
        """Speaker label of every line ("Priya", "Speaker 2"), or None per line."""
        return line_labels(self.segments, self.speakers) if self.speakers else [None] * len(self.segments)

    def lines(self, refined: bool = True) -> list[str]:
        if refined and self.refined:
            return self.refined.refined
        return [s.text for s in self.segments]


def run(audio_path, model, glossary_text: str = "", packs: list[str] | None = None,
        call_llm=llm.call_llm, call_structured=llm.call_llm_structured,
        model_size: str = "medium", runs_dir: Path = RUNS_DIR, on_stage=None,
        diarizer=None, num_speakers: int | None = None, attendees: str = "") -> PipelineResult:
    """Run every stage on one recording. `model` is a loaded faster-whisper
    model (stt.load_model), loaded once by the caller and reused. `on_stage`,
    if given, is called with the result whenever a stage starts or ends, so a
    UI can show progress.

    `diarizer` (speakers.load_diarizer(), or a fake in tests) is called as
    diarizer(audio_path, num_speakers) and returns speaker turns; None skips
    the speaker stages. `attendees` are known names ("Priya, Rahul"): they are
    added to the glossary (so Whisper spells them right) and help naming."""
    result = PipelineResult(stages=[StageStatus(name) for name in STAGES])
    stt_stage, diar_stage, flag_stage, refine_stage, names_stage, minutes_stage = result.stages
    usage_start = len(llm.usage_log)

    def start(stage: StageStatus) -> float:
        stage.status = "running"
        if on_stage:
            on_stage(result)
        return time.perf_counter()

    def finished() -> None:
        if on_stage:
            on_stage(result)

    started = start(stt_stage)
    try:
        # Typed and pack terms are known before Whisper runs, so they can bias its spelling.
        attendee_names = [t.text for t in parse_user_terms(attendees)]
        glossary_text = "\n".join(filter(None, [glossary_text, *attendee_names]))
        typed_terms = merge_terms(parse_user_terms(glossary_text), load_packs(packs) if packs else [])
        result.transcript = transcribe(audio_path, model, typed_terms, model_size=model_size)
    except (AudioInputError, FileNotFoundError) as e:  # bad recording, or unknown glossary pack
        stt_stage.status, stt_stage.message, stt_stage.seconds = "failed", str(e), _since(started)
        for stage in result.stages[1:]:
            stage.status, stage.message = "skipped", "no transcript"
        result.llm_usage = llm.usage_log[usage_start:]
        result.run_dir = save(result, runs_dir)
        finished()
        return result
    stt_stage.status, stt_stage.seconds = "done", _since(started)
    stt_stage.message = f"{len(result.transcript.segments)} segments"
    finished()

    started = start(diar_stage)
    if diarizer is None:
        diar_stage.status, diar_stage.message = "skipped", "speaker diarization is off"
    else:
        try:
            result.turns = diarizer(audio_path, num_speakers)
            whisper_lines = len(result.transcript.segments)
            result.transcript.segments = assign_speakers(result.transcript.segments, result.turns)
            result.speakers = summarize(result.transcript.segments)
            if not result.speakers:
                raise RuntimeError("no speech found by diarization")
            split = len(result.transcript.segments) - whisper_lines
            diar_stage.status = "done"
            diar_stage.message = (f"{len(result.speakers)} speakers"
                                  + (f"; {split} line{'s' * (split > 1)} split at speaker changes" if split else ""))
        except Exception as e:  # missing model, GPU trouble ...: carry on without speakers
            result.turns, result.speakers = [], []
            result.transcript.segments = [replace(s, speaker=None) for s in result.transcript.segments]
            diar_stage.status, diar_stage.message = "failed", f"{e}; continuing without speakers"
    diar_stage.seconds = _since(started)
    finished()
    segments = result.transcript.segments
    lines = [s.text for s in segments]

    started = start(flag_stage)
    result.flags = score_segments(segments, result.transcript.duration_s, diarized=bool(result.speakers))
    doubtful = doubtful_indices(result.flags)
    flag_stage.status, flag_stage.seconds = "done", _since(started)
    flag_stage.message = f"{len(doubtful)} of {len(segments)} segments flagged"
    finished()

    started = start(refine_stage)
    try:
        result.glossary, warnings = build_glossary(glossary_text, lines, call_llm, packs, doubtful)
        result.refined = refine(segments, result.glossary, call_llm, doubtful)
        refine_stage.status = "done"
        warnings += result.refined.warnings
        applied = sum(e.status == "applied" for e in result.refined.edits)
        refine_stage.message = "; ".join([f"{applied} edits applied"] + warnings)
    except Exception as e:  # LLMError or anything else: Stage 3 still runs on the raw text
        refine_stage.status, refine_stage.message = "failed", str(e)
    refine_stage.seconds = _since(started)
    finished()

    started = start(names_stage)
    if not result.speakers:
        names_stage.status, names_stage.message = "skipped", "speakers unknown"
    else:
        try:
            result.name_evidence = name_speakers(result.lines(), [s.speaker for s in segments], result.speakers,
                                                 call_llm, doubtful, attendee_names)
            named = [s for s in result.speakers if s.name]
            names_stage.status = "done"
            names_stage.message = (f"{len(named)} of {len(result.speakers)} speakers named"
                                   + (": " + ", ".join(s.name + (f" ({s.role})" if s.role else "")
                                                       for s in named) if named else ""))
        except Exception as e:  # LLMError or anything else: speakers stay "Speaker N"
            names_stage.status, names_stage.message = "failed", f"{e}; speakers stay unnamed"
    names_stage.seconds = _since(started)
    finished()

    started = start(minutes_stage)
    minutes_input = result.lines()
    try:
        result.minutes = write_minutes(minutes_input, call_structured, doubtful,
                                       [seg.start for seg in segments], call_text=call_llm,
                                       speakers=result.labels if result.speakers else None,
                                       roles=result.roles)
        minutes_stage.status = "done"
        parts = result.minutes.checks.get("parts", 1)
        minutes_stage.message = (f"{len(result.minutes.dropped)} items removed or downgraded by the checks"
                                 + (f"; long meeting, written in {parts} topic parts" if parts > 1 else ""))
    except Exception as e:
        minutes_stage.status, minutes_stage.message = "failed", str(e)
    minutes_stage.seconds = _since(started)

    result.llm_usage = llm.usage_log[usage_start:]
    result.run_dir = save(result, runs_dir)
    finished()
    return result


def rename_speakers(result: PipelineResult, names: dict[str, str],
                    roles: dict[str, str] | None = None) -> dict[str, str]:
    """Apply names (and roles) typed by the user ({speaker id: text}) to the
    transcripts and the meeting record, and rewrite the run's saved files. No
    model is run again. Returns {old label: new label}."""
    before = result.roles
    changes = speakers_mod.rename_speakers(result.speakers, names, roles)
    if (changes or result.roles != before) and result.minutes:
        rename_in_minutes(result.minutes.minutes, changes, result.labels, result.roles)
    if (changes or result.roles != before) and result.run_dir:
        write_run(result, result.run_dir)
    return changes


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 2)


def faithfulness_report(result: PipelineResult) -> dict:
    """Per-run counts behind the "nothing invented" rules: evidence support for
    the record, removed owners/deadlines, and the refine guard's blocks."""
    report = dict(result.minutes.checks) if result.minutes else {}
    if result.minutes:
        report["evidence_support_rate"] = support_rate(result.minutes.checks)
    if result.refined:
        edits = result.refined.edits
        report["edits_applied"] = sum(e.status == "applied" for e in edits)
        for reason in ("touches a number", "touches a negation", "changes words that were not hinted",
                       "result is not a glossary term"):
            report[f"edits_blocked: {reason}"] = sum(e.reason == reason for e in edits)
    if result.speakers:
        report["speakers_found"] = len(result.speakers)
        report["speakers_named"] = sum(bool(s.name) for s in result.speakers)
        report["speakers_with_role"] = sum(bool(s.role) for s in result.speakers)
        report["name_claims_proposed"] = len(result.name_evidence)
        report["name_claims_accepted"] = sum(e.accepted for e in result.name_evidence)
    return report


def transcript_text(result: PipelineResult, refined: bool) -> str:
    """The transcript as saved: one "[mm:ss.ss] Speaker: text" line per segment
    when speakers are known, otherwise the plain text as before."""
    if not result.speakers:
        return result.refined_text if refined else result.raw_text
    return "\n".join(f"[{format_timestamp(seg.start)}] {label or '(no speaker)'}: {line}"
                     for seg, label, line in zip(result.segments, result.labels, result.lines(refined)))


def save(result: PipelineResult, runs_dir: Path) -> Path:
    """Write everything to a new folder runs/<timestamp>/."""
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 1
    while run_dir.exists():  # two runs in the same second
        run_dir = runs_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    write_run(result, run_dir)
    return run_dir


def write_run(result: PipelineResult, run_dir: Path) -> None:
    """Write transcripts, edit log, speakers, meeting record and run details."""
    (run_dir / "transcript_raw.txt").write_text(transcript_text(result, refined=False) + "\n", encoding="utf-8")
    (run_dir / "transcript_refined.txt").write_text(transcript_text(result, refined=True) + "\n",
                                                    encoding="utf-8")
    if result.speakers:
        speakers = {"speakers": [asdict(s) | {"label": s.label} for s in result.speakers],
                    "name_evidence": [asdict(e) for e in result.name_evidence],
                    "turns": [asdict(t) for t in result.turns]}
        (run_dir / "speakers.json").write_text(json.dumps(speakers, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    if result.refined:
        log = {"edits": [asdict(e) for e in result.refined.edits],
               "possible_errors": [asdict(p) for p in result.refined.possible_errors],
               "hints": [asdict(h) for h in result.refined.hints]}
        (run_dir / "edit_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    if result.minutes:
        (run_dir / "minutes.md").write_text(to_markdown(result.minutes.minutes), encoding="utf-8")
        (run_dir / "minutes.json").write_text(to_json(result.minutes.minutes), encoding="utf-8")
    details = {
        "stages": [asdict(s) for s in result.stages],
        "transcript": asdict(result.transcript) if result.transcript else None,
        "flags": [asdict(f) for f in result.flags],
        "glossary": [asdict(t) for t in result.glossary],
        "speakers": [asdict(s) | {"label": s.label} for s in result.speakers],
        "minutes_dropped": result.minutes.dropped if result.minutes else [],
        "faithfulness": faithfulness_report(result),
        "llm_usage": [asdict(u) for u in result.llm_usage],
    }
    (run_dir / "run.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    from meeting_assistant.stt import load_model

    parser = argparse.ArgumentParser(description="Run the whole pipeline on an audio file.")
    parser.add_argument("audio")
    parser.add_argument("--terms", default="", help='typed glossary, e.g. "Kubeflow, ONNX"')
    parser.add_argument("--packs", default="", help='preset packs, e.g. "ml_tech"')
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--speakers", type=int, default=0, help="number of speakers (0 = find out)")
    parser.add_argument("--attendees", default="", help='known names, e.g. "Priya, Rahul"')
    parser.add_argument("--no-diarize", action="store_true", help="skip the speaker stages")
    args = parser.parse_args()

    diarizer = None
    if not args.no_diarize:
        from meeting_assistant.speakers import DiarizationError, load_diarizer
        try:
            diarizer = load_diarizer()
        except DiarizationError as e:
            print(f"Speaker diarization off: {e}")
    packs = [p.strip() for p in args.packs.split(",") if p.strip()]
    r = run(args.audio, load_model(args.model_size), args.terms, packs, model_size=args.model_size,
            diarizer=diarizer, num_speakers=args.speakers or None, attendees=args.attendees)
    for s in r.stages:
        print(f"{s.name:32} {s.status:8} {s.seconds:6.1f}s  {s.message}")
    if r.run_dir:
        print(f"\nSaved to {r.run_dir}")
    if r.minutes:
        print("\n" + to_markdown(r.minutes.minutes))
