"""Run all stages in order and save the results.

  Stage 1   stt.transcribe          audio -> raw transcript
  Stage 1b  hallucination           flag possible phantom lines
  glossary  build_glossary          typed + inferred + pack terms
  Stage 2   refine                  LLM #1 corrects domain terms
  Stage 3   minutes                 LLM #2 writes the meeting record

Each stage keeps its own status and output, so a later failure never hides an
earlier result: if Stage 2 fails, Stage 3 runs on the raw transcript. The
glossary is built (with one LLM call) as part of Stage 2 and timed with it.
Every run, including failed ones, is saved to runs/<timestamp>/.

Try it:  python -m meeting_assistant.pipeline data/audio/ES2004a_1min.wav --terms "Real Reaction"
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from meeting_assistant import llm
from meeting_assistant.glossary import Term, build_glossary, load_packs, merge_terms, parse_user_terms
from meeting_assistant.hallucination import SegmentFlag, doubtful_indices, score_segments
from meeting_assistant.minutes import MinutesResult, support_rate, to_json, to_markdown, write_minutes
from meeting_assistant.paths import RUNS_DIR
from meeting_assistant.refine import RefineResult, refine
from meeting_assistant.stt import AudioInputError, Transcript, transcribe

STAGES = ["Stage 1: speech-to-text", "Stage 1b: hallucination flags",
          "Stage 2: refinement", "Stage 3: meeting record"]


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

    @property
    def raw_text(self) -> str:
        return self.transcript.text if self.transcript else ""

    @property
    def refined_text(self) -> str:
        return self.refined.text if self.refined else self.raw_text


def run(audio_path, model, glossary_text: str = "", packs: list[str] | None = None,
        call_llm=llm.call_llm, call_structured=llm.call_llm_structured,
        model_size: str = "medium", runs_dir: Path = RUNS_DIR, on_stage=None) -> PipelineResult:
    """Run every stage on one recording. `model` is a loaded faster-whisper
    model (stt.load_model), loaded once by the caller and reused. `on_stage`,
    if given, is called with the result whenever a stage starts or ends, so a
    UI can show progress."""
    result = PipelineResult(stages=[StageStatus(name) for name in STAGES])
    stt_stage, flag_stage, refine_stage, minutes_stage = result.stages
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
        typed_terms = merge_terms(parse_user_terms(glossary_text), load_packs(packs) if packs else [])
        result.transcript = transcribe(audio_path, model, typed_terms, model_size=model_size)
    except (AudioInputError, FileNotFoundError) as e:  # bad recording, or unknown glossary pack
        stt_stage.status, stt_stage.message, stt_stage.seconds = "failed", str(e), _since(started)
        for stage in result.stages[1:]:
            stage.status, stage.message = "skipped", "no transcript"
        result.run_dir = save(result, runs_dir, llm.usage_log[usage_start:])
        finished()
        return result
    stt_stage.status, stt_stage.seconds = "done", _since(started)
    stt_stage.message = f"{len(result.transcript.segments)} segments"
    finished()
    segments = result.transcript.segments
    lines = [s.text for s in segments]

    started = start(flag_stage)
    result.flags = score_segments(segments, result.transcript.duration_s)
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

    started = start(minutes_stage)
    minutes_input = result.refined.refined if result.refined else lines
    try:
        result.minutes = write_minutes(minutes_input, call_structured, doubtful,
                                       [seg.start for seg in segments], call_text=call_llm)
        minutes_stage.status = "done"
        parts = result.minutes.checks.get("parts", 1)
        minutes_stage.message = (f"{len(result.minutes.dropped)} items removed or downgraded by the checks"
                                 + (f"; long meeting, written in {parts} topic parts" if parts > 1 else ""))
    except Exception as e:
        minutes_stage.status, minutes_stage.message = "failed", str(e)
    minutes_stage.seconds = _since(started)

    result.run_dir = save(result, runs_dir, llm.usage_log[usage_start:])
    finished()
    return result


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
    return report


def save(result: PipelineResult, runs_dir: Path, usage) -> Path:
    """Write transcripts, edit log, meeting record and run details to runs/<timestamp>/."""
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transcript_raw.txt").write_text(result.raw_text + "\n", encoding="utf-8")
    (run_dir / "transcript_refined.txt").write_text(result.refined_text + "\n", encoding="utf-8")
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
        "minutes_dropped": result.minutes.dropped if result.minutes else [],
        "faithfulness": faithfulness_report(result),
        "llm_usage": [asdict(u) for u in usage],
    }
    (run_dir / "run.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    import argparse

    from meeting_assistant.stt import load_model

    parser = argparse.ArgumentParser(description="Run the whole pipeline on an audio file.")
    parser.add_argument("audio")
    parser.add_argument("--terms", default="", help='typed glossary, e.g. "Kubeflow, ONNX"')
    parser.add_argument("--packs", default="", help='preset packs, e.g. "ml_tech"')
    parser.add_argument("--model-size", default="medium")
    args = parser.parse_args()

    packs = [p.strip() for p in args.packs.split(",") if p.strip()]
    r = run(args.audio, load_model(args.model_size), args.terms, packs, model_size=args.model_size)
    for s in r.stages:
        print(f"{s.name:32} {s.status:8} {s.seconds:6.1f}s  {s.message}")
    if r.run_dir:
        print(f"\nSaved to {r.run_dir}")
    if r.minutes:
        print("\n" + to_markdown(r.minutes.minutes))
