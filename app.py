"""Gradio front end for the meeting assistant (localhost only).

  python app.py                      # http://127.0.0.1:7860
  python app.py --model-size turbo   # start with another Whisper size

The Whisper model is loaded once at startup and kept in memory; choosing a
different size in the dropdown swaps it on the next Run. Everything else is
pipeline.run(), which does all the work and saves each run to runs/.
"""

import argparse
import gc
import json
import logging
import os
import queue
import threading
from pathlib import Path

import gradio as gr

from meeting_assistant import llm
from meeting_assistant.glossary import list_packs
from meeting_assistant.minutes import to_markdown
from meeting_assistant.pipeline import STAGES, PipelineResult, run
from meeting_assistant.refine import UNSURE_PROB
from meeting_assistant.stt import SUPPORTED_EXTENSIONS, format_timestamp, load_model

log = logging.getLogger("meeting_assistant.app")

# Figures for an RTX 4050 (6 GB); only medium is measured on this machine.
MODEL_CHOICES = [
    ("small: fastest (~20-30 s per 17-min meeting), ~1 GB GPU, noticeably more mistakes", "small"),
    ("medium: default; ~50 s per 17-min meeting, ~2-3 GB GPU, good accuracy, thresholds tuned on it", "medium"),
    ("turbo: ~40-60 s, ~2-3 GB GPU, close to large-v3 accuracy; first use downloads ~1.6 GB", "turbo"),
    ("large-v3: most accurate but slowest (~1.5-2.5 min), ~4-5 GB GPU, may not fit in 6 GB "
     "(then falls back to CPU, very slow), invents text over silence more often", "large-v3"),
]
STATUS_HEADERS = ["Stage", "Status", "Time (s)", "Message"]

SANS = ("Inter", "Helvetica Neue", "Segoe UI", "Arial", "sans-serif")


def _light_theme() -> gr.themes.Base:
    """Minimal monochrome theme: white page, near-black text, thin grey rules."""
    theme = gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="neutral",
        font=[gr.themes.Font(f) for f in SANS],
        font_mono=[gr.themes.Font(f) for f in SANS],  # tables read better in the body font
        radius_size="sm",
        spacing_size="md",
    ).set(
        body_background_fill="#ffffff",
        body_text_color="#111111",
        body_text_color_subdued="#6b6b6b",
        background_fill_primary="#ffffff",
        background_fill_secondary="#fafafa",
        block_background_fill="#ffffff",
        block_border_color="#e5e5e5",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#ffffff",
        block_label_text_color="#6b6b6b",
        block_title_text_color="#111111",
        border_color_primary="#e5e5e5",
        input_background_fill="#ffffff",
        input_border_color="#e5e5e5",
        input_shadow="none",
        button_primary_background_fill="#111111",
        button_primary_background_fill_hover="#333333",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#111111",
        table_even_background_fill="#ffffff",
        table_odd_background_fill="#fafafa",
        table_border_color="#e5e5e5",
    )
    # Always light: give every dark-mode variable its light value.
    for name in vars(theme).copy():
        if name.endswith("_dark"):
            setattr(theme, name, getattr(theme, name[:-len("_dark")]))
    return theme


THEME = _light_theme()
CSS = """
.page-header {padding: 8px 0 4px 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 8px;}
.page-header .kicker {color: #6b6b6b; letter-spacing: 0.08em; font-size: 0.85rem; text-transform: uppercase;}
.page-header h1 {font-family: "CMU Serif", "Latin Modern Roman", Georgia, "Times New Roman", serif;
                 font-weight: 700; font-size: 2.6rem; margin: 2px 0 8px 0; color: #111111;}
.page-header p {color: #333333; font-size: 1.05rem; max-width: 900px; margin: 0 0 12px 0;}
"""
HEADER_HTML = """
<div class="page-header">
  <div class="kicker">Local meeting transcription</div>
  <h1>Meeting Assistant</h1>
  <p>Recording &rarr; raw transcript &rarr; corrected transcript &rarr; meeting record.
     Speech-to-text runs on this machine; the two LLM steps use the provider set in <code>.env</code>.</p>
</div>
"""
SAVED_FILES = ["minutes.md", "minutes.json", "transcript_raw.txt", "transcript_refined.txt",
               "edit_log.json", "run.json"]


class WhisperModels:
    """Keeps exactly one Whisper model in memory and swaps it when another size is asked for."""

    def __init__(self):
        self.size, self.model = None, None
        self.lock = threading.Lock()

    def get(self, size: str):
        with self.lock:
            if size != self.size:
                self.model = None
                gc.collect()  # free the old model's GPU memory before loading the new one
                self.model, self.size = load_model(size), size
            return self.model

    def device(self) -> str:
        return f"{self.model.model.device} ({self.model.model.compute_type})" if self.model else "not loaded"


models = WhisperModels()


# ---------- Formatting results for the UI ----------

def status_rows(result: PipelineResult) -> list[list]:
    return [[s.name, s.status, s.seconds, s.message] for s in result.stages]


def run_header(result: PipelineResult, audio_name: str, glossary_text: str, packs: list[str],
               size: str) -> str:
    """What this result was produced from, so it can't be mistaken for the current inputs."""
    lines = [f"**File:** {audio_name} · **Whisper:** {size} on {models.device()} · "
             f"**LLM:** {_llm_description()}"]
    if result.transcript:
        t = result.transcript
        lines.append(f"**Audio:** {format_timestamp(t.duration_s)} (mm:ss) · **Language:** {t.language} · "
                     f"**Transcription:** {t.transcribe_s:.1f} s · **Segments:** {len(t.segments)}")
        lines.append(f"**Whisper hotwords:** {t.hotwords or '(none)'}")
    lines.append(f"**Glossary typed:** {glossary_text.strip() or '(none)'} · "
                 f"**Packs:** {', '.join(packs) or '(none)'}")
    if result.run_dir:
        lines.append(f"**Saved to:** `{result.run_dir}`")
    return "  \n".join(lines)


def _llm_description() -> str:
    text = f"{llm.PRIMARY.name} {llm.PRIMARY.model}"
    if os.getenv(llm.BACKUP.key_env, "").strip():
        text += f", backup {llm.BACKUP.name} {llm.BACKUP.model}"
    return text


def raw_highlights(result: PipelineResult) -> list[tuple[str, str | None]]:
    """Raw transcript with low-confidence words and possible hallucinations marked."""
    flagged = {f.index: f for f in result.flags if f.flagged}
    out = []
    for i, seg in enumerate(result.transcript.segments):
        out.append((f"[{format_timestamp(seg.start)}] ", None))
        if i in flagged:
            out.append((seg.text, f"possible hallucination: {flagged[i].reason_text}"))
        elif seg.words:
            for w in seg.words:
                label = f"low confidence (p<{UNSURE_PROB})" if w.probability < UNSURE_PROB else None
                out.append((w.text + " ", label))
        else:
            out.append((seg.text, None))
        out.append(("\n", None))
    return out


def refined_highlights(result: PipelineResult) -> list[tuple[str, str | None]]:
    """Refined transcript with the applied edits marked."""
    applied = {}
    for e in result.refined.edits if result.refined else []:
        if e.status == "applied":
            applied.setdefault(e.line, []).append(e.after)
    lines = result.refined.refined if result.refined else [s.text for s in result.transcript.segments]
    out = []
    for i, (seg, line) in enumerate(zip(result.transcript.segments, lines)):
        out.append((f"[{format_timestamp(seg.start)}] ", None))
        rest = line
        for after in applied.get(i, []):
            before, found, rest = rest.partition(after)
            if not found:
                rest = before
                continue
            out += [(before, None), (after, "edited")]
        out += [(rest, None), ("\n", None)]
    return out


def segment_rows(result: PipelineResult) -> list[list]:
    flags = {f.index: f for f in result.flags}
    rows = []
    for i, s in enumerate(result.transcript.segments):
        f = flags.get(i)
        rows.append([i, format_timestamp(s.start), format_timestamp(s.end), s.text,
                     round(s.no_speech_prob, 3), round(s.avg_logprob, 3), round(s.compression_ratio, 2),
                     f.score if f else 0, f.flagged if f else False, f.reason_text if f else ""])
    return rows


def details(result: PipelineResult, saved: dict) -> tuple:
    r = result.refined
    edits = [[e.line, e.before, e.after, e.status, e.reason] for e in r.edits] if r else []
    hints = ([[h.line, h.heard, h.term, h.score, round(h.sound, 1), round(h.spelling, 1), h.unsure]
              for h in r.hints] if r else [])
    possible = [[p.line, p.text, p.reason] for p in r.possible_errors] if r else []
    glossary = [[t.text, t.source, t.heard_as] for t in result.glossary]
    usage = [[u["provider"], u["model"], u["input_tokens"], u["output_tokens"], u["seconds"]]
             for u in saved.get("llm_usage", [])]
    return segment_rows(result), edits, hints, possible, glossary, usage


FAITHFULNESS_LABELS = {
    "evidence_support_rate": "Evidence support rate (decisions, action items and open questions kept with verified quotes / proposed)",
    "decisions_proposed": "Decisions proposed by the LLM",
    "decisions_kept": "Decisions kept (proposal quote verified)",
    "decisions_agreed": "Decisions marked agreed (verified agreement quote)",
    "actions_proposed": "Action items proposed by the LLM",
    "actions_kept": "Action items kept (task quote verified)",
    "actions_agreed": "Action items agreed (owner + verified acceptance quote)",
    "open_questions_proposed": "Open questions proposed by the LLM",
    "open_questions_kept": "Open questions kept (quote verified)",
    "owners_removed": "Owners not supported by a quote, set to unspecified",
    "deadlines_removed": "Deadlines not supported by a quote, set to unspecified",
    "quotes_ignored": "Agreement/rejection quotes ignored (backchannel or not found)",
    "edits_applied": "Refinement edits applied",
}


def faithfulness_rows(saved: dict) -> list[list]:
    """The run's faithfulness counts from run.json, with readable labels."""
    rows = []
    for key, value in saved.get("faithfulness", {}).items():
        label = FAITHFULNESS_LABELS.get(key, key.replace("edits_blocked: ", "Refinement edits blocked: "))
        if key == "evidence_support_rate" and value is not None:
            value = f"{value:.1%}"
        rows.append([label, value])
    return rows


def minutes_view(result: PipelineResult) -> tuple[str, str]:
    if not result.minutes:
        return "_No meeting record: see the Stage 3 status above._", ""
    dropped = result.minutes.dropped
    notes = "\n".join(f"- {d}" for d in dropped) if dropped else "_Nothing removed._"
    return to_markdown(result.minutes.minutes), notes


# ---------- The Run handler ----------

def run_meeting(audio_file, glossary_text, packs, size):
    """Generator: validates, runs the pipeline in a worker thread, and yields
    the status table as each stage starts and ends, then all results."""
    # Old results are cleared when a run starts, so nothing stale stays on screen;
    # progress updates then leave the other outputs untouched.
    cleared = ["", "", "", [], [], [], [], [], [], [], [], [], None, None]
    unchanged = [gr.update()] * len(cleared)
    if not audio_file:
        yield [gr.update(interactive=True), [[STAGES[0], "failed", 0.0, "Choose an audio file first."]]] + cleared
        return
    audio_path = Path(audio_file)
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        yield [gr.update(interactive=True),
               [[STAGES[0], "failed", 0.0, f"Unsupported format {audio_path.suffix}; use one of {allowed}."]]] + cleared
        return

    updates = queue.Queue()
    outcome = {}

    def work():
        try:
            outcome["result"] = run(audio_path, models.get(size), glossary_text or "", packs or [],
                                    model_size=size, on_stage=lambda r: updates.put(status_rows(r)))
        except Exception as e:  # unexpected: show one line, keep the traceback in the terminal
            log.exception("Pipeline crashed")
            outcome["error"] = f"Unexpected error: {e} (full details in the terminal)"
        finally:
            updates.put(None)

    loading = [[name, "waiting", 0.0, ""] for name in STAGES]
    if size != models.size:
        loading[0] = [STAGES[0], "running", 0.0, f"loading Whisper {size} (first use downloads it)"]
    yield [gr.update(interactive=False), loading] + cleared

    threading.Thread(target=work, daemon=True).start()
    while (rows := updates.get()) is not None:
        yield [gr.update(interactive=False), rows] + unchanged

    if "error" in outcome:
        yield [gr.update(interactive=True), [[STAGES[0], "failed", 0.0, outcome["error"]]]] + cleared
        return
    result = outcome["result"]
    saved = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8")) if result.run_dir else {}
    files = [str(result.run_dir / n) for n in SAVED_FILES if result.run_dir and (result.run_dir / n).exists()]
    header = run_header(result, audio_path.name, glossary_text or "", packs or [], size)
    if not result.transcript:
        yield [gr.update(interactive=True), status_rows(result), header, "", "", [], [], []] + \
              [[]] * 6 + [files, saved]
        return
    record, dropped = minutes_view(result)
    yield [gr.update(interactive=True), status_rows(result), header, record, dropped, faithfulness_rows(saved),
           raw_highlights(result), refined_highlights(result), *details(result, saved), files, saved]


# ---------- Layout ----------

def build_ui(default_size: str) -> gr.Blocks:
    global OUTPUTS
    with gr.Blocks(title="Meeting Assistant") as demo:
        gr.HTML(HEADER_HTML)
        with gr.Row():
            audio = gr.File(label="Meeting recording",
                            file_types=sorted(SUPPORTED_EXTENSIONS), type="filepath")
            with gr.Column():
                glossary = gr.Textbox(label="Glossary terms (optional)", lines=3,
                                      placeholder="Kubeflow, ONNX, Priya ...",
                                      info="Names, products, jargon. Separate with commas, semicolons or "
                                           "new lines. Sent to Whisper and used to correct the transcript.")
                packs = gr.CheckboxGroup(list_packs(), label="Preset word packs (optional)",
                                         info="Only useful when the pack matches the meeting's topic.")
            size = gr.Dropdown(MODEL_CHOICES, value=default_size, label="Whisper model size",
                               info="Changing it reloads the model on the next Run.")
        run_btn = gr.Button("Run", variant="primary")

        status = gr.Dataframe(headers=STATUS_HEADERS, value=[[n, "waiting", 0.0, ""] for n in STAGES],
                              label="Status per stage", interactive=False, wrap=True)
        header = gr.Markdown()

        with gr.Tab("Meeting record"):
            record = gr.Markdown()
            gr.Markdown("**Removed or downgraded by the quote checks**")
            dropped = gr.Markdown()
            faith = gr.Dataframe(headers=["Faithfulness check", "Value"], interactive=False, wrap=True,
                                 label="Faithfulness report (counts from this run's verification)")
        with gr.Tab("Transcripts"):
            with gr.Row():
                raw = gr.HighlightedText(label="Raw (Whisper) — low-confidence words and possible "
                                               "hallucinations marked", combine_adjacent=True)
                refined = gr.HighlightedText(label="Refined — applied edits marked", combine_adjacent=True)
        with gr.Tab("Details"):
            segments = gr.Dataframe(headers=["#", "Start", "End", "Text", "no_speech_prob", "avg_logprob",
                                             "compression_ratio", "Halluc. score", "Flagged", "Reasons"],
                                    label="Segments (times mm:ss.ss; Whisper's confidence values)",
                                    interactive=False, wrap=True, max_height=400)
            edits = gr.Dataframe(headers=["Line", "Before", "After", "Status", "Reason"],
                                 label="Edit log (guard decisions)", interactive=False, wrap=True)
            hints = gr.Dataframe(headers=["Line", "Heard", "Term", "Score", "Sound", "Spelling",
                                          "Whisper unsure"],
                                 label="Hints sent to the LLM (similarity 0-100; hint threshold 72, "
                                       "65 when Whisper was unsure)", interactive=False, wrap=True)
            possible = gr.Dataframe(headers=["Line", "Text", "Reason"],
                                    label="Possible errors (flagged by the LLM, never changed)",
                                    interactive=False, wrap=True)
            glossary_table = gr.Dataframe(headers=["Term", "Source", "Heard as"],
                                          label="Final glossary (user / inferred / pack)", interactive=False)
            usage = gr.Dataframe(headers=["Provider", "Model", "Input tokens", "Output tokens", "Seconds"],
                                 label="LLM calls", interactive=False)
        with gr.Tab("Files"):
            files = gr.File(label="Saved files for this run", file_count="multiple", interactive=False)
            run_json = gr.JSON(label="run.json")

        OUTPUTS = [run_btn, status, header, record, dropped, faith, raw, refined,
                   segments, edits, hints, possible, glossary_table, usage, files, run_json]
        run_btn.click(run_meeting, [audio, glossary, packs, size], OUTPUTS)
    return demo


OUTPUTS: list = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meeting assistant web UI (localhost).")
    parser.add_argument("--model-size", default="medium", choices=[v for _, v in MODEL_CHOICES])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info("Loading Whisper %s ...", args.model_size)
    models.get(args.model_size)
    log.info("Whisper %s ready on %s", args.model_size, models.device())

    demo = build_ui(args.model_size)
    demo.queue(default_concurrency_limit=1)  # one GPU, one model: one run at a time
    demo.launch(server_name="127.0.0.1", server_port=args.port, theme=THEME, css=CSS,
                footer_links=[])
