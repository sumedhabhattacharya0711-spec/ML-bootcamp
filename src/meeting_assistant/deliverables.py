"""The five required outputs of a run, each in a human-readable and a
machine-readable file, written to runs/<timestamp>/deliverables/:

  raw_transcript.txt / .json        speech-to-text before refinement
  refined_transcript.txt / .json    after domain-term correction
  meeting_minutes.md / .json        summary and the main discussion points
  key_decisions.md / .json          decisions reached; [] if none were reached
  action_items.md / .json           tasks with owner and deadline as stated; [] if none

Both files of a pair are rendered from the same objects, so they carry the
same content. "Decisions reached" are the agreed ones and those stated as
settled with no objection; open and rejected proposals are listed in the
minutes instead. If a stage failed, its files are left out and STATUS.txt
says why: a missing record is not the same as an empty one.
"""

import json
import zipfile
from pathlib import Path

from meeting_assistant.minutes import Evidence, Minutes
from meeting_assistant.stt import format_timestamp

FOLDER = "deliverables"
REACHED = {"agreed": "agreed", "uncontested": "decided, no objection"}


def _evidence(evidence: list[Evidence]) -> list[dict]:
    return [{"role": e.role, "quote": e.quote, "speaker": e.speaker,
             "time": format_timestamp(e.start) if e.start is not None else None, "line": e.line}
            for e in evidence]


def _evidence_md(evidence: list[Evidence]) -> list[str]:
    return [f'  - [{e["time"] or "line " + str(e["line"])}] {e["speaker"] + ", " if e["speaker"] else ""}'
            f'{e["role"]}: "{e["quote"]}"' for e in _evidence(evidence)]


def _write(folder: Path, name: str, text: str, data) -> list[Path]:
    human = folder / name[0]
    machine = folder / name[1]
    human.write_text(text.rstrip() + "\n", encoding="utf-8")
    machine.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [human, machine]


# ---------- Transcripts ----------

def transcript(segments: list, lines: list[str], labels: list[str | None]) -> tuple[str, list[dict]]:
    """One "[mm:ss.ss] Speaker: text" line per segment (no speaker part when speakers are unknown)."""
    rows, text = [], []
    for seg, line, label in zip(segments, lines, labels):
        rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "speaker": label, "text": line})
        text.append(f"[{format_timestamp(seg.start)}] {label + ': ' if label else ''}{line}")
    return "\n".join(text), rows


# ---------- Meeting record ----------

def minutes_data(m: Minutes) -> dict:
    return {
        "participants": [{"name": p, "role": m.roles.get(p) or None} for p in m.participants],
        "summary": m.summary,
        "discussion_points": m.minutes,
        "given": [{"fact": g.fact, "evidence": _evidence(g.evidence)} for g in m.given],
        "assessments": [{"topic": a.topic, "verdict": a.verdict, "evidence": _evidence(a.evidence)}
                        for a in m.assessments],
        "proposals_not_decided": [{"proposal": d.text, "status": d.status, "evidence": _evidence(d.evidence)}
                                  for d in m.decisions if d.status not in REACHED],
        "open_questions": [{"question": q.question, "evidence": _evidence(q.evidence)} for q in m.open_questions],
    }


def minutes_md(m: Minutes) -> str:
    data = minutes_data(m)
    out = ["# Meeting minutes", ""]
    if data["participants"]:
        out += ["## Participants", ""] + [f"- {p['name']}" + (f" ({p['role']})" if p["role"] else "")
                                          for p in data["participants"]] + [""]
    out += ["## Summary", "", m.summary, "", "## Main discussion points", ""]
    out += [f"- {x}" for x in m.minutes] or ["- (none)"]
    out += ["", "## Given (brief, budget, targets: not decided in this meeting)", ""]
    for g in m.given:
        out += [f"- {g.fact}"] + _evidence_md(g.evidence)
    out += [] if m.given else ["- (none)"]
    out += ["", "## Assessments (ratings and evaluations: not decisions)", ""]
    for a in m.assessments:
        out += [f"- {a.topic}: {a.verdict}"] + _evidence_md(a.evidence)
    out += [] if m.assessments else ["- (none)"]
    out += ["", "## Proposals not decided", ""]
    undecided = [d for d in m.decisions if d.status not in REACHED]
    for d in undecided:
        out += [f"- **{d.status}**: {d.text}"] + _evidence_md(d.evidence)
    out += [] if undecided else ["- (none)"]
    out += ["", "## Open questions", ""]
    for q in m.open_questions:
        out += [f"- {q.question}"] + _evidence_md(q.evidence)
    out += [] if m.open_questions else ["- (none)"]
    return "\n".join(out)


def decisions_data(m: Minutes) -> list[dict]:
    return [{"decision": d.text, "status": REACHED[d.status], "evidence": _evidence(d.evidence)}
            for d in m.decisions if d.status in REACHED]


def decisions_md(m: Minutes) -> str:
    out = ["# Key decisions", ""]
    reached = [d for d in m.decisions if d.status in REACHED]
    for d in reached:
        out += [f"- **{REACHED[d.status]}**: {d.text}"] + _evidence_md(d.evidence)
    return "\n".join(out + ([] if reached else ["No decisions were reached in this meeting."]))


def actions_data(m: Minutes) -> list[dict]:
    return [{"task": a.task, "owner": a.owner, "deadline": a.deadline, "status": a.status,
             "evidence": _evidence(a.evidence)} for a in m.action_items]


def actions_md(m: Minutes) -> str:
    out = ["# Action items", ""]
    for a in m.action_items:
        out += [f"- {a.task}", f"  - owner: {a.owner}", f"  - deadline: {a.deadline}", f"  - status: {a.status}"]
        out += _evidence_md(a.evidence)
    return "\n".join(out + ([] if m.action_items else ["No action items were assigned in this meeting."]))


# ---------- All of them ----------

def write(run_dir: Path, segments: list, raw: list[str], refined: list[str], labels: list[str | None],
          minutes: Minutes | None, stages: list) -> Path:
    """Write the deliverables folder and run_dir/deliverables.zip. Returns the folder."""
    folder = run_dir / FOLDER
    folder.mkdir(exist_ok=True)
    for old in folder.iterdir():  # a rename rewrites everything; nothing stale may stay
        old.unlink()
    written, missing = [], []
    if segments:
        written += _write(folder, ("raw_transcript.txt", "raw_transcript.json"), *transcript(segments, raw, labels))
        written += _write(folder, ("refined_transcript.txt", "refined_transcript.json"),
                          *transcript(segments, refined, labels))
    else:
        missing.append("raw_transcript, refined_transcript (no transcript)")
    if minutes is not None:
        written += _write(folder, ("meeting_minutes.md", "meeting_minutes.json"), minutes_md(minutes),
                          minutes_data(minutes))
        written += _write(folder, ("key_decisions.md", "key_decisions.json"), decisions_md(minutes),
                          decisions_data(minutes))
        written += _write(folder, ("action_items.md", "action_items.json"), actions_md(minutes),
                          actions_data(minutes))
    else:
        missing.append("meeting_minutes, key_decisions, action_items (no meeting record)")
    if missing:
        lines = ["Some deliverables could not be produced for this recording:", ""]
        lines += [f"- {m}" for m in missing] + ["", "Stage status:", ""]
        lines += [f"- {s.name}: {s.status}{' (' + s.message + ')' if s.message else ''}" for s in stages]
        (folder / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(folder / "STATUS.txt")
    with zipfile.ZipFile(run_dir / "deliverables.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in written:
            z.write(path, f"{FOLDER}/{path.name}")
    return folder
