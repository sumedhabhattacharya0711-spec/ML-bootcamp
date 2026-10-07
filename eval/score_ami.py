"""Score saved runs against the AMI Meeting Corpus' own annotations.

For each run of an AMI meeting (runs/<timestamp>/, made from <meeting>.Mix-Headset.wav):

  WER                 Whisper's words against AMI's manual transcript (lowercase, no punctuation)
  DER                 pyannote turns against AMI's manual speaker segments, scored two ways:
                      strict (no collar, overlapping speech counted) and standard
                      (0.25 s collar, overlapping speech skipped)
  right speaker       share of transcript words whose speaker matches AMI's, after the
                      best one-to-one mapping of our speakers to AMI's
  names               each named speaker: correct if AMI says the line the name came from
                      was said by the person that speaker maps to

Setup (once):
  curl -L -o data/ami_annotations.zip \\
    https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
  unzip -q data/ami_annotations.zip -d data/ami_annotations

Run (on runs made from AMI's <meeting>.Mix-Headset.wav files):
  python eval/score_ami.py runs/20261008-101500 runs/20261008-102300
  python eval/score_ami.py --record runs/20261008-101500     # also print the record next to AMI's summary
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.optimize import linear_sum_assignment

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROLE_NAMES = {"PM": "project manager", "ID": "industrial designer", "UI": "UI designer", "ME": "marketing expert"}


def norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


class Reference:
    """AMI's manual annotations for one meeting."""

    def __init__(self, ami: Path, meeting: str):
        self.ami, self.meeting = ami, meeting
        self.roles = {}
        for m in ET.parse(ami / "corpusResources/meetings.xml").getroot():
            if m.get("observation") == meeting:
                self.roles = {sp.get("nxt_agent"): ROLE_NAMES.get(sp.get("role"), sp.get("role")) for sp in m}
        if not self.roles:
            raise SystemExit(f"{meeting} is not in {ami}/corpusResources/meetings.xml")
        self.words = []  # (start, end, speaker, word)
        self.turns = Annotation()
        for spk in sorted(self.roles):
            for w in ET.parse(ami / f"words/{meeting}.{spk}.words.xml").getroot().iter("w"):
                if not w.get("punc") and w.text and w.get("starttime") is not None:
                    self.words.append((float(w.get("starttime")), float(w.get("endtime")), spk, w.text))
            for s in ET.parse(ami / f"segments/{meeting}.{spk}.segments.xml").getroot():
                a, b = float(s.get("transcriber_start")), float(s.get("transcriber_end"))
                if b > a:
                    self.turns[Segment(a, b)] = spk
        self.words.sort()
        self.starts = np.array([w[0] for w in self.words])

    @property
    def text(self) -> str:
        return " ".join(w for *_, w in self.words)

    def speaker_at(self, start: float, end: float) -> str | None:
        """Speaker of the reference word overlapping [start, end] most, or the nearest within 0.5 s."""
        i = int(np.searchsorted(self.starts, end))
        best, best_score = None, -0.5
        for j in range(max(0, i - 15), min(len(self.words), i + 2)):
            a, b, spk, _ = self.words[j]
            score = min(end, b) - max(start, a)
            if score > best_score:
                best, best_score = spk, score
        return best

    def summary(self) -> dict[str, list[str]]:
        path = self.ami / f"abstractive/{self.meeting}.abssumm.xml"
        if not path.exists():
            return {}
        root = ET.parse(path).getroot()
        return {part: [" ".join("".join(x.itertext()).split()) for el in root.iter(part) for x in el]
                for part in ("decisions", "actions", "problems")}


def score(run: Path, ref: Reference) -> dict:
    saved = json.loads((run / "run.json").read_text(encoding="utf-8"))
    segments = saved["transcript"]["segments"]
    out = {"run": run.name, "meeting": ref.meeting}

    wer = jiwer.process_words(norm(ref.text), norm(" ".join(s["text"] for s in segments)))
    out["wer"] = wer.wer

    speakers_file = run / "speakers.json"
    if not speakers_file.exists():
        return out  # run without diarization
    speakers = json.loads(speakers_file.read_text(encoding="utf-8"))
    hypothesis = Annotation()
    for t in speakers["turns"]:
        hypothesis[Segment(t["start"], t["end"])] = t["speaker"]
    out["der_strict"] = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref.turns, hypothesis)
    out["der"] = DiarizationErrorRate(collar=0.25, skip_overlap=True)(ref.turns, hypothesis)

    pairs = Counter()
    for s in segments:
        for w in s["words"]:
            r = ref.speaker_at(w["start"], w["end"])
            if r and s.get("speaker"):
                pairs[(s["speaker"], r)] += 1
    ours, theirs = sorted({h for h, _ in pairs}), sorted({r for _, r in pairs})
    matrix = np.array([[pairs[(h, r)] for r in theirs] for h in ours])
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {ours[i]: theirs[j] for i, j in zip(rows, cols)}
    out["right_speaker"] = sum(matrix[i, j] for i, j in zip(rows, cols)) / matrix.sum()
    out["speakers_found"], out["speakers_true"] = len(speakers["speakers"]), len(ref.roles)

    named, correct, details = 0, 0, []
    for sp in speakers["speakers"]:
        if not sp["name"]:
            continue
        named += 1
        source = next((e for e in speakers["name_evidence"]
                       if e["accepted"] and e["speaker"] == sp["id"] and e["kind"] == "self"
                       and norm(e["name"]) == norm(sp["name"])), None)
        if source is None:  # named by being addressed, or edited by hand: not checkable this way
            details.append(f"{sp['name']}: not from a self-introduction, not checked")
            continue
        line = segments[source["line"]]
        said_by = Counter(ref.speaker_at(w["start"], w["end"]) for w in line["words"]).most_common(1)[0][0]
        ok = said_by == mapping.get(sp["id"])
        correct += ok
        details.append(f"{sp['name']}: {'correct' if ok else 'WRONG'} ({ref.roles.get(said_by)} said "
                       f"\"{line['text']}\")")
    out["named"], out["named_correct"], out["name_details"] = named, correct, details
    return out


def print_record(run: Path, ref: Reference) -> None:
    minutes = run / "minutes.json"
    if not minutes.exists():
        print("  (no meeting record in this run)")
        return
    m = json.loads(minutes.read_text(encoding="utf-8"))
    for label, items in (("given", [g["fact"] for g in m.get("given", [])]),
                         ("decisions", [f"[{d['status']}] {d['text']}" for d in m["decisions"]]),
                         ("actions", [f"[{a['status']}] {a['task']} (owner: {a['owner']})" for a in m["action_items"]]),
                         ("open questions", [q["question"] for q in m["open_questions"]])):
        print(f"  ours, {label}:" + "".join(f"\n    - {x}" for x in items) if items else f"  ours, {label}: (none)")
    for part, items in ref.summary().items():
        print(f"  AMI, {part}:" + "".join(f"\n    - {x}" for x in items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score runs against AMI's manual annotations.")
    parser.add_argument("runs", nargs="+", help="RUN_DIR, or RUN_DIR:MEETING for runs saved before run.json "
                                                "recorded the audio file, e.g. runs/20261007-184822:ES2004a")
    parser.add_argument("--ami", type=Path, default=PROJECT_DIR / "data" / "ami_annotations",
                        help="unzipped ami_public_manual annotations (default: data/ami_annotations)")
    parser.add_argument("--record", action="store_true", help="also print the record next to AMI's summary")
    args = parser.parse_args()
    if not (args.ami / "corpusResources" / "meetings.xml").exists():
        sys.exit(f"AMI annotations not found in {args.ami}; see the setup lines at the top of this file.")

    results = []
    for item in args.runs:
        run, _, meeting = item.rpartition(":") if ":" in item else (item, "", "")
        if not meeting:  # ES2004a.Mix-Headset.wav -> ES2004a
            audio = json.loads((Path(run) / "run.json").read_text(encoding="utf-8")).get("audio", "")
            meeting = audio.split(".")[0]
            if not meeting:
                sys.exit(f"{run}: run.json does not say which recording it is; pass {run}:MEETING")
        ref = Reference(args.ami, meeting)
        r = score(Path(run), ref)
        results.append(r)
        line = f"{meeting:8} WER {r['wer']:6.1%}"
        if "der" in r:
            line += (f"  DER {r['der']:6.1%} (strict {r['der_strict']:5.1%})  right speaker {r['right_speaker']:6.1%}"
                     f"  speakers {r['speakers_found']}/{r['speakers_true']}  names {r['named_correct']}/{r['named']} correct")
        print(line)
        for d in r.get("name_details", []):
            print(f"  {d}")
        if args.record:
            print_record(Path(run), ref)

    scored = [r for r in results if "der" in r]
    if len(results) > 1:
        print(f"\nAverage over {len(results)} meetings: WER {np.mean([r['wer'] for r in results]):.1%}"
              + (f", DER {np.mean([r['der'] for r in scored]):.1%}, right speaker "
                 f"{np.mean([r['right_speaker'] for r in scored]):.1%}, names "
                 f"{sum(r['named_correct'] for r in scored)}/{sum(r['named'] for r in scored)} correct"
                 if scored else ""))


if __name__ == "__main__":
    main()
