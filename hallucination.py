"""Stage 1b: flag possible Whisper hallucinations.

A hallucination is text nobody said, usually produced over silence or noise
("Thanks for watching", "Subtitles by Amara.org"). Each segment gets a
suspicion score; segments scoring FLAG_SCORE or more are flagged with reasons.
Nothing is deleted: people really do say "thank you" and "bye" in meetings.

Phrase lists:
- data/hallucination/BoH.csv: the "Bag of Hallucinations" from Barański et al.,
  ICASSP 2025 (https://github.com/DSP-AGH/ICASSP2025_Whisper_Hallucination,
  MIT licence). They fed Whisper non-speech audio and kept the 294 phrases it
  invented most often.
- The condensed list in prompt 02 of
  https://github.com/DSP-AGH/asr_hallucination_detection_prompts

Try it:  python hallucination.py data/audio/ES2004a.wav
"""

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

BOH_PATH = Path(__file__).parent / "data" / "hallucination" / "BoH.csv"

# ---------- Thresholds (starting guesses; tune on test clips) ----------

FLAG_SCORE = 2              # score at or above this = "possible hallucination"
NO_SPEECH_PROB_MAX = 0.6    # above this, Whisper thinks there was no speech
AVG_LOGPROB_MIN = -1.0      # below this, Whisper was unsure of its words
COMPRESSION_RATIO_MAX = 2.4 # above this, the text repeats itself (looping)
END_FRACTION = 0.10         # the last 10% of the audio counts as "the end"
MIN_PLAIN_REPEAT = 3        # an ordinary line must repeat this often to get the mid-meeting point


def _norm(text: str) -> str:
    """Lowercase and turn everything except letters and digits into single
    spaces: "I'm sorry." -> "i m sorry", "Amara.org" -> "amara org".
    This is the same format BoH.csv uses, so the two can be compared."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


# ---------- Phrase lists ----------

# +2 if a segment CONTAINS one of these. They come from YouTube videos and
# subtitle files in Whisper's training data, never from a real meeting.
# Left out on purpose: bare "provided by" and "translated by", which people
# do say in meetings ("figures provided by finance").
SUBTITLE_PHRASES = [_norm(p) for p in [
    "thanks for watching", "thank you for watching", "welcome to my channel",
    "welcome back to my channel", "don't forget to subscribe", "subscribe to my channel",
    "like and subscribe", "hit the bell icon", "see you in the next video",
    "thefndc.com", "bf watch tv", "subtitles by", "amara.org", "emily beynon",
    "transcribed by eso", "closed captioning provided by", "captions by",
]]

# Phrases from the prompt-02 list that are not in BoH.csv (BoH leaves out the
# very common ones like "thank you", which also occur in real speech).
EXTRA_PHANTOM_PHRASES = [
    "thank you", "thank you very much", "thanks", "bye", "bye bye", "goodbye",
    "i'm sorry", "i love you", "oh my god", "we'll be right back",
    "i'll see you next time", "see you next time", "you", "bang", "beep", "boom",
    "meow", "cough", "sigh", "ding",
]

# BoH phrases that are normal in meetings, removed so that a real "Mm-hmm."
# "Mm-hmm." or "Hello everyone" is not flagged. Common backchannels
# ("yeah", "okay", "so", "no", "oh") are not in BoH and are not added.
MEETING_PHRASES = {_norm(p) for p in [
    "mm hmm", "hmm mm", "uh huh", "i know", "oh i see", "there we go", "let's go",
    "so let's go", "hello everyone", "hello friends", "hi guys", "hey guys",
    "good morning", "good night", "good job", "excuse me", "hold on", "come here",
    "watch out", "oops", "uh oh", "oh hey", "oh man", "oh god", "yay", "meh",
]}


def load_boh(path: Path = BOH_PATH) -> set[str]:
    """Read the phrase column of BoH.csv (columns: prediction, number of
    occurrences in noise)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {_norm(row["prediction"]) for row in csv.DictReader(f)}


# +1 if a segment is ONLY one of these (the whole line, after _norm).
PHANTOM_PHRASES = (load_boh() | {_norm(p) for p in EXTRA_PHANTOM_PHRASES}) - MEETING_PHRASES


# ---------- Result ----------

@dataclass
class SegmentFlag:
    index: int                  # position of the segment in the transcript
    score: int
    flagged: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return ", ".join(self.reasons)


def _contains_phrase(text_norm: str, phrases: list[str]) -> str | None:
    """Return the first phrase found as whole words in the text, or None.
    Spaces around both sides make "bye" not match inside "goodbye"."""
    for phrase in phrases:
        if f" {phrase} " in f" {text_norm} ":
            return phrase
    return None


def _repeat_runs(texts: list[str]) -> list[int]:
    """For each segment, how many consecutive segments (including itself) have
    the same text. 1 = not repeated."""
    run_len = [1] * len(texts)
    i = 0
    while i < len(texts):
        j = i
        while j + 1 < len(texts) and texts[j + 1] and texts[j + 1] == texts[i]:
            j += 1
        for k in range(i, j + 1):
            run_len[k] = j - i + 1
        i = j + 1
    return run_len


def score_segments(segments, duration_s: float) -> list[SegmentFlag]:
    """Score every segment. `segments` are stt.Segment objects (anything with
    start, text, no_speech_prob, avg_logprob, compression_ratio works)."""
    texts = [_norm(s.text) for s in segments]
    runs = _repeat_runs(texts)
    end_start = duration_s * (1 - END_FRACTION)

    results = []
    for i, seg in enumerate(segments):
        score, reasons = 0, []

        phrase = _contains_phrase(texts[i], SUBTITLE_PHRASES)
        if phrase:
            score += 2
            reasons.append(f'subtitle/YouTube phrase "{phrase}"')

        is_phantom = texts[i] in PHANTOM_PHRASES
        if is_phantom:
            score += 1
            reasons.append("only a known Whisper phantom phrase")

        if runs[i] >= 2:
            score += 1
            # The extra mid-meeting point only for suspicious repeats: a phantom
            # or subtitle phrase, or 3+ in a row. Two real lines in a row
            # ("Yeah." "Yeah.") stay at 1 point and are not flagged.
            suspicious_repeat = is_phantom or phrase or runs[i] >= MIN_PLAIN_REPEAT
            if seg.start < end_start and suspicious_repeat:
                score += 1
                reasons.append(f"repeated {runs[i]}x mid-meeting")
            else:
                reasons.append(f"repeated {runs[i]}x")

        if seg.no_speech_prob > NO_SPEECH_PROB_MAX or seg.avg_logprob < AVG_LOGPROB_MIN:
            score += 1
            reasons.append("low confidence")

        if seg.compression_ratio > COMPRESSION_RATIO_MAX:
            score += 1
            reasons.append("looping text")

        results.append(SegmentFlag(i, score, score >= FLAG_SCORE, reasons))
    return results


def doubtful_indices(flags: list[SegmentFlag]) -> set[int]:
    """Indices of flagged segments, for glossary.build_glossary(doubtful=...)
    and for marking lines as doubtful in Stage 3."""
    return {f.index for f in flags if f.flagged}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hallucination.py <audio file> [model size]")
        sys.exit(1)
    from stt import AudioInputError, load_model, transcribe, _fmt

    audio, size = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "medium")
    try:
        transcript = transcribe(audio, load_model(size), model_size=size)
    except AudioInputError as e:
        print(f"Stage 1 failed: {e}")
        sys.exit(1)

    flags = score_segments(transcript.segments, transcript.duration_s)
    flagged = [f for f in flags if f.flagged]
    for f in flagged:
        s = transcript.segments[f.index]
        print(f"[{_fmt(s.start)}] score {f.score}: {s.text!r}  <- {f.reason_text}")
    print(f"\n{len(flagged)} of {len(flags)} segments flagged as possible hallucinations")
