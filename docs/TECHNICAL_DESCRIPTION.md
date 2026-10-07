# Meeting Assistant: technical description

The application turns a meeting recording into a speaker-labelled transcript, a
refined transcript and a meeting record (minutes, key decisions, action items).
Three models run in a fixed order; plain Python checks every model output before
the next stage uses it.

## Models and their roles

| Stage | Model | Role |
|---|---|---|
| 1. Speech to text | **Whisper medium** (OpenAI weights) run with **faster-whisper**, Silero voice-activity detection | Transcribes the recording into timed lines, with a time and a confidence for every word |
| 1a. Who spoke when | **pyannote `speaker-diarization-community-1`** | Separates the voices into speaker turns |
| 2. Transcript refinement (and speaker names) | **Language model #1**: `openai/gpt-oss-120b` on Groq | Infers the meeting's domain terms, decides which suspected mishearings to correct ("cube flow" → Kubeflow), and points at introductions and people being addressed so speakers can be named |
| 3. Meeting record | **Language model #2**: a separate `openai/gpt-oss-120b` call with its own prompt and a strict JSON schema | Writes the summary and minutes and labels decisions, action items, given facts, assessments and open questions, each with exact quotes |

The two language-model stages are separate calls with separate prompts
(`prompts/`); `gemini-2.5-flash` takes over either one when Groq is rate limited
or unavailable.

## How outputs move between stages

```
recording ─► ffprobe checks ─► Whisper ─► timed words ─┐
recording ─► pyannote ─► speaker turns ────────────────┴─► Python: speaker per word,
                                                           lines split at speaker changes
                                                           = RAW TRANSCRIPT
raw transcript ─► Python: hallucination flags (doubtful lines are never evidence)
raw transcript ─► LM #1: glossary terms ─► Python: keep terms actually heard
             ─► Python: sound-alike hints ─► LM #1: apply hints that fit
             ─► Python guard: revert edits to numbers, negations, unhinted words
             = REFINED TRANSCRIPT
refined transcript ─► LM #1: where names and roles are revealed
             ─► Python: right speaker? quote real? did the addressed person answer?
             = NAMED SPEAKERS (editable in the app)
refined transcript + speakers ─► LM #2: record with quotes
             ─► Python: every quote must be in the transcript; status set by rules
             = MEETING MINUTES, KEY DECISIONS, ACTION ITEMS
```

Python, not the language models, decides every status:

- a decision is **agreed** only with a verified agreement quote from someone other
  than the proposer; a point stated as settled with no objection is
  **decided, no objection**; anything else stays an open proposal
- an owner or deadline is kept only if a verified quote supports it ("I'll send
  it" makes its speaker the owner; a task given to a role goes to the speaker with
  that role); otherwise it is "unspecified"
- items whose quotes are not in the transcript are dropped, and every removal is
  listed with its reason

## Outputs

Each run writes `runs/<date-time>/deliverables/` with the raw transcript, refined
transcript, meeting minutes, key decisions and action items, each as a readable
file (`.txt` / `.md`) and a structured `.json` rendered from the same objects, plus
`deliverables.zip`. The app shows them in tabs and offers them for download;
examples are in `example/IS1004a/` (an AMI meeting) and `example/PL1001a/`
(a short product launch meeting).

## Measured accuracy

On six AMI meetings (Whisper medium, four speakers), against AMI's manual
annotations: 92.8% of words with the right speaker, 10.9% diarization error rate,
26.6% word error rate, and 13 of 13 names given to the right speaker. Details and
the command to reproduce them are in the README, section 16.1.
