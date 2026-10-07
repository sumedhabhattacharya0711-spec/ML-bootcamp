# Project notes

## Refactoring log

Readability and consistency pass over the existing code, done in small
batches. Each batch keeps behaviour the same unless it says "bug fix", and
was checked with the full test suite plus real runs on the AMI ES2004a clip.

| Batch | Commit | Files | What changed |
| --- | --- | --- | --- |
| 1 | 67e8044 | llm.py, glossary.py, refine.py, stt.py, hallucination.py | Module boundaries |
| 2 | 60db549 | llm.py, tests/test_llm.py | **Bug fix**: bad structured replies crashed |
| 3 | 0d89810 | hallucination.py | Repeat counting simplified, clearer name |
| 4 | 1c230b1 | refine.py | Hint search cleans each word and term once |
| 5 | fd91469 | minutes.py | Decision rule and owner/deadline checks simplified |
| 6 | 301169b | tests/test_refine.py, glossary.py | **Test fix**: a test that checked nothing |

### Batch 1: module boundaries

- **Problem:** `refine.py` imported glossary's private `_parse_json_reply` and
  caught `GlossaryInferError` for a failure that has nothing to do with the
  glossary. `hallucination.py` imported stt's private `_fmt`.
- **Change:** the JSON-reply parser now lives in `llm.py` as
  `parse_json_reply` and raises `LLMError`; `GlossaryInferError` is gone.
  `stt._fmt` became the public `stt.format_timestamp`. Added the missing blank
  line between glossary.py's imports and constants.
- **Trade-off:** importing glossary (and so stt) now also imports llm.py,
  which loads the openai library and reads `.env`. Nothing is sent until a call
  is made.

### Batch 2: llm.py (bug fix)

- **Problem:** `call_llm_structured` was meant to retry once when the LLM's
  JSON did not match the pydantic schema, then raise
  `LLMError("LLM answer failed validation: <field>")`. But `parse()` raises
  pydantic's `ValidationError`, which is not an openai error, so `_send` never
  wrapped it and the retry branch (which waited for an `LLMError`) could not
  run. A malformed reply crashed with a raw pydantic traceback.
- **Change:** catch `ValidationError` directly around the `parse()` call.
  Also simplified the token-usage lines (`getattr(...)` was redundant).
- **Tests:** two new tests (retry succeeds; two bad replies give the friendly
  error). Both fail on the old code with the raw traceback and pass now.

### Batch 3: hallucination.py

- `_repeat_runs` (how many lines in a row share the same text) replaced a
  nested `while` loop with `itertools.groupby`. Old and new compared on 20,000
  random lists: identical.
- `end_start` renamed `end_zone_start`: the time where the last 10% of the
  recording (the "end zone", where real goodbyes repeat) begins.
- ES2004a still flags exactly 1 of 178 lines ("Thanks for watching!", score 4).

### Batch 4: refine.py

- `find_hints` recomputed `_clean(term)` and `_clean(word)` for every
  span x term. Now each term and each line's words are cleaned once. Old and
  new compared on 400 random lines: identical hints; about 2.4x faster.
- Best-first sort uses `reverse=True` instead of a negated key.
- The possible-errors loop skips out-of-range line numbers up front, like
  the loop above it.
- ES2004a still gives the same 3 hints.

### Batch 5: minutes.py

- Decision rule: four `if/elif` branches became "open if there is no
  evidence, otherwise whichever (agreement or rejection) was said last".
- The owner and deadline checks were copy-pasted twins; now one loop over the
  two fields.
- `to_json` uses `m.model_dump_json(indent=2)` (byte-identical output) and the
  `json` import went away.
- Old and new `verify()` compared on 102 edge cases (missing, invented,
  "Unspecified", backchannel-only agreements, agreement vs rejection order):
  identical.

### Batch 6: tests and comments (test fix)

- **Problem:** `test_unsure_word_uses_lower_threshold` used "graffiti" vs
  Grafana, which scores 64.2, below both thresholds (65 and 72), so its real
  assertion never ran and the test passed without checking anything.
- **Change:** it now uses "grabant" (scores 69, between the thresholds) and
  asserts the score range directly. Checked by breaking the threshold on
  purpose: the test fails, and passes again once restored.
- Fixed one stale comment in glossary.py ("Provided by llm.py later": llm.py
  exists now). Other comments were reviewed and kept: they explain why
  (threshold choices, excluded phrases, edge cases), not what.

### Not changed on purpose

- The text-normalising helpers in hallucination, minutes, refine and glossary
  look alike but differ on purpose (BoH format, quote matching, token
  comparison). A shared utils module would add a generic layer for no gain.
- Thresholds, phrase lists and prompts: behaviour, not readability.

### Known issue (behaviour, not part of the refactor)

- `stt.py` passes the glossary as Whisper's `initial_prompt`. With
  `condition_on_previous_text=False`, faster-whisper only uses that for the
  first 30-second window. faster-whisper's `hotwords` setting applies the
  hint to every window; switching to it is a pending fix.

## Build log

- **pipeline.py** (new): runs Stage 1 -> 1b -> glossary -> Stage 2 -> Stage 3
  with a status per stage (done / failed / skipped + message + seconds).
  A bad file fails Stage 1 and skips the rest; a Stage 2 failure still lets
  Stage 3 run on the raw transcript; a Stage 3 failure keeps both transcripts.
  Every run is saved to `runs/<timestamp>/`: `transcript_raw.txt`,
  `transcript_refined.txt`, `edit_log.json`, `minutes.md`, `minutes.json`,
  `run.json` (statuses, segments, flags, glossary, LLM token usage).
  Tests: `tests/test_pipeline.py` (fake Whisper model + fake LLM, 4 tests).

### Batch P: pipeline.py

- **Problem:** an unknown glossary pack name crashed the whole run with a raw
  `FileNotFoundError`, and the glossary build sat outside Stage 2's error
  handling. Failed runs were not saved although the docstring said every run
  is. Stage 1's time stayed 0.0 s when it failed.
- **Change:** loading typed + pack terms moved inside Stage 1's `try`; an
  unknown pack now gives "Stage 1 failed: Glossary pack not found: ..." and
  skips the rest. `build_glossary` moved inside Stage 2's `try`. Failed runs
  are saved too (`run.json` records why). Stage 1's time is set on failure.
  Docstring says the glossary build is timed with Stage 2.
- **Readability:** typed `flags: list[SegmentFlag]`, `glossary: list[Term]`;
  sorted the glossary import; `stage3_lines` renamed `minutes_input`.
- **Not changed:** packs are still read twice (for Whisper's hint and inside
  `build_glossary`); avoiding that needs a change to glossary.py's API, and
  reading a few short text files twice is harmless.
- **Tests:** 2 new (unknown pack gives a status; failed run is saved).

### Batch G: Groq main provider, Gemini backup (llm.py)

- **Before:** llm.py defaulted to OpenAI (`gpt-4.1`) with an optional
  `LLM_BASE_URL`; no backup provider.
- **Now:** Groq is the default main provider (`openai/gpt-oss-120b` at
  Groq's OpenAI-compatible URL; `LLM_BASE_URL` / `LLM_MODEL` still override).
  Gemini is the backup (`LLM_GEMINI_KEY`, model `gemini-3.8-flash` unless
  `LLM_GEMINI_MODEL` is set) through Gemini's OpenAI-compatible endpoint.
- **When the backup is used:** rate limit, quota exceeded, timeout, connection
  error or 5xx from Groq, and only if `LLM_GEMINI_KEY` is set. A bad key, a
  bad request or a missing model does not fall back, so setup errors stay
  visible. If both fail, one `LLMError` lists both reasons, e.g.
  "LLM call failed: groq: rate limited ... | gemini: error 503 ...".
- `usage_log` entries record which provider answered (`run.json` shows it).
- `.env.example` documents all five LLM settings.
- **Checks:** 3 new tests (rate-limited Groq falls back; bad Groq key does
  not; both failing gives one error). Live: with Groq made unreachable on
  purpose, the test call was answered by Gemini; normal runs use Groq.

### Fix: Stage 2 answer cut off ("LLM reply contained no JSON object")

- **Cause (reproduced):** Groq returned `finish_reason: "length"`: gpt-oss
  used most of the 3,072-token default output budget on hidden reasoning and
  the JSON was cut off mid-word. The answer was long because refine.py sent 15
  fake hints: everyday multi-word glossary terms ("project manager,",
  "selling price") that were already spelled right. The "already the term"
  check only worked for one-word terms.
- **refine.py:** a span is skipped when it already contains the term's words
  (ignoring capitals and punctuation). "G D P R" -> GDPR still gets a hint.
- **llm.py:** every call sends `max_tokens` (8,192) and `reasoning_effort:
  "low"` (dropped automatically for models that reject it, like temperature).
  A cut-off answer is retried once with double the limit; only then
  "the answer was cut off ... even after retrying". Truncated answers are never
  used, because a half answer would put half sentences into the transcript.
- **Result on ES2004a:** Stage 2 completes (refine call 1,226 output tokens,
  was 3,072 and cut off); Stage 3 13.5 s (was 52 s).
- **Tests:** 5 new (multi-word term already correct; cut-off retried for text
  and structured calls; cut off twice gives the error; rejected
  reasoning_effort dropped).

### Fix: wrong "I mean," -> "menu" edit; Whisper hint for the whole recording

- **Problem (seen on ES2004a):** the hint "mean" -> menu (sound-alike) was
  accepted by the LLM, which also swallowed the "I" before it; the guard let it
  through because "menu" is a glossary term. Separately, the glossary reached
  Whisper only for the first 30-second window (`initial_prompt` with
  `condition_on_previous_text=False`).
- **Guard (refine.py):** an edit may only change words inside one of that
  line's hints; anything else is blocked with "changes words that were not
  hinted". Number, negation and deletion checks come first, as before.
- **Hints (refine.py):** everyday verbs, nouns and fillers seen as false hints
  ("mean", "stuff", "work", "people", ...) added to `COMMON_WORDS`.
- **Whisper (stt.py):** the glossary is passed as `hotwords`, which faster-whisper
  adds to every window. `glossary.to_initial_prompt` renamed `to_hotwords`;
  `Transcript.initial_prompt` renamed `hotwords`.
- **Result on ES2004a:** 1 edit applied ("user friendly," -> "User-friendly,"),
  2 bad edits blocked (one by the new hinted-words rule); no "menu" edit; all
  4 stages done.
- **Tests:** 2 new (edit outside the hinted words is blocked; "mean" is not
  hinted as menu); guard tests now pass the hinted word ranges.

## app.py (Gradio UI)

- `python app.py` serves on http://127.0.0.1:7860 (localhost only);
  `--model-size small|medium|turbo|large-v3`, `--port`.
- Whisper loads once at startup; the size dropdown swaps the single loaded
  model on the next Run (old one freed first). One run at a time
  (`queue(default_concurrency_limit=1)`).
- Run is the only event. `pipeline.run` runs in a worker thread; its new
  optional `on_stage` callback (pipeline.py, the only backend change) streams
  the status table live. Old results are cleared when a run starts.
- Shows: status per stage, what the result was produced from (file, model,
  device, LLM, hotwords, glossary, packs, run folder), meeting record + items
  removed by the checks, raw transcript (low-confidence words p<0.5 and
  possible hallucinations marked) beside the refined one (applied edits
  marked), tables for segments / edit log / hints / possible errors / glossary
  / LLM calls, and the saved files + run.json.

## Research additions (from the team's Part 04/05/07 notes)

- **Timestamped, multi-span evidence (Part 04, MeetingQA):** every decision
  and action item carries all of its verified quotes (`evidence`: role,
  quote, line, start seconds): proposal + agreement/rejection for decisions;
  task, owner, deadline and acceptance quotes for action items. Shown in the
  Markdown as `[mm:ss.ss] role: "quote"` and saved in minutes.json.
- **Action-item agreement (Part 04, Purver et al.'s D/O/T/A):** the LLM also
  returns `agreement_quote`; Python sets `status`: `agreed` (named owner and a
  verified, non-backchannel acceptance quote), `proposed` (owner but no
  verified acceptance), `unassigned` (no supported owner).
- **Faithfulness report (Part 07's key metrics, computed per run without gold
  data):** decisions/action items proposed vs kept, agreed counts, owners and
  deadlines removed, ignored agreement quotes, evidence support rate, and the
  refine guard's applied/blocked edits by reason. In `run.json`
  ("faithfulness") and as a table in the UI.
- **Considered, not done:** Fast Conformer hybrid (needs training), AlignScore
  (1.4 GB model; exact quote matching is stricter for extractive evidence),
  SelfCheckGPT (5x LLM calls vs free-tier limits), Self-Refine / ReAct / DSPy /
  RL (extra LLM rounds or training), pyannote speaker labels (needs HF token;
  optional in the brief).

### Fix: weak hints and run-to-run determinism

- **Hint thresholds (refine.py):** 72 -> 82 and 65 -> 75 (Whisper unsure),
  tuned on ES2004a: real misheard terms scored 83+, everyday sound-alikes
  74-80 ("point" -> Pound, "makes it" -> Market, "functions" -> Functional
  design) and gpt-oss sometimes accepted them. Weak hints are no longer sent.
- **Whisper (stt.py):** `temperature=0.0` turns off faster-whisper's sampling
  fallback, so the same audio and hotwords give the same transcript.
- **LLM (llm.py):** `seed` added next to `temperature 0` (both best effort on
  Groq). Since Groq still varied, answers are cached in `.llm_cache/` keyed by
  the exact model, prompt and input: the same input always gives the same
  answer (`LLM_CACHE=0` turns it off; `run.json` shows cache hits as provider
  "cache"). A first run on a new recording is still the LLM's own answer.
- **Check:** two full ES2004a runs gave byte-identical raw/refined transcripts,
  edit log, minutes.json and minutes.md; the second run used the cache for all
  three LLM calls.
