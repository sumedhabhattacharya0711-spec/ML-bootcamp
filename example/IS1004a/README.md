# Example: AMI meeting IS1004a

A 13-minute kickoff meeting of four people (project manager, user interface
designer, industrial designer, marketing expert) planning a new remote control,
and the outputs the application produced for it.

| File | Contents |
|---|---|
| `IS1004a.Mix-Headset.mp3` | the recording (mono, 16 kHz, compressed from the corpus' WAV) |
| `outputs/raw_transcript.txt` / `.json` | speech-to-text before refinement, one `[mm:ss.ss] Speaker: text` line per segment |
| `outputs/refined_transcript.txt` / `.json` | after domain-term refinement |
| `outputs/meeting_minutes.md` / `.json` | participants and roles, summary, discussion points, given facts, open questions |
| `outputs/key_decisions.md` / `.json` | decisions reached (none in this kickoff: an empty list) |
| `outputs/action_items.md` / `.json` | tasks with owner and deadline as stated |
| `deliverables.zip` | all of the above in one file |

The outputs are copied unchanged from the application's run folder. They were
produced from the original WAV with Whisper medium and no typed glossary;
diarization found the four speakers. The meeting record (language model #2) came from
`openai/gpt-oss-120b` on Groq; language model #1's steps (glossary, refinement,
speaker names) were answered by the backup, `gemini-2.5-flash`, because Groq's
free daily quota was used up at the time.

What the run shows:

- four voices separated (95.9% of words with the right speaker against AMI's
  manual annotations), and two speakers named from their own introductions
  with their roles: Sebastian (project manager) and Michael (user interface
  designer); the other two never said their names and stay "Speaker 3" and
  "Speaker 4"
- the brief handed to the meeting (25 euro selling price, 50 million euro
  profit target, international market) listed as given facts, not as
  decisions
- every task the project manager handed out, each with its timestamped quote;
  owners stay "unspecified" because the tasks were given to "you", not by name
- an empty key-decisions list, because no decision was reached in this meeting

To reproduce: upload `IS1004a.Mix-Headset.mp3` in the app with Number of
speakers set to 4 (the speech-to-text and diarization results can differ
slightly from the WAV, and the language-model answers vary between runs).

## Source and licence

The recording is from the AMI Meeting Corpus, meeting IS1004a, headset mix,
<https://groups.inf.ed.ac.uk/ami/corpus/>, licensed under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0). It was converted to MP3;
nothing else was changed.

J. Carletta et al., "The AMI Meeting Corpus: A Pre-Announcement", Machine
Learning for Multimodal Interaction (MLMI), 2005.
