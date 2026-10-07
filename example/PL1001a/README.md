# Example: meeting PL1001a

A 2-minute meeting of four people finalising the launch plan for a smart
thermostat, and the outputs the application produced for it.

| File | Contents |
|---|---|
| `PL1001a.mp3` | the meeting (mono, 16 kHz) |
| `outputs/raw_transcript.txt` / `.json` | speech-to-text before refinement, one `[mm:ss.ss] Speaker: text` line per segment |
| `outputs/refined_transcript.txt` / `.json` | after domain-term refinement |
| `outputs/meeting_minutes.md` / `.json` | participants and roles, summary, discussion points, given facts, assessments, proposals not decided, open questions |
| `outputs/key_decisions.md` / `.json` | decisions reached |
| `outputs/action_items.md` / `.json` | tasks with owner and deadline as stated |
| `deliverables.zip` | all of the above in one file |

The outputs are copied unchanged from the application's run folder (Whisper
medium, Number of speakers = 4, no typed glossary, `openai/gpt-oss-120b` on Groq
for both language-model stages).

## Speakers

| Speaker | How the application knows |
|---|---|
| Daniel, project manager | his own introduction, name and role |
| Emma, product designer | her own introduction; owner of a task she accepts |
| Raj, machine learning engineer | asked by name to introduce himself, then introduces himself |
| Chloe, marketing lead | her own introduction; owner of a task given to her role |

## What the application found

| In the meeting | In the outputs |
|---|---|
| Emma proposes matte white, Raj accepts | key decision, **agreed** |
| "No, a gold edition is too expensive … let's drop that" | **rejected** proposal |
| Launch date 1 June, budget 80,000 euro | **given** facts (the brief), not decisions |
| "Nine out of ten users liked it" | an **assessment**, not a decision |
| Emma: "I'll send the colour samples by Friday" | action item, owner Emma, by Friday |
| Raj: "I'll finish the model evaluation report by next Wednesday" | action item, owner Raj, by next Wednesday |
| "The marketing lead will prepare the launch video by the end of May" | action item, owner **Chloe** (from her role), by the end of May |
| "We still need to decide the price for Europe" | open question |

To reproduce: upload `PL1001a.mp3` in the app with Number of speakers
set to 4. The language model's answer can vary slightly between runs.
