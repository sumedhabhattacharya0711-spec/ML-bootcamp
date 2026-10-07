from meeting_assistant.speakers import (
    NameEvidence, Speaker, Turn, assign_names, assign_speakers, check_claim, line_labels, name_speakers,
    relabel, rename_speakers, replace_labels, smooth_labels, summarize,
)
from meeting_assistant.stt import Segment, Word


def words(text, start, end):
    """Evenly timed Whisper words for a line."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return [Word(start + i * step, start + (i + 1) * step, t, 0.9) for i, t in enumerate(tokens)]


def seg(text, start, end, speaker=None):
    return Segment(start, end, text, 0.05, -0.3, 1.4, words(text, start, end), speaker)


# ---------- Stage 1a: turns -> one speaker per line ----------

def test_relabel_numbers_speakers_by_first_appearance():
    turns = relabel([(5.0, 8.0, "SPEAKER_00"), (0.0, 5.0, "SPEAKER_03"), (8.0, 9.0, "SPEAKER_03")])
    assert [(t.start, t.speaker) for t in turns] == [(0.0, "S1"), (5.0, "S2"), (8.0, "S1")]


def test_line_with_one_speaker_keeps_its_text():
    s = seg("We moved the training jobs to cube flow.", 0, 4)
    out = assign_speakers([s], [Turn(0, 4.2, "S1")])
    assert len(out) == 1 and out[0].text == s.text and out[0].speaker == "S1"


def test_line_is_split_where_the_speaker_changes():
    s = seg("I propose we keep it. Yes, let's keep it.", 0, 9)  # 9 words, 1 s each
    out = assign_speakers([s], [Turn(0, 5, "S1"), Turn(5, 9, "S2")])
    assert [(o.speaker, o.text) for o in out] == [("S1", "I propose we keep it."), ("S2", "Yes, let's keep it.")]
    assert out[1].start == 5 and out[0].end == 5
    assert len(out[0].words) == 5 and len(out[1].words) == 4


def test_change_one_word_into_a_sentence_snaps_to_the_sentence_end():
    # Diarization switches one word late: "Yes," still sounds like S1 to it.
    s = seg("I propose we keep it. Yes, let's keep it.", 0, 9)
    out = assign_speakers([s], [Turn(0, 6, "S1"), Turn(6, 9, "S2")])
    assert [o.text for o in out] == ["I propose we keep it.", "Yes, let's keep it."]


def test_one_word_blip_inside_a_sentence_is_ignored():
    labels = smooth_labels("so we should really go".split(), ["S1", "S1", "S2", "S1", "S1"])
    assert labels == ["S1"] * 5


def test_short_whole_sentence_by_another_speaker_is_kept():
    labels = smooth_labels("Keep it. Yes. Good.".split(), ["S1", "S1", "S2", "S1"])
    assert labels == ["S1", "S1", "S2", "S1"]


def test_word_outside_every_turn_takes_a_close_turn_or_none():
    near = assign_speakers([seg("Okay.", 10.2, 10.6)], [Turn(0, 10, "S1")])
    far = assign_speakers([seg("Thanks for watching!", 30, 32)], [Turn(0, 10, "S1")])
    assert near[0].speaker == "S1" and far[0].speaker is None


def test_summary_counts_talk_time():
    segs = [seg("a b", 0, 2, "S1"), seg("c d", 2, 5, "S2"), seg("e f", 5, 6, "S1")]
    speakers = summarize(segs)
    assert [(s.id, s.talk_s, s.lines) for s in speakers] == [("S1", 3.0, 2), ("S2", 3.0, 1)]
    assert line_labels(segs, speakers) == ["Speaker 1", "Speaker 2", "Speaker 1"]


# ---------- Stage 2b: checking name claims ----------

LINES = ["Hi everyone, I'm Priya, let's start.",          # 0 S1
         "I propose we move the launch to May.",           # 1 S1
         "Rahul, what do you think?",                      # 2 S1
         "Sounds good to me.",                             # 3 S2
         "Thanks, Rahul. Sam, anything from design?",      # 4 S1
         "Not yet, I'll share mocks on Friday.",           # 5 S3
         "Priya said the budget is fixed."]                # 6 S2
WHO = ["S1", "S1", "S1", "S2", "S1", "S3", "S2"]


def claim(speaker, name, kind, line, quote):
    return {"speaker": speaker, "name": name, "kind": kind, "line": line, "quote": quote}


def check(c, doubtful=(), attendees=()):
    return check_claim(c, LINES, WHO, set(doubtful), list(attendees))


def test_self_introduction_is_accepted():
    ev = check(claim("S1", "Priya", "self", 0, "I'm Priya"))
    assert ev.accepted and ev.reason == "said their own name"


def test_self_introduction_by_someone_else_is_rejected():
    ev = check(claim("S2", "Priya", "self", 0, "I'm Priya"))
    assert not ev.accepted and "said by S1" in ev.reason


def test_addressed_person_who_answers_next_is_accepted():
    ev = check(claim("S2", "Rahul", "addressed", 2, "Rahul, what do you think"))
    assert ev.accepted and "answered right after" in ev.reason


def test_thanks_goes_to_the_one_who_just_spoke():
    ev = check(claim("S2", "Rahul", "addressed", 4, "Thanks, Rahul"))
    assert ev.accepted and "spoke just before" in ev.reason


def test_addressed_person_who_never_answers_is_rejected():
    ev = check(claim("S3", "Rahul", "addressed", 2, "Rahul, what do you think"))
    assert not ev.accepted and "neither spoke just before nor answered" in ev.reason


def test_cannot_address_yourself():
    assert not check(claim("S1", "Rahul", "addressed", 2, "Rahul, what do you think")).accepted


def test_invented_quote_and_missing_name_are_rejected():
    assert check(claim("S2", "Rahul", "addressed", 2, "Rahul, please take the notes")).reason == \
        "quote not found in the transcript"
    assert check(claim("S2", "Rahul", "addressed", 3, "Sounds good to me")).reason == \
        "the quote does not contain the name"


def test_not_a_name_and_doubtful_lines_are_rejected():
    assert check(claim("S2", "everyone", "addressed", 0, "Hi everyone")).reason == "not a person's name"
    assert not check(claim("S1", "Priya", "self", 0, "I'm Priya"), doubtful={0}).accepted


def test_wrong_line_number_is_tolerated():
    ev = check(claim("S1", "Priya", "self", 1, "I'm Priya"))
    assert ev.accepted and ev.line == 0


def test_heard_name_is_spelled_like_the_attendee():
    ev = check(claim("S1", "priya", "self", 0, "I'm Priya"), attendees=["Priya Sharma"])
    assert ev.name == "Priya Sharma"


def ev(speaker, name, kind="addressed", accepted=True):
    return NameEvidence(speaker, name, kind, 0, name, accepted)


def test_names_are_given_one_to_one_best_first():
    speakers = [Speaker("S1"), Speaker("S2"), Speaker("S3")]
    assign_names(speakers, [ev("S1", "Priya", "self"), ev("S2", "Priya"), ev("S2", "Rahul"),
                            ev("S2", "Rahul"), ev("S3", "Sam", accepted=False)])
    assert [(s.name, s.source, s.confidence) for s in speakers] == [
        ("Priya", "self-introduction", "high"), ("Rahul", "addressed", "high"), ("", "", "")]


def test_two_equally_likely_names_leave_the_speaker_unnamed():
    speakers = [Speaker("S1")]
    assign_names(speakers, [ev("S1", "Priya"), ev("S1", "Rahul")])
    assert speakers[0].name == "" and "Priya or Rahul" in speakers[0].note


def test_one_name_fitting_two_speakers_equally_is_not_given():
    speakers = [Speaker("S1"), Speaker("S2")]
    assign_names(speakers, [ev("S1", "Sam"), ev("S2", "Sam")])
    assert [s.name for s in speakers] == ["", ""] and "unclear" in speakers[0].note


def test_edited_names_are_kept_when_naming_again():
    speakers = [Speaker("S1", "Dev", "edited")]
    assign_names(speakers, [ev("S1", "Priya", "self")])
    assert speakers[0].name == "Dev"


def test_name_speakers_calls_the_llm_and_checks_every_claim():
    import json

    def fake_llm(system, user):
        assert "[0] S1: Hi everyone, I'm Priya" in user
        return json.dumps({"names": [claim("S1", "Priya", "self", 0, "I'm Priya"),
                                     claim("S2", "Rahul", "addressed", 2, "Rahul, what do you think"),
                                     claim("S2", "Priya", "addressed", 6, "Priya said the budget")]})

    speakers = summarize([seg(t, i, i + 1, w) for i, (t, w) in enumerate(zip(LINES, WHO))])
    evidence = name_speakers(LINES, WHO, speakers, fake_llm)
    assert [e.accepted for e in evidence] == [True, True, False]  # third person mention is no evidence
    assert [s.label for s in speakers] == ["Priya", "Rahul", "Speaker 3"]


# ---------- Editing names ----------

def test_rename_reports_label_changes_and_allows_reset():
    speakers = [Speaker("S1", "Priya", "self-introduction", "high"), Speaker("S2")]
    changes = rename_speakers(speakers, {"S1": "Priya Sharma", "S2": "Rahul"})
    assert changes == {"Priya": "Priya Sharma", "Speaker 2": "Rahul"}
    assert speakers[0].source == "edited" and speakers[0].confidence == ""
    assert rename_speakers(speakers, {"S2": ""}) == {"Rahul": "Speaker 2"}


def test_replace_labels_is_whole_word_and_handles_swaps():
    text = "Speaker 2 agreed with Speaker 21; Priya and Rahul left."
    assert replace_labels(text, {"Speaker 2": "Sam"}) == "Sam agreed with Speaker 21; Priya and Rahul left."
    assert replace_labels(text, {"Priya": "Rahul", "Rahul": "Priya"}).endswith("Rahul and Priya left.")


def test_backchannel_by_a_third_person_is_not_the_answer():
    lines = ["Rahul, what do you think?", "Mm-hmm.", "I think May works."]
    who = ["S1", "S3", "S2"]
    good = check_claim(claim("S2", "Rahul", "addressed", 0, "Rahul, what do you think"), lines, who, set(), [])
    bad = check_claim(claim("S3", "Rahul", "addressed", 0, "Rahul, what do you think"), lines, who, set(), [])
    assert good.accepted and not bad.accepted


def test_first_name_and_full_name_count_as_one_person():
    speakers = [Speaker("S1")]
    assign_names(speakers, [ev("S1", "Bart", "self"), ev("S1", "Bart Beute", "self"), ev("S1", "Richard", "self")])
    assert speakers[0].name == "Bart Beute" and speakers[0].confidence == "high"


# ---------- Roles ----------

from meeting_assistant.speakers import assign_roles, clean_role, role_mentioned  # noqa: E402

INTRO = ["I'm Nick, I'm the industrial designer.",   # 0 S1
         "And I'm the marketing expert.",            # 1 S2
         "The industrial designer will do the working design."]  # 2 S3
INTRO_WHO = ["S1", "S2", "S3"]


def role_claim(speaker, role, line, quote, kind="role", name=""):
    return {"speaker": speaker, "name": role if kind == "role" else name, "kind": kind, "line": line,
            "quote": quote}


def test_role_said_by_the_speaker_is_accepted():
    ev = check_claim(role_claim("S2", "the Marketing Expert", 1, "I'm the marketing expert"),
                     INTRO, INTRO_WHO, set(), [])
    assert ev.accepted and ev.name == "marketing expert"


def test_role_said_about_someone_else_is_rejected():
    ev = check_claim(role_claim("S1", "industrial designer", 2, "The industrial designer will do"),
                     INTRO, INTRO_WHO, set(), [])
    assert not ev.accepted and "said by S3" in ev.reason


def test_role_not_in_the_quote_is_rejected():
    ev = check_claim(role_claim("S1", "project manager", 0, "I'm Nick, I'm the industrial designer"),
                     INTRO, INTRO_WHO, set(), [])
    assert not ev.accepted and "does not contain the role" in ev.reason


def test_name_speakers_finds_names_and_roles_from_one_introduction():
    import json

    def fake_llm(system, user):
        return json.dumps({"names": [
            {"speaker": "S1", "name": "Nick", "role": "industrial designer", "kind": "self", "line": 0,
             "quote": "I'm Nick, I'm the industrial designer"},
            {"speaker": "S2", "name": "", "role": "marketing expert", "kind": "role", "line": 1,
             "quote": "I'm the marketing expert"}]})

    speakers = summarize([seg(t, i, i + 1, w) for i, (t, w) in enumerate(zip(INTRO, INTRO_WHO))])
    name_speakers(INTRO, INTRO_WHO, speakers, fake_llm)
    assert [(s.label, s.role) for s in speakers] == [("Nick", "industrial designer"),
                                                    ("Speaker 2", "marketing expert"), ("Speaker 3", "")]


def test_one_role_per_speaker_and_edited_roles_kept():
    speakers = [Speaker("S1"), Speaker("S2", role="designer", role_source="edited")]
    assign_roles(speakers, [ev("S1", "project manager", "role"), ev("S1", "project manager", "role"),
                            ev("S1", "designer", "role"), ev("S2", "project manager", "role")])
    assert [s.role for s in speakers] == ["project manager", "designer"]


def test_role_mentions():
    assert role_mentioned("industrial designer", "The industrial designer will do the working design")
    assert role_mentioned("marketing manager", "and the marketing expert prepares the requirements")
    assert not role_mentioned("project manager", "the project will start next week")
    assert clean_role("The User Interface Designer.") == "user interface designer"


def test_roles_can_be_edited():
    speakers = [Speaker("S1", "Nick", role="industrial designer", role_source="self")]
    rename_speakers(speakers, {"S1": "Nick"}, {"S1": "Lead Designer"})
    assert (speakers[0].role, speakers[0].role_source) == ("lead designer", "edited")


def test_role_given_by_name_goes_to_that_person():
    import json

    lines = ["I'm Mandy, the project manager.", "Courtney, you're our marketing person.", "Yes, that's me.",
             "Sam is our industrial designer."]
    who = ["S1", "S1", "S2", "S1"]

    def fake_llm(system, user):
        return json.dumps({"names": [
            {"speaker": "S1", "name": "Mandy", "role": "project manager", "kind": "self", "line": 0,
             "quote": "I'm Mandy, the project manager"},
            {"speaker": "S2", "name": "Courtney", "role": "", "kind": "addressed", "line": 1,
             "quote": "Courtney, you're our marketing person"},
            {"speaker": "S2", "name": "Courtney", "role": "marketing person", "kind": "assigned_role", "line": 1,
             "quote": "Courtney, you're our marketing person"},
            {"speaker": "", "name": "Sam", "role": "industrial designer", "kind": "assigned_role", "line": 3,
             "quote": "Sam is our industrial designer"},
            {"speaker": "", "name": "Sam", "role": "chef", "kind": "assigned_role", "line": 3,
             "quote": "Sam is our industrial designer"}]})

    speakers = summarize([seg(t, i, i + 1, w) for i, (t, w) in enumerate(zip(lines, who))])
    evidence = name_speakers(lines, who, speakers, fake_llm)
    assert [(s.label, s.role, s.role_source) for s in speakers] == [
        ("Mandy", "project manager", "self"), ("Courtney", "marketing person", "named by others")]
    from meeting_assistant.speakers import named_roles
    assert named_roles(evidence) == {"Courtney": "marketing person", "Sam": "industrial designer"}
    assert [e.reason for e in evidence if e.kind == "assigned_role" and not e.accepted] == [
        "the quote does not contain the role"]


def test_unclear_speaker_still_blocks_a_tied_name():
    # S1 is unclear (Priya or Rahul); S2 fits Priya exactly as well as S1 does,
    # so Priya must not go to S2 either.
    speakers = [Speaker("S1"), Speaker("S2")]
    assign_names(speakers, [ev("S1", "Priya"), ev("S1", "Rahul"), ev("S2", "Priya")])
    assert [s.name for s in speakers] == ["", ""]
    assert "unclear" in speakers[1].note


def test_speaker_already_named_does_not_block_another():
    speakers = [Speaker("S1"), Speaker("S2")]
    assign_names(speakers, [ev("S1", "Rahul", "self"), ev("S1", "Priya"), ev("S2", "Priya")])
    assert [s.name for s in speakers] == ["Rahul", "Priya"]
