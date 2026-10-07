import json

from meeting_assistant.minutes import (
    ActionItemEvidence, DecisionEvidence, MinutesDraft, OpenQuestionEvidence, QuoteIndex, to_json, to_markdown,
    format_transcript, support_rate, verify, write_minutes,
)

SEGMENTS = [
    "Okay, let's start. First item is the remote control design.",
    "I propose we make the case out of rubber.",
    "Yes, rubber is a good idea, let's go with that.",
    "Maybe we should add a speech recognition feature?",
    "Mm-hmm.",
    "And what about a solar panel?",
    "No, a solar panel is too expensive, let's drop that.",
    "Sarah will send the cost spreadsheet by Friday.",
    "Someone should also check the battery supplier.",
    "Thanks for watching!",
]
STARTS = [10.0 * i for i in range(len(SEGMENTS))]
EVIDENCE = QuoteIndex(list(enumerate(SEGMENTS[:-1])), STARTS)  # segment 9 is doubtful


def decision(text, proposal, agree=(), reject=()):
    return DecisionEvidence(text=text, proposal_quote=proposal,
                            agreement_quotes=list(agree), rejection_quotes=list(reject))


def action(task, quote, owner="unspecified", owner_quote="", deadline="unspecified", deadline_quote="",
           agreement_quote=""):
    return ActionItemEvidence(task=task, task_quote=quote, owner=owner, owner_quote=owner_quote,
                              deadline=deadline, deadline_quote=deadline_quote,
                              agreement_quote=agreement_quote)


def draft(decisions=(), actions=(), questions=()):
    return MinutesDraft(summary="Design meeting.", minutes=["Remote control design"],
                        decisions=list(decisions), action_items=list(actions),
                        open_questions=list(questions))


def test_proposal_with_agreement_is_agreed():
    d = decision("Rubber case", "I propose we make the case out of rubber",
                 agree=["rubber is a good idea, let's go with that"])
    result = verify(draft([d]), EVIDENCE)
    assert result.minutes.decisions[0].status == "agreed"


def test_lone_mm_hmm_is_not_agreement():
    d = decision("Speech recognition", "Maybe we should add a speech recognition feature", agree=["Mm-hmm."])
    result = verify(draft([d]), EVIDENCE)
    assert result.minutes.decisions[0].status == "open"
    assert any("backchannel" in note for note in result.dropped)


def test_rejection_is_rejected():
    d = decision("Solar panel", "And what about a solar panel",
                 reject=["a solar panel is too expensive, let's drop that"])
    assert verify(draft([d]), EVIDENCE).minutes.decisions[0].status == "rejected"


def test_invented_proposal_quote_is_dropped():
    d = decision("Titanium case", "We agreed to use a titanium case", agree=["great idea"])
    result = verify(draft([d]), EVIDENCE)
    assert result.minutes.decisions == []
    assert "Decision dropped" in result.dropped[0]


def test_invented_agreement_does_not_upgrade_proposal():
    d = decision("Speech recognition", "Maybe we should add a speech recognition feature",
                 agree=["Everyone loved the speech recognition idea"])
    assert verify(draft([d]), EVIDENCE).minutes.decisions[0].status == "open"


def test_action_item_with_owner_and_deadline():
    a = action("Send cost spreadsheet", "Sarah will send the cost spreadsheet by Friday",
               owner="Sarah", owner_quote="Sarah will send the cost spreadsheet",
               deadline="by Friday", deadline_quote="send the cost spreadsheet by Friday")
    item = verify(draft(actions=[a]), EVIDENCE).minutes.action_items[0]
    assert (item.owner, item.deadline) == ("Sarah", "by Friday")


def test_missing_owner_becomes_unspecified():
    a = action("Check battery supplier", "Someone should also check the battery supplier",
               owner="John", owner_quote="John will check it")  # invented owner
    result = verify(draft(actions=[a]), EVIDENCE)
    assert result.minutes.action_items[0].owner == "unspecified"
    assert any('owner "John"' in note for note in result.dropped)


def test_invented_action_item_is_dropped():
    a = action("Book a venue", "Tom will book the venue for the launch")
    assert verify(draft(actions=[a]), EVIDENCE).minutes.action_items == []


def test_doubtful_lines_are_marked_and_not_evidence():
    for_llm, evidence = format_transcript(SEGMENTS, doubtful={9})
    assert "[9] [DOUBTFUL] Thanks for watching!" in for_llm
    assert all(line != 9 for line, _ in evidence)


def test_quote_from_doubtful_line_does_not_count():
    a = action("Like the video", "Thanks for watching!")
    result = verify(draft(actions=[a]), QuoteIndex(format_transcript(SEGMENTS, {9})[1]))
    assert result.minutes.action_items == []


def test_write_minutes_with_fake_llm():
    sent = {}

    def fake_structured(system, user, schema):
        sent["user"] = user
        assert schema is MinutesDraft
        return draft([decision("Rubber case", "I propose we make the case out of rubber",
                               agree=["let's go with that"])])

    result = write_minutes(SEGMENTS, fake_structured, doubtful={9})
    assert result.minutes.decisions[0].status == "agreed"
    assert "[9] [DOUBTFUL]" in sent["user"]


def test_markdown_and_json_have_same_content():
    d = decision("Rubber case", "I propose we make the case out of rubber", agree=["let's go with that"])
    a = action("Send cost spreadsheet", "Sarah will send the cost spreadsheet by Friday",
               owner="Sarah", owner_quote="Sarah will send", deadline="by Friday",
               deadline_quote="spreadsheet by Friday")
    m = verify(draft([d], [a]), EVIDENCE).minutes
    md, js = to_markdown(m), json.loads(to_json(m))
    for value in [js["summary"], js["decisions"][0]["text"], js["decisions"][0]["status"],
                  js["action_items"][0]["owner"], js["action_items"][0]["deadline"]]:
        assert value in md


def test_decision_evidence_has_every_quote_with_timestamps():
    d = decision("Rubber case", "I propose we make the case out of rubber",
                 agree=["rubber is a good idea, let's go with that"])
    ev = verify(draft([d]), EVIDENCE).minutes.decisions[0].evidence
    assert [(e.role, e.line, e.start) for e in ev] == [("proposal", 1, 10.0), ("agreement", 2, 20.0)]


def test_action_status_agreed_proposed_unassigned():
    agreed = action("Send cost spreadsheet", "Sarah will send the cost spreadsheet by Friday",
                    owner="Sarah", owner_quote="Sarah will send the cost spreadsheet",
                    agreement_quote="Sarah will send the cost spreadsheet by Friday")
    proposed = action("Send cost spreadsheet", "Sarah will send the cost spreadsheet by Friday",
                      owner="Sarah", owner_quote="Sarah will send the cost spreadsheet")
    unassigned = action("Check battery supplier", "Someone should also check the battery supplier",
                        agreement_quote="Mm-hmm.")
    items = verify(draft(actions=[agreed, proposed, unassigned]), EVIDENCE).minutes.action_items
    assert [a.status for a in items] == ["agreed", "proposed", "unassigned"]


def test_invented_agreement_quote_keeps_action_proposed():
    a = action("Send cost spreadsheet", "Sarah will send the cost spreadsheet by Friday",
               owner="Sarah", owner_quote="Sarah will send the cost spreadsheet",
               agreement_quote="Sarah said she would love to do it")
    result = verify(draft(actions=[a]), EVIDENCE)
    assert result.minutes.action_items[0].status == "proposed"
    assert any("agreement quote not verified" in n for n in result.dropped)


def test_checks_count_what_was_kept_and_removed():
    d_ok = decision("Rubber case", "I propose we make the case out of rubber", agree=["let's go with that"])
    d_bad = decision("Titanium case", "We agreed to use a titanium case")
    a = action("Check battery supplier", "Someone should also check the battery supplier",
               owner="John", owner_quote="John will check it")
    checks = verify(draft([d_ok, d_bad], [a]), EVIDENCE).checks
    assert (checks["decisions_proposed"], checks["decisions_kept"], checks["decisions_agreed"]) == (2, 1, 1)
    assert checks["owners_removed"] == 1
    assert support_rate(checks) == round(2 / 3, 3)


def test_open_question_kept_with_timestamp_and_invented_one_dropped():
    real = OpenQuestionEvidence(question="Battery supplier not chosen yet",
                                quote="Someone should also check the battery supplier")
    fake = OpenQuestionEvidence(question="Budget unclear", quote="We still need to settle the budget")
    result = verify(draft(questions=[real, fake]), EVIDENCE)
    qs = result.minutes.open_questions
    assert [q.question for q in qs] == ["Battery supplier not chosen yet"]
    assert (qs[0].evidence[0].line, qs[0].evidence[0].start) == (8, 80.0)
    assert result.checks["open_questions_proposed"] == 2 and result.checks["open_questions_kept"] == 1
    assert "## Open questions" in to_markdown(result.minutes)


def test_long_transcript_is_written_in_parts_and_merged():
    from meeting_assistant.minutes import merge_drafts  # noqa: F401  (imported to show it exists)
    sent = []

    def fake_structured(system, user, schema):
        sent.append(user)
        if "part 1 of" in user:
            return MinutesDraft(summary="Case material discussed.", minutes=["Case material"],
                                decisions=[decision("Rubber case", "I propose we make the case out of rubber")],
                                action_items=[], open_questions=[])
        return MinutesDraft(summary="Costs and suppliers discussed.", minutes=["Case material", "Costs"],
                            decisions=[decision("Rubber case", "I propose we make the case out of rubber",
                                                agree=["rubber is a good idea, let's go with that"])],
                            action_items=[action("Send cost spreadsheet",
                                                 "Sarah will send the cost spreadsheet by Friday")],
                            open_questions=[])

    merged_summary = []
    result = write_minutes(SEGMENTS, fake_structured, doubtful={9}, starts=STARTS, max_tokens=40,
                           call_text=lambda s, u: merged_summary.append(u) or "Whole meeting summary.")
    assert result.checks["parts"] == len(sent) >= 2
    assert "EARLIER PARTS" in sent[1] and "Case material discussed." in sent[1]
    m = result.minutes
    assert [d.status for d in m.decisions] == ["agreed"]      # proposal and agreement merged
    assert m.minutes == ["Case material", "Costs"]            # duplicate topic line merged
    assert m.summary == "Whole meeting summary." and merged_summary


def test_part_summary_merge_falls_back_when_llm_fails():
    from meeting_assistant.minutes import merge_drafts
    a = MinutesDraft(summary="First half.", minutes=[], decisions=[], action_items=[], open_questions=[])
    b = MinutesDraft(summary="Second half.", minutes=[], decisions=[], action_items=[], open_questions=[])

    def broken(system, user):
        raise RuntimeError("rate limited")

    assert merge_drafts([a, b], broken).summary == "First half. Second half."


# ---------- With speakers (diarization) ----------

TALK = ["I propose we keep the rubber case.",      # 0 Priya
        "Yes, I agree, rubber it is.",             # 1 Priya (the proposer agreeing with herself)
        "Rubber sounds right to me.",              # 2 Rahul
        "Someone has to send the cost sheet.",     # 3 Priya
        "I'll send the cost sheet by Friday.",     # 4 Rahul
        "Sam will check the supplier.",            # 5 Priya
        "Okay, fine."]                             # 6 Rahul
WHO = ["Priya", "Priya", "Rahul", "Priya", "Rahul", "Priya", "Rahul"]
TALK_INDEX = QuoteIndex(list(enumerate(TALK)), [5.0 * i for i in range(len(TALK))], WHO)


def test_agreement_by_the_proposer_themself_does_not_count():
    alone = decision("Rubber case", "I propose we keep the rubber case", agree=["Yes, I agree, rubber it is"])
    result = verify(draft([alone]), TALK_INDEX)
    assert result.minutes.decisions[0].status == "open"
    assert any("by the proposer (Priya) themself" in n for n in result.dropped)
    other = decision("Rubber case", "I propose we keep the rubber case", agree=["Rubber sounds right to me"])
    assert verify(draft([other]), TALK_INDEX).minutes.decisions[0].status == "agreed"


def test_i_will_do_it_makes_its_speaker_the_owner():
    a = action("Send the cost sheet", "Someone has to send the cost sheet",
               agreement_quote="I'll send the cost sheet by Friday",
               deadline="by Friday", deadline_quote="I'll send the cost sheet by Friday")
    result = verify(draft(actions=[a]), TALK_INDEX)
    item = result.minutes.action_items[0]
    assert (item.owner, item.deadline, item.status) == ("Rahul", "by Friday", "agreed")
    assert result.checks["owners_from_speaker"] == 1
    assert {e.speaker for e in item.evidence} == {"Priya", "Rahul"}


def test_owner_given_as_the_volunteering_speaker_is_supported():
    a = action("Send the cost sheet", "I'll send the cost sheet", owner="Rahul",
               owner_quote="I'll send the cost sheet by Friday")
    item = verify(draft(actions=[a]), TALK_INDEX).minutes.action_items[0]
    assert item.owner == "Rahul"


def test_volunteer_line_by_someone_else_does_not_support_the_owner():
    a = action("Send the cost sheet", "I'll send the cost sheet", owner="Priya",
               owner_quote="I'll send the cost sheet by Friday")
    result = verify(draft(actions=[a]), TALK_INDEX)
    assert result.minutes.action_items[0].owner == "Rahul"  # the one who said "I'll" gets it
    assert any('owner "Priya" not supported' in n for n in result.dropped)


def test_acceptance_by_someone_other_than_the_owner_is_only_proposed():
    a = action("Check the supplier", "Sam will check the supplier", owner="Sam",
               owner_quote="Sam will check the supplier", agreement_quote="Okay, fine.")
    item = verify(draft(actions=[a]), TALK_INDEX).minutes.action_items[0]
    assert (item.owner, item.status) == ("Sam", "agreed")  # Sam is no known speaker: no speaker check
    rahul = action("Check the supplier", "Sam will check the supplier", owner="Priya",
                   owner_quote="Someone has to send the cost sheet", agreement_quote="Rubber sounds right to me")
    # owner "Priya" is not in her quote and she did not volunteer: unspecified, so unassigned
    assert verify(draft(actions=[rahul]), TALK_INDEX).minutes.action_items[0].status == "unassigned"


def test_speaker_label_copied_into_a_quote_is_tolerated():
    d = decision("Rubber case", "Priya: I propose we keep the rubber case", agree=["Rahul: Rubber sounds right"])
    assert verify(draft([d]), TALK_INDEX).minutes.decisions[0].status == "agreed"


def test_llm_sees_speakers_and_markdown_shows_them():
    text, _ = format_transcript(TALK, {6}, WHO)
    assert text.split("\n")[0] == "[0] Priya: I propose we keep the rubber case."
    assert text.split("\n")[6] == "[6] [DOUBTFUL] Rahul: Okay, fine."
    d = decision("Rubber case", "I propose we keep the rubber case", agree=["Rubber sounds right to me"])
    md = to_markdown(verify(draft([d]), TALK_INDEX).minutes)
    assert 'Rahul, agreement: _"Rubber sounds right to me"_' in md


def test_rename_in_minutes_changes_names_but_not_quotes():
    from meeting_assistant.minutes import rename_in_minutes

    a = action("Rahul sends the cost sheet", "Someone has to send the cost sheet",
               agreement_quote="I'll send the cost sheet by Friday")
    m = verify(draft(actions=[a]), TALK_INDEX).minutes
    m.summary = "Rahul will send the sheet."
    new_who = ["Priya", "Priya", "Rahul V.", "Priya", "Rahul V.", "Priya", "Rahul V."]
    rename_in_minutes(m, {"Rahul": "Rahul V."}, new_who)
    item = m.action_items[0]
    assert (m.summary, item.task, item.owner) == ("Rahul V. will send the sheet.",
                                                  "Rahul V. sends the cost sheet", "Rahul V.")
    assert {e.speaker for e in item.evidence} == {"Priya", "Rahul V."}
    assert m.participants == ["Priya", "Rahul V."]


def test_label_copied_into_owner_quote_is_not_support():
    a = action("Send the cost sheet", "Someone has to send the cost sheet", owner="Rahul",
               owner_quote="Rahul: Rubber sounds right to me.")
    result = verify(draft(actions=[a]), TALK_INDEX)
    assert result.minutes.action_items[0].owner == "unspecified"
    assert all(not e.quote.startswith("Rahul:") for e in result.minutes.action_items[0].evidence)
