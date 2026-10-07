import json

from meeting_assistant.minutes import (
    ActionItemEvidence, DecisionEvidence, MinutesDraft, QuoteIndex, to_json, to_markdown,
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


def draft(decisions=(), actions=()):
    return MinutesDraft(summary="Design meeting.", minutes=["Remote control design"],
                        decisions=list(decisions), action_items=list(actions))


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
