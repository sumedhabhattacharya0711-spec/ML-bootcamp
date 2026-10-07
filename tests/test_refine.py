import json
from types import SimpleNamespace

from meeting_assistant.glossary import Term
from meeting_assistant.refine import candidate_spans, find_hints, guard_line, refine, score_match

GLOSSARY = ["Kubeflow", "ONNX", "LoRA", "GDPR", "Grafana"]


def fake_llm(lines=None, possible_errors=None):
    """Stand-in for llm.call_llm. Records what it was sent."""
    def call(system, user):
        call.sent.append(user)
        return json.dumps({"lines": lines or {}, "possible_errors": possible_errors or []})
    call.sent = []
    return call


def seg(text, probs=None):
    words = [SimpleNamespace(probability=p) for p in probs] if probs else []
    return SimpleNamespace(text=text, words=words)


# ---------- scoring and hints ----------

def test_cube_flow_scores_high_for_kubeflow():
    sound, spelling, score = score_match("cube flow", "Kubeflow")
    assert sound == 100 and score > 90


def test_revenue_does_not_match_grafana():
    assert score_match("revenue", "Grafana")[2] < 72
    assert find_hints(["Our revenue grew last quarter."], ["Grafana"]) == []


def test_spans_never_cross_a_sentence_end():
    tokens = "we use cube. flow is next".split()
    spans = list(candidate_spans(tokens))
    assert (2, 3) in spans          # "cube."
    assert (2, 4) not in spans      # "cube. flow" crosses the full stop


def test_hint_found_for_misheard_term():
    hints = find_hints(["We moved the training jobs to cube flow last week."], GLOSSARY)
    assert [(h.heard, h.term) for h in hints] == [("cube flow", "Kubeflow")]


def test_correct_term_gives_no_hint():
    assert find_hints(["The dashboards are in Grafana."], GLOSSARY) == []


def test_unsure_word_uses_lower_threshold():
    # "point" vs Pound scores 80: no hint normally (needs > 82),
    # a hint when Whisper was unsure of that word (needs > 75).
    assert 75 < score_match("point", "Pound")[2] <= 82
    sure = find_hints([seg("a key point now", [0.9, 0.9, 0.9, 0.9])], ["Pound"])
    unsure = find_hints([seg("a key point now", [0.9, 0.9, 0.2, 0.9])], ["Pound"])
    assert sure == []
    assert len(unsure) == 1 and unsure[0].unsure


def test_everyday_sound_alikes_from_es2004a_are_not_hinted():
    lines = ["that was my main point, we do have to use metal",
             "which makes it fairly obvious what you're trying to do.",
             "to the other functions when you can do sound or options"]
    assert find_hints(lines, ["Pound", "Market", "Functional design"]) == []


def test_hints_capped_and_non_overlapping():
    lines = ["we use cube flow here."] * 30
    hints = find_hints(lines, GLOSSARY)
    assert len(hints) == 15
    assert len({(h.line, h.start) for h in hints}) == 15


def test_doubtful_lines_get_no_hints():
    assert find_hints(["move jobs to cube flow."], GLOSSARY, doubtful={0}) == []


# ---------- guard ----------

def test_guard_allows_glossary_fix():
    line, edits = guard_line(0, "jobs run on cube flow now.", "jobs run on Kubeflow now.", GLOSSARY, [(3, 5)])
    assert line == "jobs run on Kubeflow now."
    assert edits[0].status == "applied"


def test_guard_blocks_not_to_now():
    line, edits = guard_line(0, "we will not ship it.", "we will now ship it.", GLOSSARY, [(2, 3)])
    assert line == "we will not ship it."
    assert edits[0].status == "blocked" and edits[0].reason == "touches a negation"


def test_guard_blocks_15th_to_16th():
    line, edits = guard_line(0, "due on the 15th.", "due on the 16th.", GLOSSARY, [(3, 4)])
    assert line == "due on the 15th."
    assert edits[0].reason == "touches a number"


def test_guard_blocks_non_glossary_rewrite():
    line, edits = guard_line(0, "we kind of like it.", "we really like it.", GLOSSARY, [(1, 3)])
    assert line == "we kind of like it."
    assert edits[0].reason == "result is not a glossary term"


def test_guard_ignores_punctuation_only_changes():
    line, edits = guard_line(0, "ok so cube flow", "OK, so Kubeflow.", GLOSSARY, [(2, 4)])
    assert line == "ok so Kubeflow."
    assert [e.before for e in edits] == ["cube flow"]


# ---------- full refine ----------

def test_no_hints_means_no_llm_call():
    llm = fake_llm()
    result = refine(["Let's talk about the budget."], GLOSSARY, llm)
    assert llm.sent == [] and not result.llm_called
    assert result.refined == ["Let's talk about the budget."]


def test_refine_applies_safe_edit_and_blocks_unsafe_one():
    segments = ["We fine-tune with Laura adapters.", "We will not use cube flow."]
    llm = fake_llm(lines={"0": "We fine-tune with LoRA adapters.", "1": "We will now use Kubeflow."})
    result = refine(segments, GLOSSARY, llm)
    assert result.refined == ["We fine-tune with LoRA adapters.", "We will not use Kubeflow."]
    statuses = {(e.before, e.status) for e in result.edits}
    assert ("not", "blocked") in statuses and ("cube flow.", "applied") in statuses


def test_laura_from_sales_left_alone_when_llm_keeps_it():
    llm = fake_llm(lines={})
    result = refine(["Laura from sales will call."], ["LoRA"], llm)
    assert result.llm_called
    assert result.refined == ["Laura from sales will call."]


def test_only_hinted_lines_and_neighbours_are_sent():
    segments = ["intro.", "budget talk.", "jobs go to cube flow.", "more budget.", "far away line."]
    llm = fake_llm()
    refine(segments, GLOSSARY, llm)
    sent = json.loads(llm.sent[0])
    assert set(sent["lines"]) == {"1", "2", "3"}
    assert sent["hints"][0]["term"] == "Kubeflow"


def test_possible_errors_kept_only_if_in_raw_line():
    llm = fake_llm(possible_errors=[
        {"line": 0, "text": "pie torch", "reason": "maybe PyTorch"},
        {"line": 0, "text": "TensorFlow", "reason": "invented"},
    ])
    result = refine(["we use cube flow and pie torch."], GLOSSARY, llm)
    assert [p.text for p in result.possible_errors] == ["pie torch"]
    assert result.refined == ["we use cube flow and pie torch."]  # flags never applied


def test_bad_llm_reply_keeps_raw_transcript():
    result = refine(["jobs go to cube flow."], GLOSSARY, lambda s, u: "sorry, no JSON here")
    assert result.refined == ["jobs go to cube flow."]
    assert "Refinement skipped" in result.warnings[0]


def test_accepts_glossary_terms_objects():
    hints = find_hints(["jobs go to cube flow."], [Term("Kubeflow", "user")])
    assert hints[0].term == "Kubeflow"


def test_everyday_words_are_not_hinted():
    lines = ["The cat could not find the app.", "What kind of granny remote?"]
    assert find_hints(lines, ["CUDA", "API", "kinetic"]) == []


def test_short_spoken_forms_still_hinted():
    pairs = [("we export to on x format.", "ONNX"), ("check G P U memory.", "GPU"),
             ("follow G D P R rules.", "GDPR"), ("written in pie torch.", "PyTorch")]
    for line, term in pairs:
        assert [h.term for h in find_hints([line], [term])] == [term], line


def test_correct_multi_word_term_gives_no_hint():
    # Same words, different capitals and punctuation: nothing to fix.
    assert find_hints(["I'm Sarah, project manager, and this is our meeting."], ["Project manager"]) == []
    assert [h.term for h in find_hints(["follow G D P R rules."], ["GDPR"])] == ["GDPR"]


def test_guard_blocks_edit_outside_the_hinted_words():
    # Hint was only "mean," (word 4); the LLM also swallowed "I" (word 3).
    raw = "that's the main stuff anyway. I mean, you don't want to"
    line, edits = guard_line(0, raw, "that's the main stuff anyway. menu you don't want to", ["menu"], [(4, 5)])
    assert line == raw
    assert edits[0].reason == "changes words that were not hinted"


def test_mean_is_not_hinted_as_menu():
    assert find_hints(["I mean, you don't want to."], ["menu"]) == []
