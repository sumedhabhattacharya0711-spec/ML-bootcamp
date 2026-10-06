import json

from meeting_assistant.glossary import (
    Term,
    build_glossary,
    infer_terms,
    list_packs,
    load_packs,
    merge_terms,
    parse_user_terms,
    to_initial_prompt,
)

RAW = (
    "so we moved the training jobs to cube flow last week. "
    "the dashboards are in Grafana and the model export uses on x. "
    "Laura from sales will send the numbers by the 15th."
)


def fake_llm(reply):
    """A stand-in for llm.call_llm that always returns the given reply."""
    def call(system, user):
        return reply if isinstance(reply, str) else json.dumps(reply)
    return call


def texts(terms):
    return [t.text for t in terms]


def test_parse_user_terms_splits_and_trims():
    terms = parse_user_terms(" Kubeflow, ONNX ;\nPriya\n\n , ")
    assert texts(terms) == ["Kubeflow", "ONNX", "Priya"]
    assert all(t.source == "user" for t in terms)


def test_parse_user_terms_empty():
    assert parse_user_terms("") == []
    assert parse_user_terms(None) == []


def test_example_pack_loads():
    assert "ml_tech" in list_packs()
    terms = load_packs(["ml_tech"])
    assert "Kubeflow" in texts(terms)
    assert not any(t.text.startswith("#") for t in terms)


def test_infer_keeps_terms_found_in_transcript():
    reply = {"terms": [
        {"term": "Kubeflow", "heard_as": "cube flow"},
        {"term": "Grafana", "heard_as": "Grafana"},
        {"term": "ONNX", "heard_as": "on x"},
    ]}
    terms = infer_terms(RAW, fake_llm(reply))
    assert texts(terms) == ["Kubeflow", "Grafana", "ONNX"]
    assert all(t.source == "inferred" for t in terms)


def test_infer_drops_invented_terms():
    reply = {"terms": [
        {"term": "Kubeflow", "heard_as": "cube flow"},
        {"term": "Snowflake", "heard_as": "snow flake"},  # never said
    ]}
    assert texts(infer_terms(RAW, fake_llm(reply))) == ["Kubeflow"]


def test_infer_accepts_json_inside_code_fence():
    reply = '```json\n{"terms": [{"term": "Kubeflow", "heard_as": "cube flow"}]}\n```'
    assert texts(infer_terms(RAW, fake_llm(reply))) == ["Kubeflow"]


def test_merge_user_spelling_wins_over_inferred_and_pack():
    merged = merge_terms(
        [Term("kubeflow", "pack")],
        [Term("KubeFlow", "inferred")],
        [Term("Kubeflow", "user")],
    )
    assert len(merged) == 1
    assert merged[0].text == "Kubeflow" and merged[0].source == "user"


def test_build_glossary_survives_bad_llm_reply():
    terms, warnings = build_glossary("Priya", RAW, fake_llm("sorry, I can't help"), packs=None)
    assert texts(terms) == ["Priya"]
    assert len(warnings) == 1 and "inference failed" in warnings[0]


def test_build_glossary_without_llm_skips_inference():
    terms, warnings = build_glossary("Priya, ONNX", RAW, call_llm=None, packs=["ml_tech"])
    assert texts(terms)[:2] == ["Priya", "ONNX"]  # user terms first
    assert texts(terms).count("ONNX") == 1  # the pack's copy was deduplicated
    assert warnings == []


def test_initial_prompt_format_and_cap():
    terms = [Term(f"term{i}", "pack") for i in range(100)]
    prompt = to_initial_prompt([Term("Kubeflow", "user")] + terms, max_terms=10)
    parts = prompt.split(", ")
    assert len(parts) == 10
    assert parts[0] == "Kubeflow"  # most trusted first


def test_initial_prompt_never_cuts_a_term_in_half():
    terms = [Term(f"{'A' * 30}{i}", "user") for i in range(5)]
    prompt = to_initial_prompt(terms, max_chars=70)
    assert prompt.split(", ") == [terms[0].text, terms[1].text]


def test_infer_short_heard_as_must_match_exactly():
    reply = {"terms": [{"term": "AWS", "heard_as": "a ws"}]}  # not in transcript
    assert infer_terms(RAW, fake_llm(reply)) == []


def test_doubtful_segments_are_not_sent_to_llm():
    from meeting_assistant.glossary import transcript_for_inference
    segments = ["we moved jobs to cube flow", "thanks for watching, subscribe to NordVPN", "see you"]
    assert transcript_for_inference(segments, {1}) == "we moved jobs to cube flow see you"


def test_term_only_heard_in_doubtful_segment_is_dropped():
    segments = ["we moved the training jobs to cube flow", "this video is sponsored by NordVPN"]
    reply = {"terms": [
        {"term": "Kubeflow", "heard_as": "cube flow"},
        {"term": "NordVPN", "heard_as": "NordVPN"},
    ]}
    sent = []

    def spy_llm(system, user):
        sent.append(user)
        return json.dumps(reply)

    terms, warnings = build_glossary(None, segments, spy_llm, doubtful={1})
    assert texts(terms) == ["Kubeflow"]
    assert "NordVPN" not in sent[0]
    assert warnings == []
