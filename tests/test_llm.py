from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from meeting_assistant import llm


class Answer(BaseModel):
    owner: str


def fake_response(content="", parsed=None, refusal=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, parsed=parsed, refusal=refusal)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=usage)


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def _next(self, **params):
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    create = parse = _next


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.fixture
def fake(monkeypatch):
    """Groq answers with `replies`; no Gemini backup configured."""
    def install(*replies):
        completions = FakeCompletions(replies)
        monkeypatch.delenv("LLM_GEMINI_KEY", raising=False)
        monkeypatch.setattr(llm, "get_client", lambda provider: fake_client(completions))
        return completions
    return install


@pytest.fixture
def fake_both(monkeypatch):
    """Groq answers with `groq_replies`, Gemini with `gemini_replies`."""
    def install(groq_replies, gemini_replies):
        groq, gemini = FakeCompletions(groq_replies), FakeCompletions(gemini_replies)
        monkeypatch.setenv("LLM_GEMINI_KEY", "test-key")
        monkeypatch.setattr(llm, "get_client", lambda provider: fake_client(
            groq if provider is llm.PRIMARY else gemini))
        return groq, gemini
    return install


def api_error(cls, message, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com"))
    return cls(message, response=response, body=None)


def test_missing_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    llm.get_client.cache_clear()
    with pytest.raises(llm.LLMError, match="LLM_API_KEY is not set"):
        llm.get_client(llm.PRIMARY)
    llm.get_client.cache_clear()


def test_call_llm_returns_text_and_logs_usage(fake):
    calls = fake(fake_response("OK"))
    assert llm.call_llm("sys", "user") == "OK"
    assert calls.calls[0]["temperature"] == 0
    assert llm.usage_log[-1].input_tokens == 10


def test_structured_returns_pydantic_object(fake):
    fake(fake_response(parsed=Answer(owner="Sarah")))
    assert llm.call_llm_structured("sys", "user", Answer).owner == "Sarah"


def test_structured_falls_back_to_raw_json(fake):
    fake(fake_response(content='{"owner": "Sarah"}', parsed=None))
    assert llm.call_llm_structured("sys", "user", Answer).owner == "Sarah"


def test_structured_bad_json_names_the_field(fake):
    fake(fake_response(content='{"name": "x"}'), fake_response(content='{"name": "x"}'))
    with pytest.raises(llm.LLMError, match="failed validation: owner: Field required"):
        llm.call_llm_structured("sys", "user", Answer)


def test_structured_retries_once_after_refusal(fake):
    calls = fake(fake_response(refusal="no"), fake_response(parsed=Answer(owner="Sarah")))
    assert llm.call_llm_structured("sys", "user", Answer).owner == "Sarah"
    assert len(calls.calls) == 2


def test_auth_error_is_friendly(fake):
    fake(api_error(openai.AuthenticationError, "bad key", 401))
    with pytest.raises(llm.LLMError, match="API key was rejected"):
        llm.call_llm("sys", "user")


def test_quota_error_is_friendly(fake):
    fake(api_error(openai.RateLimitError, "insufficient_quota", 429))
    with pytest.raises(llm.LLMError, match="quota exceeded"):
        llm.call_llm("sys", "user")


def test_model_without_temperature_is_retried(fake):
    calls = fake(api_error(openai.BadRequestError, "Unsupported parameter: 'temperature'", 400),
                 fake_response("OK"))
    assert llm.call_llm("sys", "user", model="reasoning-test-model") == "OK"
    assert "temperature" not in calls.calls[1]


def validation_error():
    try:
        Answer.model_validate({})
    except ValidationError as e:
        return e


def test_structured_retries_when_parse_raises_validation_error(fake):
    calls = fake(validation_error(), fake_response(parsed=Answer(owner="Sarah")))
    assert llm.call_llm_structured("sys", "user", Answer).owner == "Sarah"
    assert len(calls.calls) == 2


def test_structured_validation_error_twice_is_friendly(fake):
    fake(validation_error(), validation_error())
    with pytest.raises(llm.LLMError, match="failed validation: owner: Field required"):
        llm.call_llm_structured("sys", "user", Answer)


def test_rate_limited_groq_falls_back_to_gemini(fake_both):
    groq, gemini = fake_both([api_error(openai.RateLimitError, "rate limited", 429)],
                             [fake_response("from gemini")])
    assert llm.call_llm("sys", "user") == "from gemini"
    assert gemini.calls[0]["model"] == llm.BACKUP.model
    assert llm.usage_log[-1].provider == "gemini"


def test_bad_groq_key_does_not_fall_back(fake_both):
    groq, gemini = fake_both([api_error(openai.AuthenticationError, "bad key", 401)],
                             [fake_response("from gemini")])
    with pytest.raises(llm.LLMError, match="groq: the API key was rejected"):
        llm.call_llm("sys", "user")
    assert gemini.calls == []


def test_both_providers_failing_gives_one_error(fake_both):
    fake_both([api_error(openai.RateLimitError, "rate limited", 429)],
              [api_error(openai.InternalServerError, "down", 503)])
    with pytest.raises(llm.LLMError, match="groq: rate limited.*gemini: error 503"):
        llm.call_llm("sys", "user")


def test_cut_off_answer_is_retried_with_a_larger_limit(fake):
    calls = fake(fake_response('{"lines": {"3": "half a sen', finish_reason="length"),
                 fake_response('{"lines": {}}'))
    assert llm.call_llm("sys", "user") == '{"lines": {}}'
    assert [c["max_tokens"] for c in calls.calls] == [llm.MAX_OUTPUT_TOKENS, 2 * llm.MAX_OUTPUT_TOKENS]


def test_cut_off_structured_answer_is_retried(fake):
    cut_off = openai.LengthFinishReasonError(completion=fake_response(finish_reason="length"))
    calls = fake(cut_off, fake_response(parsed=Answer(owner="Sarah")))
    assert llm.call_llm_structured("sys", "user", Answer).owner == "Sarah"
    assert len(calls.calls) == 2


def test_cut_off_twice_gives_a_clear_error(fake):
    fake(fake_response("{", finish_reason="length"), fake_response("{", finish_reason="length"))
    with pytest.raises(llm.LLMError, match="cut off .* even after retrying"):
        llm.call_llm("sys", "user")


def test_rejected_reasoning_effort_is_dropped(fake):
    calls = fake(api_error(openai.BadRequestError, "reasoning_effort is not supported", 400),
                 fake_response("OK"))
    assert llm.call_llm("sys", "user", model="no-reasoning-test-model") == "OK"
    assert calls.calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in calls.calls[1]
