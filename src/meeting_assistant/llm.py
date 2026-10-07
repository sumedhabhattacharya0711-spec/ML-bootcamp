"""The one place that talks to the LLM providers.

Groq is the main provider and Gemini the backup. Both are reached through their
OpenAI-compatible APIs, so one client library covers them. If Groq is rate
limited, out of quota, down or unreachable, the same call goes to Gemini (when
LLM_GEMINI_KEY is set). Setup errors (bad key, bad request) are not retried
elsewhere, so they stay visible.

call_llm(system, user)                    -> reply text       (glossary.py, refine.py)
call_llm_structured(system, user, Model)  -> pydantic object  (minutes.py)

Every failure becomes an LLMError with a plain-English message that the UI
can show next to the stage that failed.

Try it:  python -m meeting_assistant.llm            (sends a tiny test message)
         python -m meeting_assistant.llm --models   (lists the models your key can use)
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from meeting_assistant.paths import PROJECT_DIR

load_dotenv(PROJECT_DIR / ".env")

# ---------- Settings (move to config.yaml later) ----------

GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 8192  # doubled once if an answer is cut off
TIMEOUT_S = 60
MAX_RETRIES = 3  # retries rate limits, timeouts and server errors, waiting as long as the
                 # provider asks (free tiers like Groq allow only a few thousand tokens per minute)

# Sent when the model accepts them. Reasoning models (gpt-oss) reject temperature;
# low reasoning effort keeps their hidden "thinking" from using up the output budget.
OPTIONAL_PARAMS = {"temperature": TEMPERATURE, "reasoning_effort": "low"}

# Optional params each model rejected, filled in at runtime so they aren't sent again.
_unsupported: dict[str, set[str]] = {}


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str   # the .env variable holding this provider's API key
    base_url: str
    model: str


PRIMARY = Provider("groq", "LLM_API_KEY", os.getenv("LLM_BASE_URL") or GROQ_URL,
                   os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
BACKUP = Provider("gemini", "LLM_GEMINI_KEY", GEMINI_URL,
                  os.getenv("LLM_GEMINI_MODEL", "gemini-3.8-flash"))

# Temporary provider problems: worth trying the backup.
FALLBACK_ERRORS = (openai.RateLimitError, openai.APITimeoutError,
                   openai.APIConnectionError, openai.InternalServerError)


class LLMError(Exception):
    """An LLM failure. The message is shown to the user as is."""


@dataclass
class LLMUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


# One entry per call, so pipeline.py can save tokens and timings into runs/.
usage_log: list[LLMUsage] = []


@lru_cache(maxsize=None)
def get_client(provider: Provider) -> openai.OpenAI:
    """Create each provider's client once and reuse it."""
    key = os.getenv(provider.key_env, "").strip()
    if not key:
        raise LLMError(f"{provider.key_env} is not set. Add it to the .env file in the project folder.")
    return openai.OpenAI(api_key=key, base_url=provider.base_url, timeout=TIMEOUT_S,
                         max_retries=MAX_RETRIES)


def _friendly(e: Exception, provider: Provider, model: str) -> str:
    """Turn an openai exception into a message a user can act on."""
    return f"{provider.name}: {_reason(e, provider, model)}"


def _reason(e: Exception, provider: Provider, model: str) -> str:
    if isinstance(e, openai.AuthenticationError):
        return f"the API key was rejected (check {provider.key_env} in .env)."
    if isinstance(e, openai.RateLimitError):
        if "insufficient_quota" in str(e):
            return "quota exceeded (add credit / billing on the provider account)."
        return "rate limited, try again in a minute."
    if isinstance(e, openai.APITimeoutError):
        return f"no response after {TIMEOUT_S} s."
    if isinstance(e, openai.APIConnectionError):
        return "could not reach the API (check the internet connection)."
    if isinstance(e, openai.NotFoundError):
        return f"model not found or not available to this key: {model}"
    if isinstance(e, openai.APIStatusError):
        return f"error {e.status_code}: {e.message}"
    return str(e)


def _send(method: str, system: str, user: str, model: str | None, **kwargs):
    """Send one request with chat.completions.<method> ("create" or "parse").
    Tries Groq, then Gemini for temporary problems. `model` overrides Groq's model."""
    providers = [PRIMARY] + ([BACKUP] if os.getenv(BACKUP.key_env, "").strip() else [])
    failures = []
    for provider in providers:
        provider_model = (model if provider is PRIMARY and model else provider.model)
        try:
            return _send_to(provider, provider_model, method, system, user, **kwargs)
        except openai.OpenAIError as e:
            failures.append(_friendly(e, provider, provider_model))
            if not isinstance(e, FALLBACK_ERRORS):
                break
    raise LLMError("LLM call failed: " + " | ".join(failures))


def _send_to(provider: Provider, model: str, method: str, system: str, user: str, **kwargs):
    """One request to one provider. Records usage. If the answer is cut off at
    the output limit, asks again once with double the limit."""
    client = get_client(provider)
    params = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    params.update({k: v for k, v in OPTIONAL_PARAMS.items() if k not in _unsupported.get(model, set())})

    started = time.perf_counter()
    for limit in (MAX_OUTPUT_TOKENS, 2 * MAX_OUTPUT_TOKENS):
        params["max_tokens"] = limit
        try:
            response = _create(client, method, params, model)
        except openai.LengthFinishReasonError:  # parse() raises this for a cut-off answer
            continue
        if response.choices[0].finish_reason == "length":
            continue
        usage = response.usage
        usage_log.append(LLMUsage(
            provider=provider.name,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            seconds=round(time.perf_counter() - started, 2),
        ))
        return response
    raise LLMError(f"LLM call failed: {provider.name}: the answer was cut off at {limit} tokens, "
                   "even after retrying with a larger limit.")


def _create(client: openai.OpenAI, method: str, params: dict, model: str):
    """Call chat.completions.<method>; drop any optional param the model rejects."""
    while True:
        try:
            return getattr(client.chat.completions, method)(**params)
        except openai.BadRequestError as e:
            rejected = next((p for p in OPTIONAL_PARAMS if p in params and p in str(e)), None)
            if rejected is None:
                raise
            _unsupported.setdefault(model, set()).add(rejected)
            params.pop(rejected)


def parse_json_reply(reply: str) -> dict:
    """Parse a plain-text reply that should contain one JSON object,
    tolerating ```json fences or text around it."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise LLMError("LLM reply contained no JSON object")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM reply was not valid JSON: {e}") from e


def call_llm(system: str, user: str, model: str | None = None) -> str:
    """Plain-text call: instructions in `system`, data in `user`. Returns the reply."""
    response = _send("create", system, user, model)
    return response.choices[0].message.content or ""


def _missing_field(e: ValidationError) -> str:
    """'action_items.0.owner: Field required' from a pydantic error."""
    err = e.errors()[0]
    where = ".".join(str(p) for p in err["loc"]) or "answer"
    return f"{where}: {err['msg']}"


def call_llm_structured(system: str, user: str, schema: type[BaseModel],
                        model: str | None = None) -> BaseModel:
    """JSON call: OpenAI structured outputs force the reply into `schema`'s
    shape and we get a validated pydantic object back. One retry if the model
    refuses or the reply doesn't validate."""
    last_problem = ""
    for _attempt in range(2):
        try:
            response = _send("parse", system, user, model, response_format=schema)
        except ValidationError as e:  # parse() validates the reply itself and raises this
            last_problem = _missing_field(e)
            continue
        message = response.choices[0].message
        if message.refusal:
            last_problem = f"the model refused: {message.refusal}"
            continue
        if message.parsed is not None:
            return message.parsed
        # Fallback for providers without schema support: validate the raw text ourselves.
        try:
            return schema.model_validate_json(message.content or "")
        except ValidationError as e:
            last_problem = _missing_field(e)
    raise LLMError(f"LLM answer failed validation: {last_problem}")


if __name__ == "__main__":
    try:
        if "--models" in sys.argv:
            ids = sorted(m.id for m in get_client(PRIMARY).models.list())
            print("\n".join(i for i in ids if i.startswith(("gpt", "o"))))
        else:
            reply = call_llm("You are a test endpoint.", "Reply with exactly: OK")
            u = usage_log[-1]
            print(f"provider={u.provider}  model={u.model}  reply={reply!r}  tokens in/out={u.input_tokens}/{u.output_tokens}  "
                  f"time={u.seconds}s")
    except LLMError as e:
        print(e)
        sys.exit(1)
