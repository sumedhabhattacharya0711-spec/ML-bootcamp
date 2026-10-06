"""The one place that talks to the LLM provider (OpenAI).

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

MODEL = os.getenv("LLM_MODEL", "gpt-4.1")  # non-reasoning model: accepts temperature 0
TEMPERATURE = 0
TIMEOUT_S = 60
MAX_RETRIES = 1  # the openai library retries rate limits, timeouts and server errors once

# Models that rejected the temperature setting (reasoning models). Filled in at runtime.
_no_temperature: set[str] = set()


class LLMError(Exception):
    """An LLM failure. The message is shown to the user as is."""


@dataclass
class LLMUsage:
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


# One entry per call, so pipeline.py can save tokens and timings into runs/.
usage_log: list[LLMUsage] = []


@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    """Create the OpenAI client once and reuse it."""
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise LLMError("LLM_API_KEY is not set. Add it to the .env file in the project folder.")
    return openai.OpenAI(api_key=key, timeout=TIMEOUT_S, max_retries=MAX_RETRIES)


def _friendly(e: Exception) -> LLMError:
    """Turn an openai exception into a message a user can act on."""
    if isinstance(e, openai.AuthenticationError):
        return LLMError("LLM call failed: the API key was rejected (check LLM_API_KEY in .env).")
    if isinstance(e, openai.RateLimitError):
        if "insufficient_quota" in str(e):
            return LLMError("LLM call failed: quota exceeded (add credit / billing on the OpenAI account).")
        return LLMError("LLM call failed: rate limited by OpenAI, try again in a minute.")
    if isinstance(e, openai.APITimeoutError):
        return LLMError(f"LLM call failed: no response after {TIMEOUT_S} s.")
    if isinstance(e, openai.APIConnectionError):
        return LLMError("LLM call failed: could not reach the OpenAI API (check the internet connection).")
    if isinstance(e, openai.NotFoundError):
        return LLMError(f"LLM call failed: model not found or not available to this key: {MODEL}")
    if isinstance(e, openai.LengthFinishReasonError):
        return LLMError("LLM call failed: the answer was cut off because it was too long.")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"LLM call failed ({e.status_code}): {e.message}")
    return LLMError(f"LLM call failed: {e}")


def _send(method: str, system: str, user: str, model: str, **kwargs):
    """Send one request with chat.completions.<method> ("create" or "parse"),
    record usage, and convert errors. Retries once without temperature if the
    model rejects it."""
    client = get_client()
    params = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    if model not in _no_temperature:
        params["temperature"] = TEMPERATURE

    started = time.perf_counter()
    try:
        try:
            response = getattr(client.chat.completions, method)(**params)
        except openai.BadRequestError as e:
            if "temperature" not in str(e) or "temperature" not in params:
                raise
            _no_temperature.add(model)  # reasoning model: remember and resend without it
            params.pop("temperature")
            response = getattr(client.chat.completions, method)(**params)
    except openai.OpenAIError as e:
        raise _friendly(e) from e

    usage = response.usage
    usage_log.append(LLMUsage(
        model=model,
        input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        seconds=round(time.perf_counter() - started, 2),
    ))
    return response


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
    response = _send("create", system, user, model or MODEL)
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
            response = _send("parse", system, user, model or MODEL, response_format=schema)
        except LLMError as e:
            if isinstance(e.__cause__, ValidationError):
                last_problem = _missing_field(e.__cause__)
                continue
            raise
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
            ids = sorted(m.id for m in get_client().models.list())
            print("\n".join(i for i in ids if i.startswith(("gpt", "o"))))
        else:
            reply = call_llm("You are a test endpoint.", "Reply with exactly: OK")
            u = usage_log[-1]
            print(f"model={u.model}  reply={reply!r}  tokens in/out={u.input_tokens}/{u.output_tokens}  "
                  f"time={u.seconds}s")
    except LLMError as e:
        print(e)
        sys.exit(1)
