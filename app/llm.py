"""Provider-agnostic LLM wrapper.

Supports Anthropic and OpenAI via env, plus a deterministic `mock` provider used
for offline self-tests (so CI / a reviewer can run the eval harness wiring with
no API key). The mock is NOT for the real demo — record your Loom with a real key.
"""
from __future__ import annotations
import re
import time
from typing import Optional
from .config import settings


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str, temperature: Optional[float] = None) -> str:
    provider = settings.llm_provider.lower()
    temp = settings.temperature if temperature is None else temperature

    if provider == "mock":
        return _mock_complete(system, user)
    if provider == "anthropic":
        return _anthropic_complete(system, user, temp)
    if provider == "openai":
        return _openai_complete(system, user, temp)
    if provider == "google":
        return _google_complete(system, user, temp)
    raise LLMError(f"Unknown LLM_PROVIDER={provider!r}")


def _anthropic_complete(system: str, user: str, temp: float) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("pip install anthropic") from e
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=settings.model,
        max_tokens=1024,
        temperature=temp,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def _openai_complete(system: str, user: str, temp: float) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("pip install openai") from e
    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=settings.model,
        temperature=temp,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _google_complete(system: str, user: str, temp: float) -> str:
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError
    except ImportError as e:
        raise LLMError("pip install google-genai") from e
    client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY
    config = types.GenerateContentConfig(system_instruction=system, temperature=temp)

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=settings.model, contents=user, config=config
            )
            return (resp.text or "").strip()
        except ClientError as e:
            if e.code != 429:
                raise
            if "PerDay" in str(e):
                # Daily quota won't recover within a request's lifetime — retrying
                # just makes the caller hang. Fail immediately with a clear message.
                raise LLMError(
                    f"Gemini free-tier daily quota exhausted for {settings.model}. "
                    "Try again after it resets, switch MODEL, or enable billing."
                ) from e
            if attempt == max_retries:
                raise LLMError(
                    f"Gemini rate limit hit repeatedly for {settings.model}; giving up "
                    f"after {max_retries} retries."
                ) from e
            time.sleep(_retry_delay_seconds(e))


def _retry_delay_seconds(e: "ClientError") -> float:  # noqa: F821 - type only
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(e))
    return min(float(match.group(1)) + 1 if match else 5.0, 10.0)


# --- Mock provider: deterministic, rule-based. Only for offline self-test. ---
def _mock_complete(system: str, user: str) -> str:
    """Fakes the two calls the app makes: the grounded answer and the grader.

    Grounded answer: echoes the provided CONTEXT if the question keywords appear
    in it, else emits the refusal token. This lets the harness demonstrate the
    *behavioural* difference (refuse vs. answer) without a real model.
    """
    text = user.lower()
    if "you are grading" in system.lower():
        # grader call -> yes/no whether context answers the question
        return "yes" if "context:" in text and _keyword_overlap(text) else "no"
    # answer call
    if _keyword_overlap(text):
        return "Based on the lesson: " + _first_context_sentence(user) + " [00:00]"
    return "NOT_IN_LESSON"


def _keyword_overlap(user_text: str) -> bool:
    q = _extract(user_text, "question:")
    ctx = _extract(user_text, "context:")
    if not q or not ctx:
        return False
    qs = {w for w in q.split() if len(w) > 4}
    return any(w in ctx for w in qs)


def _first_context_sentence(user_text: str) -> str:
    ctx = _extract(user_text, "context:")
    return ctx.split(".")[0][:160] if ctx else ""


def _extract(text: str, marker: str) -> str:
    i = text.find(marker)
    if i == -1:
        return ""
    return text[i + len(marker):]
