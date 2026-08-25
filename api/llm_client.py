"""
Unified LLM client — routes to Groq (cloud) or Ollama (local).

Priority: GROQ_API_KEY set → Groq cloud (Gemini as rate-limit fallback);
          otherwise → Ollama local.
"""
from __future__ import annotations

import os
from typing import Generator

_GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
_GROQ_MODEL     = os.getenv("GROQ_MODEL",      "groq/compound-mini")
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY",  "")
_GEMINI_MODEL   = os.getenv("GEMINI_MODEL",    "gemini-3.6-flash")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",    "llama3.1:latest")

ACTIVE_PROVIDER = "groq" if _GROQ_API_KEY else "ollama"
ACTIVE_MODEL    = _GROQ_MODEL if _GROQ_API_KEY else _OLLAMA_MODEL


def llm_complete(
    messages:    list[dict],
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
) -> str:
    """Non-streaming LLM call — returns full response text."""
    if _GROQ_API_KEY:
        return _groq_complete(messages, temperature, max_tokens)
    return _ollama_complete(messages, temperature)


def llm_json(
    messages:    list[dict],
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
) -> str:
    """
    Non-streaming LLM call with JSON output enforced.
    Returns the raw JSON string (caller must parse).
    """
    if _GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    else:
        import ollama
        resp = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[{"role": "user", "content": messages[-1]["content"]}],
            format="json",
            options={"temperature": temperature, "num_gpu": 99},
        )
        try:
            return resp.message.content or ""
        except AttributeError:
            return resp["message"]["content"] or ""


def llm_stream(
    messages:    list[dict],
    temperature: float = 0.0,
    max_tokens:  int   = 4096,
) -> Generator[str, None, None]:
    """Streaming LLM call — yields tokens as they arrive."""
    if _GROQ_API_KEY:
        yield from _groq_stream(messages, temperature, max_tokens)
    else:
        yield from _ollama_stream(messages, temperature)


# ── Groq ──────────────────────────────────────────────────────────────────────

def _rate_limit_wait(err: str) -> float:
    """Parse Groq rate-limit error and return seconds to sleep."""
    import re as _re
    # "try again in 1.5675s"  (seconds)
    m = _re.search(r'try again in ([\d.]+)s\b', err, _re.I)
    if m:
        return float(m.group(1)) + 0.5
    # "try again in 772.5ms"  (milliseconds)
    m = _re.search(r'try again in ([\d.]+)ms', err, _re.I)
    if m:
        return float(m.group(1)) / 1000 + 0.3
    return 3.0   # safe fallback


def _groq_complete(messages, temperature, max_tokens) -> str:
    import time
    from groq import Groq
    client = Groq(api_key=_GROQ_API_KEY)
    last_err = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_GROQ_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = str(e)
            if attempt < 2 and ("rate_limit" in last_err.lower() or "429" in last_err or "ratelimit" in last_err.lower()):
                time.sleep(_rate_limit_wait(last_err))
            else:
                break
    # Groq exhausted — fall back to Gemini if available
    if _GEMINI_API_KEY and ("rate_limit" in last_err.lower() or "429" in last_err):
        result = _gemini_complete(messages, temperature, max_tokens)
        if result:
            return result
    raise RuntimeError(last_err)


def _groq_stream(messages, temperature, max_tokens) -> Generator[str, None, None]:
    import time
    from groq import Groq
    client = Groq(api_key=_GROQ_API_KEY)
    stream = None
    last_err = ""
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(
                model=_GROQ_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                stream=True,
            )
            break
        except Exception as e:
            last_err = str(e)
            if attempt < 2 and ("rate_limit" in last_err.lower() or "429" in last_err or "ratelimit" in last_err.lower()):
                time.sleep(_rate_limit_wait(last_err))
            else:
                break
    if stream is not None:
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
        return
    # Groq exhausted — fall back to Gemini if available
    if _GEMINI_API_KEY and ("rate_limit" in last_err.lower() or "429" in last_err):
        yield from _gemini_stream(messages, temperature, max_tokens)
        return
    raise RuntimeError(last_err)


# ── Gemini fallback (used when Groq is rate-limited) ─────────────────────────

def _gemini_complete(messages: list[dict], temperature: float, max_tokens: int) -> str:
    if not _GEMINI_API_KEY:
        return ""
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
        client = _genai.Client(api_key=_GEMINI_API_KEY)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages if m.get("content") and m.get("role") in ("user", "assistant", "system")
        ]
        cfg = _gtypes.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)
        resp = client.models.generate_content(model=_GEMINI_MODEL, contents=contents, config=cfg)
        return resp.text or ""
    except Exception:
        return ""


def _gemini_stream(messages: list[dict], temperature: float, max_tokens: int) -> Generator[str, None, None]:
    if not _GEMINI_API_KEY:
        return
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
        client = _genai.Client(api_key=_GEMINI_API_KEY)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages if m.get("content") and m.get("role") in ("user", "assistant", "system")
        ]
        cfg = _gtypes.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)
        for chunk in client.models.generate_content_stream(model=_GEMINI_MODEL, contents=contents, config=cfg):
            text = chunk.text or ""
            if text:
                yield text
    except Exception:
        return


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama_complete(messages, temperature) -> str:
    import ollama
    resp = ollama.chat(
        model=_OLLAMA_MODEL,
        messages=messages,
        stream=False,
        options={"temperature": temperature, "num_predict": 1024},
    )
    return resp.message.content or ""


def _ollama_stream(messages, temperature) -> Generator[str, None, None]:
    import ollama
    stream = ollama.chat(
        model=_OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": temperature, "num_predict": -1},
    )
    for chunk in stream:
        try:
            token = chunk.message.content
        except AttributeError:
            token = chunk["message"]["content"]
        if token:
            yield token
