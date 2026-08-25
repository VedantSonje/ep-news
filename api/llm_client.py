"""
Unified LLM client — routes to Groq (cloud) or Ollama (local).

Priority: GROQ_API_KEY set → Groq cloud; otherwise → Ollama local.
Drop-in replacement so chat_handler and local_extractor need no
provider-specific logic.
"""
from __future__ import annotations

import os
from typing import Generator

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL   = os.getenv("GROQ_MODEL",   "groq/compound-mini")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",  "llama3.1:latest")

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

def _groq_complete(messages, temperature, max_tokens) -> str:
    import re, time
    from groq import Groq
    client = Groq(api_key=_GROQ_API_KEY)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            if attempt < 2 and ("rate_limit" in err.lower() or "429" in err or "ratelimit" in err.lower()):
                m = re.search(r'try again in ([\d.]+)ms', err, re.I)
                time.sleep((float(m.group(1)) / 1000 + 0.3) if m else 2.0)
            else:
                raise
    return ""


def _groq_stream(messages, temperature, max_tokens) -> Generator[str, None, None]:
    import re, time
    from groq import Groq
    client = Groq(api_key=_GROQ_API_KEY)
    stream = None
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            break
        except Exception as e:
            err = str(e)
            if attempt < 2 and ("rate_limit" in err.lower() or "429" in err or "ratelimit" in err.lower()):
                m = re.search(r'try again in ([\d.]+)ms', err, re.I)
                time.sleep((float(m.group(1)) / 1000 + 0.3) if m else 2.0)
            else:
                raise
    if stream is None:
        return
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


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
