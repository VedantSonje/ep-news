"""
EPAgent — dual-agent orchestrator for equity research chat.

Architecture (two parallel agents per stock query):
─────────────────────────────────────────────────────────────
  User Query
      │
      ├─► DB Agent    — ChatHandler.retrieve() → SQLite + ChromaDB
      │                 (financial results, order wins, announcements)
      │
      └─► Web Agent   — DuckDuckGo news search
                        (real-time news, latest price moves, analyst views)
      │
      └─► [Context merge] → PydanticAI streaming LLM → answer
─────────────────────────────────────────────────────────────

When no specific company is detected the web search is skipped and only
the DB agent runs (identical to the existing stream_response() path).

Tool-calling notes:
  llama3:latest does NOT support function calling. When the user pulls
  llama3.1 (4.7 GB, fits in 6 GB VRAM), set OLLAMA_TOOL_MODEL=llama3.1
  in .env to activate the full tool-calling agent (the commented section
  at the bottom of this file).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Generator

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

if TYPE_CHECKING:
    from api.chat_handler import ChatHandler

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_TEXT_MODEL      = os.getenv("OLLAMA_TEXT_MODEL", "llama3:latest")
_WEB_TIMEOUT     = int(os.getenv("WEB_SEARCH_TIMEOUT", "8"))   # seconds


@dataclass
class EPDeps:
    handler: "ChatHandler"
    query:   str        = ""
    history: list[dict] = field(default_factory=list)


# ── PydanticAI text agent ─────────────────────────────────────────────────────

_provider   = OllamaProvider(base_url=_OLLAMA_BASE_URL)
_text_model = OllamaModel(_TEXT_MODEL, provider=_provider)

_SYSTEM_PROMPT = """You are an equity research analyst at an Indian stockbroker.
Answer questions about BSE/NSE stocks using the data provided in the context below.

The context may contain two sections:
  1. DATABASE CONTEXT  — verified data from our BSE/NSE announcements database
  2. LIVE WEB NEWS     — real-time news fetched from the internet today

Rules:
- Use BOTH sources when available — cite DB data for verified financials, web news for latest developments
- Clearly label where each fact comes from: "[DB]" for database, "[Web]" for web news
- Be direct and specific — cite exact company names, rupee values, and dates
- For numerical data (revenue, PAT, order values): prefer the DATABASE source over web summaries
- For recent events (last few days): web news may be more current than our DB
- NEVER invent numbers — only use what the context provides
- Use Indian number format (Rs. Cr)
- Today's date is {today}"""

ep_agent: Agent[EPDeps, str] = Agent(
    _text_model,
    deps_type=EPDeps,
    output_type=str,
    system_prompt=_SYSTEM_PROMPT.format(today=date.today().strftime("%d %B %Y")),
)


# ── Dual-agent streaming (DB + Web in parallel) ───────────────────────────────

def stream_dual_agent(
    message:     str,
    results:     list[dict],
    intent:      str,
    history:     list[dict],
    handler:     "ChatHandler",
    min_cr:      float | None = None,
    company_sym: str   | None = None,
) -> Generator[str, None, None]:
    """
    Orchestrate both agents in parallel:
      1. DB context    — built from pre-retrieved SQL/vector results (instant)
      2. Web context   — fetched from DuckDuckGo in a background thread (~1-3s)

    Yields SSE-ready text tokens to the caller.
    """
    from api.chat_handler import _build_context, _format_direct_list, _is_listing_query

    # Pure listing queries bypass the LLM entirely — same as stream_response()
    if _is_listing_query(message) and intent in ("order_win", "financials") and results:
        yield _format_direct_list(results, intent, min_cr=min_cr)
        return

    # ── DB context (instant — data already in `results`) ─────────────────────
    db_context = _build_context(results, intent, min_cr=min_cr)

    # ── Web context (parallel thread) ─────────────────────────────────────────
    web_context = ""
    if company_sym:
        company_name = _resolve_company_name(company_sym, results, handler)
        web_context  = _fetch_web_context(company_sym, company_name, message)

    # ── Combine contexts ──────────────────────────────────────────────────────
    if web_context:
        combined = (
            "DATABASE CONTEXT (verified BSE/NSE data):\n"
            + db_context
            + "\n\n"
            + "─" * 60
            + "\n\n"
            + web_context
        )
    else:
        combined = db_context

    # ── Build user prompt with history ────────────────────────────────────────
    hist_text = ""
    for turn in history[-4:]:
        role    = turn.get("role", "")
        content = str(turn.get("content", ""))[:800]
        if role in ("user", "assistant") and content:
            hist_text += f"\n{role.upper()}: {content}"

    user_prompt = (
        f"Today is {date.today().strftime('%d %B %Y')}.\n\n"
        + combined
        + (f"\n\nCONVERSATION HISTORY:{hist_text}\n" if hist_text else "")
        + f"\n\nQUESTION: {message}"
    )

    deps = EPDeps(handler=handler, query=message, history=history)

    # ── PydanticAI streaming ──────────────────────────────────────────────────
    try:
        with ep_agent.run_stream_sync(user_prompt, deps=deps) as response:
            for delta in response.stream_text(delta=True, debounce_by=None):
                if delta:
                    yield delta
    except Exception as exc:
        yield f"\n\n⚠️ Agent error: {exc}\n"
        yield "\nFalling back to direct context:\n\n"
        yield combined


def stream_with_agent(
    message: str,
    results: list[dict],
    intent:  str,
    history: list[dict],
    handler: "ChatHandler",
    min_cr:  float | None = None,
) -> Generator[str, None, None]:
    """
    Text-only PydanticAI agent (no web search). Used when USE_AGENT=1 but
    WEB_SEARCH=0, or when no company symbol is detected.
    """
    yield from stream_dual_agent(
        message=message, results=results, intent=intent,
        history=history, handler=handler, min_cr=min_cr,
        company_sym=None,   # no web search
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_company_name(symbol: str, results: list[dict], handler: "ChatHandler") -> str:
    """Get the full company name for a symbol from retrieved results or DB."""
    for r in results:
        if r.get("symbol", "").upper() == symbol.upper() and r.get("company"):
            return r["company"]
    # Fallback: query DB directly
    try:
        import sqlite3
        conn = sqlite3.connect(str(handler._db_path))
        row  = conn.execute(
            "SELECT company FROM announcements WHERE symbol=? LIMIT 1", (symbol,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return symbol


def _fetch_web_context(symbol: str, company: str, query: str) -> str:
    """
    Run web search in a thread-pool with timeout.
    Returns empty string on timeout or error — never blocks the LLM stream.
    """
    from api.web_search_agent import search_stock_news

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(search_stock_news, symbol, company, query)
        try:
            return future.result(timeout=_WEB_TIMEOUT)
        except FutureTimeoutError:
            return ""
        except Exception:
            return ""


# ── Tool-capable agent (enable when OLLAMA_TOOL_MODEL=llama3.1) ───────────────
#
# from pydantic_ai import RunContext
# _TOOL_MODEL = os.getenv("OLLAMA_TOOL_MODEL", "")
# if _TOOL_MODEL:
#     _tool_model = OllamaModel(_TOOL_MODEL, provider=_provider)
#     tool_agent: Agent[EPDeps, str] = Agent(
#         _tool_model, deps_type=EPDeps, output_type=str,
#         system_prompt=_SYSTEM_PROMPT.format(today=date.today().strftime("%d %B %Y"))
#     )
#
#     @tool_agent.tool
#     def lookup_financials(ctx: RunContext[EPDeps], symbol: str = "", sector: str = "",
#                           from_date: str = "", to_date: str = "") -> str:
#         """Look up financial results by company, sector, or date range."""
#         from api.chat_handler import _build_context
#         h = ctx.deps.handler
#         from_dt = _safe_date(from_date); to_dt = _safe_date(to_date) or from_dt
#         if symbol:
#             results = h._sql_financials_by_symbol(symbol.upper())
#         elif from_dt:
#             results = h._sql_financials_by_date(from_dt, to_dt, sector=sector or None)
#         elif sector:
#             results = h._sql_financials_by_sector(sector)
#         else:
#             results = h._store.search_financials(ctx.deps.query)
#         return _build_context(results, "financials") if results else "No results."
#
#     @tool_agent.tool
#     def lookup_order_wins(ctx: RunContext[EPDeps], symbol: str = "", sector: str = "",
#                           from_date: str = "", to_date: str = "", min_cr: float = 0) -> str:
#         """Look up order win announcements."""
#         from api.chat_handler import _build_context
#         h = ctx.deps.handler
#         from_dt = _safe_date(from_date); to_dt = _safe_date(to_date) or from_dt
#         if symbol:
#             results = h._sql_order_wins_by_symbol(symbol.upper())
#         elif from_dt:
#             results = h._sql_order_wins_by_date(from_dt, to_dt, min_cr=min_cr or None, sector=sector or None)
#         else:
#             results = h._store.search_order_wins(ctx.deps.query, min_cr=min_cr or None)
#         return _build_context(results, "order_win", min_cr=min_cr or None) if results else "No results."
#
#     @tool_agent.tool
#     def search_web_news(ctx: RunContext[EPDeps], symbol: str, company: str) -> str:
#         """Fetch latest web news for a stock from the internet."""
#         from api.web_search_agent import search_stock_news
#         return search_stock_news(symbol, company, ctx.deps.query)
#
# def _safe_date(s: str):
#     from datetime import date
#     try: return date.fromisoformat(s) if s else None
#     except ValueError: return None
