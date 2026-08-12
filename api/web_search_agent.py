"""
WebSearchAgent — real-time web search for stock-specific queries.

Uses DuckDuckGo (free, no API key) to fetch recent news and web results
about a specific company. Results are formatted as a context string that
can be combined with DB context for the LLM generation stage.

Runs in a background thread so web search and DB retrieval are parallel.
"""
from __future__ import annotations

import re
from datetime import date

# ── Query enrichment keywords ─────────────────────────────────────────────────

_RESULT_KW   = {"result", "results", "quarterly", "quarter", "revenue", "profit",
                "pat", "q1", "q2", "q3", "q4", "earnings", "financial"}
_ORDER_KW    = {"order", "contract", "win", "bagged", "awarded", "l1"}
_GUIDANCE_KW = {"guidance", "outlook", "forecast", "target", "management"}
_NEWS_KW     = {"news", "latest", "recent", "update", "today", "this week",
                "announcement", "what happened"}


def _build_search_query(symbol: str, company: str, query: str) -> str:
    """Build a focused search query based on what the user is asking."""
    tl = set(re.split(r'\W+', query.lower()))
    company_short = company.split(" ")[0] if company else symbol  # "Tata" from "Tata Consultancy..."

    if tl & _RESULT_KW:
        return f"{company_short} {symbol} quarterly results FY2026 FY2027"
    if tl & _ORDER_KW:
        return f"{company_short} {symbol} order win contract 2026"
    if tl & _GUIDANCE_KW:
        return f"{company_short} {symbol} management guidance outlook 2026"
    # Default: recent news
    return f"{company_short} {symbol} BSE NSE news 2026"


def search_stock_news(
    symbol:      str,
    company:     str,
    query:       str,
    max_results: int = 6,
) -> str:
    """
    Search DuckDuckGo news for recent articles about a stock.
    Returns a formatted context string for the LLM.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Web search unavailable (duckduckgo-search not installed)."

    search_q = _build_search_query(symbol, company, query)

    try:
        raw = DDGS().news(search_q, max_results=max_results)
    except Exception as exc:
        return f"Web search error: {exc}"

    if not raw:
        # Try a simpler fallback query
        try:
            raw = DDGS().news(f"{symbol} stock news", max_results=3)
        except Exception:
            raw = []

    if not raw:
        return f"No recent web news found for {symbol}."

    lines = [
        f"LIVE WEB NEWS — {company} ({symbol})",
        f"(Fetched from the internet on {date.today().strftime('%d %b %Y')})",
        "─" * 60,
    ]
    for r in raw:
        title  = r.get("title", "").strip()
        source = r.get("source", "").strip()
        dt     = str(r.get("date", ""))[:10]
        body   = r.get("body", "").strip()
        url    = r.get("url", "").strip()

        if not title:
            continue
        lines.append(f"\n• {title}")
        if source or dt:
            lines.append(f"  [{source}  {dt}]")
        if body:
            lines.append(f"  {body[:250]}")
        if url:
            lines.append(f"  Source: {url}")

    return "\n".join(lines)


def search_stock_web(
    symbol:      str,
    company:     str,
    query:       str,
    max_results: int = 4,
) -> str:
    """
    Fallback: regular DuckDuckGo web search (broader than news-only).
    Used when news search returns no results.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ""

    search_q = f"{company} {symbol} BSE NSE 2026 site:economictimes.indiatimes.com OR site:moneycontrol.com OR site:businessstandard.com"
    try:
        raw = DDGS().text(search_q, max_results=max_results)
    except Exception:
        return ""

    if not raw:
        return ""

    lines = [f"WEB RESULTS — {company} ({symbol}):"]
    for r in raw:
        title = r.get("title", "").strip()
        body  = r.get("body", "").strip()
        href  = r.get("href", "").strip()
        if title:
            lines.append(f"\n• {title}")
            if body:
                lines.append(f"  {body[:200]}")
            if href:
                lines.append(f"  {href}")
    return "\n".join(lines)
