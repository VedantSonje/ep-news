"""
Tool Agent — agentic chat where the LLM decides which DB tools to call.

The agent loop:
  1. LLM sees user query + tool definitions
  2. LLM picks tool(s) and arguments
  3. Tools execute against the SQLite DB
  4. Results fed back; loop repeats up to MAX_ROUNDS
  5. LLM streams the final answer once it has enough data

Yields SSE-ready dicts:
  {"type": "tool_call",   "tool": name, "args": args}
  {"type": "tool_result", "tool": name, "count": n, "preview": "..."}
  {"type": "token",       "text": token}
  {"type": "done"}
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Generator

_OLLAMA_MODEL      = "llama3.1:latest"
_MAX_ROUNDS        = 5      # max tool-call rounds before forcing an answer
_MAX_ROWS_CTX      = 15     # rows passed back to LLM per tool result
_MAX_CTX_CHARS     = 8_000  # total chars of accumulated tool results in messages
_MAX_ANSWER_CHARS  = 4_000  # hard cap on final streamed answer

# ── Tool definitions (Ollama function-calling schema) ─────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": (
                "Search order win announcements (contracts, L1 bids, work orders) from the BSE/NSE database. "
                "Returns company, order value in Cr, sector, client, and date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":  {"type": "string",  "description": "Stock symbol (e.g. LT, AFCONS). Omit for all companies."},
                    "sector":  {"type": "string",  "description": "Sector (e.g. 'aerospace & defence', 'railways', 'roads', 'power & utilities'). Omit for all."},
                    "from_dt": {"type": "string",  "description": "Start date YYYY-MM-DD."},
                    "to_dt":   {"type": "string",  "description": "End date YYYY-MM-DD."},
                    "min_cr":  {"type": "number",  "description": "Minimum order value in Crores."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_financials",
            "description": (
                "Search quarterly/annual financial results from the BSE/NSE database. "
                "Returns revenue, PAT, growth %, EBITDA margin, EPS, and period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":      {"type": "string", "description": "Stock symbol. Omit for all companies."},
                    "sector":      {"type": "string", "description": "Sector (e.g. 'healthcare', 'bank', 'i.t', 'fmcg', 'auto')."},
                    "from_dt":     {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "to_dt":       {"type": "string", "description": "End date YYYY-MM-DD."},
                    "period_type": {"type": "string", "enum": ["quarterly", "annual"], "description": "Period type filter."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_announcements",
            "description": (
                "Search corporate announcements from the BSE/NSE database. "
                "Use for acquisitions, management changes, dividends, buybacks, credit ratings, insolvency (CIRP), and fundraising."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":  {"type": "string", "description": "Stock symbol."},
                    "subject": {"type": "string", "description": "Announcement type (e.g. 'Acquisition', 'Dividend', 'Buyback', 'Change in Management', 'Credit Rating- Revision')."},
                    "from_dt": {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "to_dt":   {"type": "string", "description": "End date YYYY-MM-DD."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_breakouts",
            "description": (
                "Search volume breakout signals — stocks with unusual volume surges (HVY signals). "
                "Use when asked about breakouts, momentum, volume spikes, or technical signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":    {"type": "string", "description": "Stock symbol."},
                    "sector":    {"type": "string", "description": "Sector name."},
                    "marketcap": {"type": "string", "enum": ["smallcap", "midcap", "largecap"], "description": "Market cap filter."},
                    "from_dt":   {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "to_dt":     {"type": "string", "description": "End date YYYY-MM-DD."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_orders",
            "description": (
                "Aggregate order win totals — sum of values, count, or breakdown by sector or company. "
                "Use for 'total orders', 'how much', 'which sector had most orders', or ranking questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector":   {"type": "string", "description": "Filter by sector."},
                    "from_dt":  {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "to_dt":    {"type": "string", "description": "End date YYYY-MM-DD."},
                    "group_by": {"type": "string", "enum": ["sector", "company", "date"], "description": "Grouping dimension."},
                },
                "required": [],
            },
        },
    },
]


def _make_system_prompt() -> str:
    return f"""You are an expert Indian equity research analyst with live access to a BSE/NSE announcements database.

Today's date: {date.today().isoformat()}

You have 5 database tools. Use them BEFORE answering — never guess from memory.

Tool selection guide:
- search_orders       → contracts, L1 bids, work orders, order wins
- search_financials   → revenue, PAT, profit, earnings, EBITDA, margins, EPS
- search_announcements → acquisitions, dividends, buybacks, management changes, credit ratings
- search_breakouts    → volume breakouts, HVY signals, momentum, high-volume stocks
- aggregate_orders    → total order value, sector rankings, count summaries

Rules:
1. Always call at least one tool before answering
2. For multi-part questions, call multiple tools in sequence
3. For comparisons (e.g. INFY vs TCS), call search_financials once per company
4. If a tool returns 0 results, retry with a wider date range or relaxed filters
5. Once you have the data, give a direct, specific answer with exact Rs. Cr values and dates
6. Use Indian format: Rs. X Cr (not millions or dollars)"""


# ── Result summariser ─────────────────────────────────────────────────────────

def _preview(name: str, rows: list[dict]) -> str:
    if not rows:
        return "0 results"
    n = len(rows)
    r = rows[0]
    if name == "search_orders":
        val = f"Rs.{r['order_value_cr']:,.0f}Cr" if r.get("order_value_cr") else "value undisclosed"
        return f"{n} orders — top: {r.get('symbol')} {val} ({str(r.get('broadcast_dt',''))[:10]})"
    if name == "search_financials":
        rev = f"Rev Rs.{r['revenue_cr']:,.0f}Cr" if r.get("revenue_cr") else ""
        pat = f"PAT Rs.{r['pat_cr']:,.0f}Cr"     if r.get("pat_cr")     else ""
        return f"{n} results — top: {r.get('symbol')} {rev} {pat} [{r.get('period','')}]"
    if name == "search_announcements":
        return f"{n} announcements — top: {r.get('symbol')} | {r.get('subject','')} ({str(r.get('broadcast_dt',''))[:10]})"
    if name == "search_breakouts":
        return f"{n} breakouts — top: {r.get('symbol')} {r.get('marketcap','')} ({r.get('signal_date','')})"
    if name == "aggregate_orders":
        total = sum(row.get("total_cr", 0) or 0 for row in rows)
        return f"{n} groups — total Rs.{total:,.0f}Cr"
    return f"{n} results"


# ── ToolAgent ─────────────────────────────────────────────────────────────────

class ToolAgent:
    def __init__(self, db_path: Path | str, chroma_path: Path | str) -> None:
        from api.chat_handler import ChatHandler
        self._handler   = ChatHandler(chroma_path=chroma_path, db_path=db_path)
        self._db_path   = Path(db_path)

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute(self, name: str, args: dict) -> list[dict]:
        from_dt = _parse_date(args.get("from_dt"))
        to_dt   = _parse_date(args.get("to_dt"))

        if name == "search_orders":
            if args.get("symbol"):
                return self._handler._sql_order_wins_by_symbol(args["symbol"].upper())
            _f = from_dt or date.today() - timedelta(days=30)
            _t = to_dt   or date.today()
            return self._handler._sql_order_wins_by_date(
                _f, _t, n=50,
                min_cr=args.get("min_cr"),
                sector=args.get("sector"),
            )

        if name == "search_financials":
            if args.get("symbol"):
                return self._handler._sql_financials_by_symbol(args["symbol"].upper())
            sec = args.get("sector")
            if sec and not from_dt:
                return self._handler._sql_financials_by_sector(sec, n=30)
            _f = from_dt or date.today() - timedelta(days=30)
            _t = to_dt   or date.today()
            return self._handler._sql_financials_by_date(
                _f, _t, n=30,
                sector=sec,
                period_type=args.get("period_type"),
            )

        if name == "search_announcements":
            if args.get("symbol"):
                return self._handler._sql_announcements_by_symbol(args["symbol"].upper())
            _f = from_dt or date.today() - timedelta(days=30)
            _t = to_dt   or date.today()
            return self._handler._sql_announcements_by_date(_f, _t, subject=args.get("subject"))

        if name == "search_breakouts":
            return self._handler._sql_volume_breakouts(
                symbol    = args.get("symbol"),
                sector    = args.get("sector"),
                marketcap = args.get("marketcap"),
                from_dt   = str(from_dt) if from_dt else None,
                to_dt     = str(to_dt)   if to_dt   else None,
                n         = 50,
            )

        if name == "aggregate_orders":
            return self._aggregate_orders(args, from_dt, to_dt)

        return []

    def _aggregate_orders(
        self,
        args:    dict,
        from_dt: date | None,
        to_dt:   date | None,
    ) -> list[dict]:
        _f = (from_dt or date.today() - timedelta(days=30)).isoformat()
        _t = (to_dt   or date.today()).isoformat()
        group_by = args.get("group_by", "company")
        sector   = args.get("sector")

        sector_clause = ""
        params: list = [_f, _t]
        if sector:
            sector_clause = "AND LOWER(order_sector) LIKE ?"
            params.append(f"%{sector.lower()}%")

        if group_by == "sector":
            sql = f"""
                SELECT COALESCE(order_sector,'Other') as grp, COUNT(*) as cnt,
                       SUM(COALESCE(order_value_cr,0)) as total_cr
                FROM order_wins
                WHERE DATE(broadcast_dt) BETWEEN ? AND ? {sector_clause}
                GROUP BY 1 ORDER BY total_cr DESC LIMIT 15
            """
        elif group_by == "date":
            sql = f"""
                SELECT DATE(broadcast_dt) as grp, COUNT(*) as cnt,
                       SUM(COALESCE(order_value_cr,0)) as total_cr
                FROM order_wins
                WHERE DATE(broadcast_dt) BETWEEN ? AND ? {sector_clause}
                GROUP BY 1 ORDER BY 1 DESC LIMIT 30
            """
        else:
            sql = f"""
                SELECT symbol as grp, company, COUNT(*) as cnt,
                       SUM(COALESCE(order_value_cr,0)) as total_cr
                FROM order_wins
                WHERE DATE(broadcast_dt) BETWEEN ? AND ? {sector_clause}
                GROUP BY symbol ORDER BY total_cr DESC LIMIT 20
            """

        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        if group_by == "company":
            return [{"group": r[0], "company": r[1], "count": r[2], "total_cr": round(r[3], 2)} for r in rows]
        return [{"group": r[0], "count": r[1], "total_cr": round(r[2], 2)} for r in rows]

    # ── Agentic loop ──────────────────────────────────────────────────────────

    def stream(
        self,
        message: str,
        history: list[dict],
    ) -> Generator[dict, None, None]:
        import ollama

        messages: list[Any] = [{"role": "system", "content": _make_system_prompt()}]
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:800]})
        messages.append({"role": "user", "content": message})

        yield {"type": "status", "text": "Analyzing query…"}

        for _round in range(_MAX_ROUNDS):
            yield {"type": "status", "text": f"Selecting tool (round {_round + 1})…" if _round > 0 else "Selecting tool…"}
            try:
                resp = ollama.chat(
                    model=_OLLAMA_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    stream=False,
                    options={"temperature": 0},
                )
            except Exception as e:
                yield {"type": "token", "text": f"⚠️ Agent error: {e}"}
                yield {"type": "done"}
                return

            if not resp.message.tool_calls:
                # No more tools — stream final answer
                messages.append({"role": "assistant", "content": resp.message.content or ""})
                yield from _stream_final(messages)
                return

            # Execute each tool call
            messages.append(resp.message)

            # Guard: measure total tool-result chars already in context
            total_ctx_chars = sum(
                len(m.get("content", "")) for m in messages if m.get("role") == "tool"
            )

            for tc in resp.message.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}

                # Action guardrail: only allow known tool names
                if name not in {t["function"]["name"] for t in TOOLS}:
                    yield {"type": "tool_result", "tool": name,
                           "count": 0, "preview": "blocked: unknown tool"}
                    continue

                yield {"type": "tool_call", "tool": name, "args": args}

                try:
                    result = self._execute(name, args)
                except Exception as e:
                    result = [{"error": str(e)}]

                yield {"type": "tool_result", "tool": name,
                       "count": len(result), "preview": _preview(name, result)}

                result_json = json.dumps(result[:_MAX_ROWS_CTX], default=str)

                # Context size guardrail: stop adding tool results if already at limit
                if total_ctx_chars + len(result_json) > _MAX_CTX_CHARS:
                    messages.append({
                        "role":    "tool",
                        "content": json.dumps(result[:3], default=str) + "\n[truncated — context limit reached]",
                    })
                    total_ctx_chars = _MAX_CTX_CHARS  # mark as full
                else:
                    messages.append({"role": "tool", "content": result_json})
                    total_ctx_chars += len(result_json)

        # Max rounds — force answer with available data
        yield {"type": "token", "text": "*(max tool calls reached — answering with available data)*\n\n"}
        yield from _stream_final(messages)


def _stream_final(messages: list) -> Generator[dict, None, None]:
    import ollama
    total = 0
    try:
        stream = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0, "num_predict": -1},
        )
        for chunk in stream:
            try:    token = chunk.message.content
            except: token = chunk["message"]["content"]
            if token:
                total += len(token)
                if total > _MAX_ANSWER_CHARS:
                    yield {"type": "token", "text": "\n\n*(answer truncated — response too long)*"}
                    break
                yield {"type": "token", "text": token}
    except Exception as e:
        yield {"type": "token", "text": f"\n\n⚠️ Error streaming answer: {e}"}
    yield {"type": "done"}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None
