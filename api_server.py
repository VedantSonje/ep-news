"""
EP News API Server — FastAPI backend with streaming RAG chat.

Start:
    python api_server.py

Endpoints:
    GET  /              → chat UI
    GET  /api/stats     → DB + ChromaDB counts
    GET  /api/suggestions → quick query chips
    POST /api/chat      → SSE stream: meta event then token events

Auto-update:
    On startup, if the DB is stale (last entry is not today) a catch-up run fires
    immediately.  After that, a full update runs every day at 22:00 local time:
      1. fetch_today.py  — pull new NSE announcements
      2. main.py extract — download PDFs + extract order wins / financial results
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3 as _sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from api.chat_handler import ChatHandler

import os as _os
_USE_AGENT   = _os.getenv("USE_AGENT",   "0").lower() in ("1", "true", "yes")
_WEB_SEARCH  = _os.getenv("WEB_SEARCH",  "0").lower() in ("1", "true", "yes")

# ── API key auth for write / admin endpoints ──────────────────────────────────
_ADMIN_API_KEY = _os.getenv("EP_ADMIN_KEY", "")   # set EP_ADMIN_KEY in .env to enable

def _require_admin(x_api_key: str | None) -> None:
    """Raise 403 if admin key is configured and the request doesn't match."""
    if _ADMIN_API_KEY and x_api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

_log = logging.getLogger("ep_news.scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

_HERE = Path(__file__).parent
cfg   = AppConfig.from_env()
cfg.ensure_dirs()

# R2 download must happen before ChatHandler init (which opens the DB)
try:
    from storage.r2_sync import download_if_needed as _r2_download
    _r2_download(_HERE / "data")
except Exception as _e:
    _log.warning("R2 download skipped: %s", _e)


# ── Auto-update helpers ───────────────────────────────────────────────────────

def _last_db_date() -> date | None:
    """Return the most recent broadcast date in the announcements table."""
    try:
        conn = _sqlite3.connect(str(cfg.db_path))
        row  = conn.execute("SELECT MAX(DATE(broadcast_dt)) FROM announcements").fetchone()
        conn.close()
        if row and row[0]:
            return date.fromisoformat(row[0])
    except Exception:
        pass
    return None


async def _run_cmd(*args: str) -> int:
    """Run a subprocess command, stream stdout to logger, return exit code."""
    _log.info("  $ %s", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(_HERE),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    async for line in proc.stdout:
        _log.info("    %s", line.decode(errors="replace").rstrip())
    await proc.wait()
    return proc.returncode


async def _do_update(target_date: date) -> None:
    """Fetch + extract for target_date."""
    date_str = target_date.isoformat()
    _log.info("=== Auto-update for %s ===", date_str)

    rc = await _run_cmd(sys.executable, "fetch_today.py", "--date", date_str)
    if rc != 0:
        _log.warning("fetch_today.py exited %d", rc)

    rc = await _run_cmd(sys.executable, "main.py", "extract", "--date", date_str, "--workers", "8")
    if rc != 0:
        _log.warning("main.py extract exited %d", rc)

    # Fetch Chartink breakout signals
    rc = await _run_cmd(sys.executable, "fetch_breakouts.py", "--date", date_str)
    if rc != 0:
        _log.warning("fetch_breakouts.py exited %d", rc)

    # Generate daily brief after ingestion
    _log.info("Generating daily brief for %s…", date_str)
    try:
        from api.brief_agent import generate_daily_brief
        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_daily_brief, cfg.db_path, date_str
        )
        if result:
            _log.info("Brief generated: %d chars, %d highlights", len(result["content"]), len(result["highlights"]))
        else:
            _log.info("Brief skipped (no data for %s)", date_str)
    except Exception as exc:
        _log.warning("Brief generation failed: %s", exc)

    _log.info("=== Auto-update complete for %s ===", date_str)


def _dates_missing_briefs() -> list[date]:
    """Return dates that have announcement data but no stored brief."""
    try:
        conn = _sqlite3.connect(str(cfg.db_path))
        rows = conn.execute("""
            SELECT DISTINCT DATE(broadcast_dt) as d
            FROM announcements
            WHERE DATE(broadcast_dt) NOT IN (SELECT brief_date FROM daily_briefs)
            ORDER BY d
        """).fetchall()
        conn.close()
        return [date.fromisoformat(r[0]) for r in rows if r[0]]
    except Exception:
        return []


async def _backfill_briefs() -> None:
    """Generate briefs for any dates that have data but no stored brief."""
    missing = _dates_missing_briefs()
    if not missing:
        return
    _log.info("Backfilling briefs for %d dates: %s … %s", len(missing), missing[0], missing[-1])
    for d in missing:
        _log.info("  Generating brief for %s…", d)
        try:
            from api.brief_agent import generate_daily_brief
            result = await asyncio.get_event_loop().run_in_executor(
                None, generate_daily_brief, cfg.db_path, d.isoformat()
            )
            if result:
                _log.info("  Brief for %s: %d chars", d, len(result["content"]))
            else:
                _log.info("  Brief for %s: skipped (no data)", d)
        except Exception as exc:
            _log.warning("  Brief for %s failed: %s", d, exc)


async def _scheduler() -> None:
    """
    1. On startup: backfill briefs for any dates missing one, then catch up new data.
    2. Then: every day at 22:00 local time, update for today.
    """
    today     = date.today()
    last_date = _last_db_date()

    # ── Backfill briefs for dates with data but no brief ─────────────────────
    await _backfill_briefs()

    # ── Startup catch-up ──────────────────────────────────────────────────────
    if last_date is None or last_date < today:
        start = (last_date + timedelta(days=1)) if last_date else today
        d = start
        while d <= today:
            await _do_update(d)
            d += timedelta(days=1)

    # ── Daily loop at 22:00 ───────────────────────────────────────────────────
    while True:
        now    = datetime.now()
        target = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        delay = (target - now).total_seconds()
        _log.info("Next auto-update scheduled at %s (in %.0f min)", target.strftime("%Y-%m-%d %H:%M"), delay / 60)
        await asyncio.sleep(delay)
        await _do_update(date.today())


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app):
    # Ensure daily_briefs table exists before scheduler starts
    from api.brief_agent import ensure_table as _ensure_brief_table
    await asyncio.get_event_loop().run_in_executor(None, _ensure_brief_table, cfg.db_path)

    task = asyncio.create_task(_scheduler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App setup ─────────────────────────────────────────────────────────────────

app         = FastAPI(title="EP News AI", version="2.0", lifespan=lifespan)
handler     = ChatHandler(chroma_path=cfg.chroma_path, db_path=cfg.db_path)

from api.tool_agent import ToolAgent
tool_agent  = ToolAgent(db_path=cfg.db_path, chroma_path=cfg.chroma_path)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/stats")
async def stats():
    return handler.stats()


@app.get("/api/suggestions")
async def suggestions():
    return {"suggestions": handler.suggestions}


@app.get("/api/results")
async def results(days: int = 30, sort: str = "date"):
    """Recent quarterly earnings — used by the Results tab."""
    conn = _sqlite3.connect(str(cfg.db_path))
    order_col = {
        "revenue": "COALESCE(revenue_cr, -1e9) DESC",
        "pat":     "COALESCE(pat_cr,     -1e9) DESC",
        "date":    "broadcast_dt DESC",
    }.get(sort, "broadcast_dt DESC")
    date_clause = (
        f"AND DATE(broadcast_dt) >= DATE('now', '-{int(days)} days')" if days > 0 else ""
    )
    rows = conn.execute(f"""
        SELECT symbol, company, period, revenue_cr, pat_cr, pat_growth_pct, broadcast_dt
        FROM financial_results
        WHERE period_type NOT IN
          ('order_win','acquisition','restructuring','credit_rating',
           'cirp','fundraising','buyback','open_offer')
        {date_clause}
        ORDER BY {order_col}
        LIMIT 300
    """).fetchall()
    conn.close()
    return [
        {"symbol": r[0], "company": r[1] or "", "period": r[2] or "",
         "revenue_cr": r[3], "pat_cr": r[4], "pat_growth_pct": r[5],
         "broadcast_dt": (r[6] or "")[:10]}
        for r in rows
    ]


@app.get("/api/compare/symbols")
async def compare_symbols():
    """All companies that have financial_statements rows."""
    conn = _sqlite3.connect(str(cfg.db_path))
    rows = conn.execute(
        "SELECT DISTINCT symbol, company FROM financial_statements ORDER BY symbol"
    ).fetchall()
    conn.close()
    return [{"symbol": r[0], "company": r[1] or ""} for r in rows]


@app.get("/api/compare")
async def compare(symbol: str):
    """Latest financial statement for a symbol — all line items, all 4 columns."""
    conn = _sqlite3.connect(str(cfg.db_path))
    row = conn.execute(
        """SELECT symbol, company, unit, col1_label, col2_label, col3_label, col4_label,
                  line_items, broadcast_dt, source_url
           FROM financial_statements WHERE UPPER(symbol)=UPPER(?)
           ORDER BY broadcast_dt DESC LIMIT 1""",
        (symbol,),
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"No financial statement found for {symbol}"}
    sym, company, unit, c1, c2, c3, c4, items_json, bdt, src_url = row
    return {
        "symbol":     sym,
        "company":    company or "",
        "unit":       unit or "Cr",
        "col1":       c1 or "",
        "col2":       c2 or "",
        "col3":       c3 or "",
        "col4":       c4 or "",
        "broadcast_dt": (bdt or "")[:10],
        "source_url": src_url or "",
        "line_items": json.loads(items_json),
    }


@app.get("/api/breakouts")
async def breakouts(days: int = 14, sector: str = "", marketcap: str = ""):
    """Volume breakout signals enriched with nearest DB reason (earnings / order / announcement)."""
    conn = _sqlite3.connect(str(cfg.db_path))

    where, params = [], []
    if days > 0:
        where.append(f"signal_date >= DATE('now', '-{int(days)} days')")
    if sector:
        where.append("LOWER(sector) LIKE ?")
        params.append(f"%{sector.lower()}%")
    if marketcap and marketcap != "all":
        where.append("LOWER(marketcap) = ?")
        params.append(marketcap.lower())

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(f"""
        SELECT vb.symbol, vb.signal_date,
               vb.marketcap,
               COALESCE(vb.sector, ss.sector) as sector,
               vb.close, vb.per_chg, vb.volume,
               vb.company
        FROM volume_breakouts vb
        LEFT JOIN stock_sectors ss ON ss.symbol = vb.symbol
        {where_sql}
        ORDER BY vb.signal_date DESC LIMIT 300
    """, params).fetchall()

    enriched = []
    for symbol, signal_date, mktcap, sec, close, per_chg, vol, co_chartink in rows:
        # 1) financial result within ±5 days (highest priority)
        fin = conn.execute("""
            SELECT period, revenue_cr, pat_cr, pat_growth_pct, broadcast_dt, company
            FROM financial_results
            WHERE symbol = ?
              AND period_type NOT IN
                ('order_win','acquisition','restructuring','credit_rating',
                 'cirp','fundraising','buyback','open_offer')
              AND DATE(broadcast_dt) BETWEEN DATE(?, '-5 days') AND DATE(?, '+5 days')
            ORDER BY ABS(JULIANDAY(broadcast_dt) - JULIANDAY(?)) LIMIT 1
        """, (symbol, signal_date, signal_date, signal_date)).fetchone()

        # 2) order-win announcement within ±5 days
        ord_ann = conn.execute("""
            SELECT subject, company, broadcast_dt, order_value_cr
            FROM announcements
            WHERE symbol = ? AND LOWER(subject) LIKE '%order%'
              AND DATE(broadcast_dt) BETWEEN DATE(?, '-5 days') AND DATE(?, '+5 days')
            ORDER BY ABS(JULIANDAY(broadcast_dt) - JULIANDAY(?)) LIMIT 1
        """, (symbol, signal_date, signal_date, signal_date)).fetchone()

        # 3) any announcement within ±5 days
        any_ann = conn.execute("""
            SELECT subject, company, broadcast_dt
            FROM announcements
            WHERE symbol = ?
              AND DATE(broadcast_dt) BETWEEN DATE(?, '-5 days') AND DATE(?, '+5 days')
            ORDER BY ABS(JULIANDAY(broadcast_dt) - JULIANDAY(?)) LIMIT 1
        """, (symbol, signal_date, signal_date, signal_date)).fetchone()

        reason_type = reason_text = reason_date = company = None

        if fin:
            period, rev, pat, pat_g, fin_dt, co = fin
            reason_type = "earnings"
            company = co
            parts = []
            if rev and rev > 0: parts.append(f"Rev ₹{rev:,.0f}Cr")
            if pat: parts.append(f"PAT ₹{pat:,.0f}Cr")
            if pat_g and pat_g > -900:
                parts.append(f"{'+' if pat_g > 0 else ''}{pat_g:.1f}% YoY")
            reason_text = (f"{period} | " if period else "") + " · ".join(parts)
            reason_date = (fin_dt or "")[:10]
        elif ord_ann:
            subj, co, ann_dt, oval = ord_ann
            reason_type = "order"
            company = co
            val_str = f" — ₹{oval:,.0f}Cr" if oval and oval > 0 else ""
            reason_text = (subj or "Order Win") + val_str
            reason_date = (ann_dt or "")[:10]
        elif any_ann:
            subj, co, ann_dt = any_ann
            reason_type = "announcement"
            company = co
            reason_text = subj or "Announcement"
            reason_date = (ann_dt or "")[:10]

        enriched.append({
            "symbol":      symbol,
            "signal_date": signal_date,
            "marketcap":   mktcap or "",
            "sector":      sec or "",
            "company":     company or co_chartink or "",
            "close":       close,
            "per_chg":     per_chg,
            "volume":      vol,
            "reason_type": reason_type,
            "reason_text": reason_text,
            "reason_date": reason_date,
        })

    conn.close()
    return enriched


@app.get("/api/brief")
async def get_brief(date: str = ""):
    """Return the latest stored daily brief (or today's by date param)."""
    from api.brief_agent import get_latest_brief
    brief = await asyncio.get_event_loop().run_in_executor(
        None, get_latest_brief, cfg.db_path, date or None
    )
    if not brief:
        return {"available": False}
    return {"available": True, **brief}


@app.post("/api/brief/generate")
async def trigger_brief(req_date: str = "", x_api_key: str | None = Header(default=None)):
    """Manually trigger brief generation for a date (for testing). Requires EP_ADMIN_KEY."""
    _require_admin(x_api_key)
    from api.brief_agent import generate_daily_brief
    target = req_date or date.today().isoformat()
    result = await asyncio.get_event_loop().run_in_executor(
        None, generate_daily_brief, cfg.db_path, target
    )
    return result or {"error": "No data for that date"}


@app.get("/api/investigate/{symbol}")
async def investigate(symbol: str, signal_date: str = ""):
    """
    SSE stream for the Breakout Investigation Agent.
    Events: step (db/web/llm progress) · token (LLM text) · done
    """
    from api.investigate import investigate_stream

    if not signal_date:
        signal_date = date.today().isoformat()

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _run() -> None:
        try:
            for event in investigate_stream(
                symbol=symbol.upper(),
                signal_date=signal_date,
                db_path=cfg.db_path,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    async def generate():
        loop.run_in_executor(None, _run)
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {event}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class ChatRequest(BaseModel):
    message: str
    n:       int        = 8
    history: list[dict] = []   # [{"role": "user"/"assistant", "content": "..."}]


_CHAT_MAX_CHARS  = 1_000
_INJECT_RE = __import__("re").compile(
    r'(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?'
    r'|you\s+are\s+now\s+a'
    r'|<\s*(system|instructions?)\s*>'
    r'|###\s*(system|instruction))',
    __import__("re").IGNORECASE,
)

def _validate_chat_input(msg: str) -> str | None:
    """Return an error string if the message fails guardrails, else None."""
    if len(msg) > _CHAT_MAX_CHARS:
        return f"Message too long ({len(msg)} chars, max {_CHAT_MAX_CHARS})."
    if _INJECT_RE.search(msg):
        return "Message contains disallowed content."
    return None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    SSE stream. Events:
      {"type": "meta",  "intent": "order_win", "count": 3}
      {"type": "token", "text": "Found 3 ..."}
      {"type": "done"}
    """
    if not req.message.strip():
        async def empty():
            yield f"data: {json.dumps({'type':'token','text':'Please type a question.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    err = _validate_chat_input(req.message.strip())
    if err:
        async def _bad():
            yield f"data: {json.dumps({'type':'token','text':err})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_bad(), media_type="text/event-stream")

    def generate():
        msg = req.message.strip()

        # Stage 1: DB Agent — retrieve (fast, no LLM, ~50ms)
        results, intent, min_cr = handler.retrieve(msg, n=req.n)

        # Detect company symbol — used to decide whether to run web agent
        company_sym = handler._extract_company_symbol(msg) if _WEB_SEARCH else None

        # Collect unique source URLs from retrieved results
        sources: list[str] = []
        seen_urls: set[str] = set()
        for r in results:
            url = r.get("source_url") or r.get("attachment") or ""
            if url and url not in seen_urls and url.endswith(".pdf"):
                seen_urls.add(url)
                sources.append(url)

        # Send metadata first so UI can show badge immediately
        meta: dict = {"type": "meta", "intent": intent, "count": len(results), "sources": sources}
        if _WEB_SEARCH and company_sym:
            meta["web_search"] = True   # tells UI a web search is in progress
            meta["symbol"]     = company_sym
        yield f"data: {json.dumps(meta)}\n\n"

        # Stage 2: generate response
        if _WEB_SEARCH:
            # Dual-agent: DB results + live web search combined
            from api.ep_agent import stream_dual_agent
            for token in stream_dual_agent(
                message     = msg,
                results     = results,
                intent      = intent,
                history     = req.history,
                handler     = handler,
                min_cr      = min_cr,
                company_sym = company_sym,
            ):
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

        elif _USE_AGENT:
            # PydanticAI text agent (no web search)
            from api.ep_agent import stream_with_agent
            for token in stream_with_agent(
                message = msg,
                results = results,
                intent  = intent,
                history = req.history,
                handler = handler,
                min_cr  = min_cr,
            ):
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

        else:
            # Default: raw Ollama streaming (original path)
            for token in handler.stream_response(
                message = msg,
                results = results,
                intent  = intent,
                history = req.history,
                min_cr  = min_cr,
            ):
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/agent")
async def agent_chat(req: ChatRequest):
    """
    Tool-agent SSE stream. Events:
      {"type": "tool_call",   "tool": "search_orders", "args": {...}}
      {"type": "tool_result", "tool": "search_orders", "count": 12, "preview": "..."}
      {"type": "token",       "text": "..."}
      {"type": "done"}
    """
    if not req.message.strip():
        async def _empty():
            yield f"data: {json.dumps({'type':'token','text':'Please type a question.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    err = _validate_chat_input(req.message.strip())
    if err:
        async def _bad_agent():
            yield f"data: {json.dumps({'type':'token','text':err})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_bad_agent(), media_type="text/event-stream")

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _run() -> None:
        try:
            for event in tool_agent.stream(req.message.strip(), req.history):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    async def _generate():
        loop.run_in_executor(None, _run)
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── Watchlist ─────────────────────────────────────────────────────────────────

def _ensure_watchlist(conn: _sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol   TEXT PRIMARY KEY,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


@app.get("/api/watchlist")
async def get_watchlist():
    """Return all symbols on the user's watchlist with latest breakout + result info."""
    conn = _sqlite3.connect(str(cfg.db_path))
    _ensure_watchlist(conn)
    syms = [r[0] for r in conn.execute("SELECT symbol FROM watchlist ORDER BY added_at DESC").fetchall()]
    result = []
    for sym in syms:
        latest_breakout = conn.execute("""
            SELECT signal_date, close, per_chg FROM volume_breakouts
            WHERE symbol = ? ORDER BY signal_date DESC LIMIT 1
        """, (sym,)).fetchone()
        latest_fin = conn.execute("""
            SELECT period, pat_cr, pat_growth_pct FROM financial_results
            WHERE symbol = ? AND period_type NOT IN
              ('order_win','acquisition','restructuring','credit_rating',
               'cirp','fundraising','buyback','open_offer')
            ORDER BY broadcast_dt DESC LIMIT 1
        """, (sym,)).fetchone()
        result.append({
            "symbol": sym,
            "breakout": {"date": latest_breakout[0], "close": latest_breakout[1], "pct_chg": latest_breakout[2]} if latest_breakout else None,
            "financials": {"period": latest_fin[0], "pat_cr": latest_fin[1], "pat_growth_pct": latest_fin[2]} if latest_fin else None,
        })
    conn.close()
    return result


class WatchlistAddRequest(BaseModel):
    symbol: str


@app.post("/api/watchlist")
async def add_to_watchlist(req: WatchlistAddRequest, x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    sym = req.symbol.strip().upper()
    if not sym or not sym.isalnum():
        return {"error": "symbol must be alphanumeric"}
    conn = _sqlite3.connect(str(cfg.db_path))
    _ensure_watchlist(conn)
    conn.execute("INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (sym,))
    conn.commit()
    conn.close()
    return {"added": sym}


@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    sym = symbol.strip().upper()
    conn = _sqlite3.connect(str(cfg.db_path))
    _ensure_watchlist(conn)
    conn.execute("DELETE FROM watchlist WHERE symbol = ?", (sym,))
    conn.commit()
    conn.close()
    return {"removed": sym}


@app.get("/api/watchlist/alerts")
async def watchlist_alerts():
    """Return today's breakout signals for watchlisted symbols."""
    today = date.today().isoformat()
    conn = _sqlite3.connect(str(cfg.db_path))
    _ensure_watchlist(conn)
    syms = [r[0] for r in conn.execute("SELECT symbol FROM watchlist").fetchall()]
    if not syms:
        conn.close()
        return []
    placeholders = ",".join("?" * len(syms))
    alerts = conn.execute(f"""
        SELECT symbol, signal_date, close, per_chg, volume FROM volume_breakouts
        WHERE symbol IN ({placeholders}) AND signal_date = ?
        ORDER BY per_chg DESC
    """, (*syms, today)).fetchall()
    conn.close()
    return [{"symbol": r[0], "date": r[1], "close": r[2], "pct_chg": r[3], "volume": r[4]} for r in alerts]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting EP News AI on http://localhost:8080")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=False)
