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

import time as _time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from api.chat_handler import ChatHandler
from api.llm_client import RateLimitError, _rate_limit_wait

# ── Simple TTL in-memory cache ────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}

def _c_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and entry[0] > _time.time():
        return entry[1]
    _cache.pop(key, None)
    return None

def _c_set(key: str, value: Any, ttl: int = 300) -> None:
    _cache[key] = (_time.time() + ttl, value)

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
    import traceback as _tb
    global handler, tool_agent

    # R2 download before ChatHandler (which opens the DB)
    try:
        from storage.r2_sync import download_if_needed as _r2_download
        await asyncio.get_event_loop().run_in_executor(None, _r2_download, _HERE / "data")
    except Exception as _e:
        _log.warning("R2 download skipped: %s", _e)

    # Heavy init — each step wrapped so errors are visible in logs
    try:
        _log.info("[startup] Initializing ChatHandler …")
        handler = ChatHandler(chroma_path=cfg.chroma_path, db_path=cfg.db_path)
        _log.info("[startup] ChatHandler OK")
    except Exception:
        _log.error("[startup] ChatHandler FAILED:\n%s", _tb.format_exc())
        raise

    # ToolAgent loaded lazily on first /api/chat/agent request (avoids double ChromaDB load)

    try:
        from api.brief_agent import ensure_table as _ensure_brief_table
        await asyncio.get_event_loop().run_in_executor(None, _ensure_brief_table, cfg.db_path)
    except Exception as _e:
        _log.warning("[startup] brief table init failed: %s", _e)

    _log.info("[startup] All components ready — starting scheduler")
    task = asyncio.create_task(_scheduler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App setup ─────────────────────────────────────────────────────────────────

handler:    "ChatHandler | None" = None
tool_agent: "ToolAgent | None"   = None

app = FastAPI(title="EP News AI", version="2.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _require_ready():
    if handler is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service warming up — retry in a moment")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ping", include_in_schema=False)
async def ping():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/stats")
async def stats():
    _require_ready()
    return handler.stats()



@app.get("/api/suggestions")
async def suggestions():
    _require_ready()
    return {"suggestions": handler.suggestions}


@app.get("/api/results")
async def results(days: int = 30, sort: str = "date"):
    """Recent quarterly earnings — used by the Results tab."""
    ck = f"results:{days}:{sort}"
    hit = _c_get(ck)
    if hit is not None:
        return hit
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
    data = [
        {"symbol": r[0], "company": r[1] or "", "period": r[2] or "",
         "revenue_cr": r[3], "pat_cr": r[4], "pat_growth_pct": r[5],
         "broadcast_dt": (r[6] or "")[:10]}
        for r in rows
    ]
    _c_set(ck, data, ttl=300)
    return data


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
    ck = f"breakouts:{days}:{sector}:{marketcap}"
    hit = _c_get(ck)
    if hit is not None:
        return hit
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
    _c_set(ck, enriched, ttl=300)
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

_VAGUE_RE = __import__("re").compile(
    r'^\s*(top\s*\d*\s*stocks?'
    r'|best\s+stocks?'
    r'|good\s+stocks?'
    r'|hot\s+stocks?'
    r'|which\s+stocks?\s+(to\s+)?(buy|invest)'
    r'|show\s+(me\s+)?stocks?'
    r'|recommend\s+(me\s+)?stocks?'
    r'|any\s+good\s+(stock|company|pick)'
    r')\s*[?!.]*\s*$',
    __import__("re").IGNORECASE,
)

# Greetings, social pleasantries, and noise — catch before any DB/LLM call
_OFFTOPIC_RE = __import__("re").compile(
    r'^\s*('
    # greetings
    r'hi+|hello+|hey+|heyy+|hii+|helo|hola|howdy|namaste|namaskar|sup|yo\b'
    # how-are-you variants
    r'|how\s+are\s+(you|r\s+u)|what\'?s?\s+up|wh?atsup|wassup|wazz?up'
    # thanks / bye
    r'|thank(s|\s+you)?|thx|ty\b|bye+|goodbye|cya|see\s+ya'
    # random / test
    r'|test(ing)?|check(ing)?|hello\s+world|ping|ok+|okay|k\b|lol|haha|lmao'
    # pure noise: no letters or only 1-2 real chars
    r'|[^a-zA-Z]*|[a-zA-Z]{1,2}'
    r')\s*[?!.,]*\s*$',
    __import__("re").IGNORECASE,
)

_OFFTOPIC_REPLY = (
    "👋 Hi! I'm **EP News AI** — I search BSE/NSE filings for equity research.\n\n"
    "I can help you with:\n"
    "📦 **Order wins** — e.g. \"Defence order wins above Rs.100 Cr this month\"\n"
    "📈 **Financial results** — e.g. \"Companies with PAT growth above 20%\"\n"
    "⚡ **Volume breakouts** — e.g. \"Breakout stocks in pharma last 14 days\"\n"
    "🏭 **Sector news** — e.g. \"Latest railway sector announcements\"\n\n"
    "Or use the **🔍 Screener** tab for instant pre-built filters — no typing needed!"
)

_VAGUE_REPLY = (
    "To find the right stocks, please tell me what you're looking for:\n\n"
    "📦 **Order wins** — e.g. \"Defence order wins above Rs.100 Cr this month\"\n"
    "📈 **Financial results** — e.g. \"Companies with PAT growth above 20% this quarter\"\n"
    "⚡ **Volume breakouts** — e.g. \"Breakout stocks in pharma sector last 14 days\"\n"
    "🏭 **By sector** — e.g. \"Top order wins in railways or defence\"\n"
    "📊 **EBITDA / margins** — e.g. \"Companies with EBITDA margin above 15%\"\n\n"
    "Just type your question and I'll search our BSE/NSE database!"
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
    _require_ready()
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

    if _VAGUE_RE.match(req.message.strip()):
        async def _vague():
            yield f"data: {json.dumps({'type':'meta','intent':'all','count':0,'sources':[]})}\n\n"
            yield f"data: {json.dumps({'type':'token','text':_VAGUE_REPLY})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_vague(), media_type="text/event-stream")

    if _OFFTOPIC_RE.match(req.message.strip()):
        async def _offtopic():
            yield f"data: {json.dumps({'type':'meta','intent':'all','count':0,'sources':[]})}\n\n"
            yield f"data: {json.dumps({'type':'token','text':_OFFTOPIC_REPLY})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_offtopic(), media_type="text/event-stream")

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
        try:
            if _WEB_SEARCH:
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
                for token in handler.stream_response(
                    message = msg,
                    results = results,
                    intent  = intent,
                    history = req.history,
                    min_cr  = min_cr,
                ):
                    yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

        except RateLimitError as exc:
            wait = max(int(_rate_limit_wait(str(exc)) + 1), 10)
            yield f"data: {json.dumps({'type':'rate_limit','seconds':wait})}\n\n"

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

    global tool_agent
    if tool_agent is None:
        from api.tool_agent import ToolAgent as _ToolAgent
        tool_agent = _ToolAgent(db_path=cfg.db_path, chroma_path=cfg.chroma_path)

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


@app.get("/api/screen")
async def screen(filter_id: str):
    """Run a pre-defined screener query — zero LLM calls, pure SQL."""
    _FIN  = ("symbol, company, period, revenue_cr, pat_cr, "
             "pat_growth_pct, revenue_growth_pct, ebitda_margin_pct, broadcast_dt")
    _FIN_EX = ("period_type NOT IN ('order_win','acquisition','restructuring',"
               "'credit_rating','cirp','fundraising','buyback','open_offer')")
    _ANN  = "symbol, company, subject, order_value_cr, sector_tags, broadcast_dt"
    _BRK  = ("vb.symbol, vb.company, vb.signal_date, vb.marketcap, "
             "COALESCE(vb.sector, ss.sector) as sector, vb.close, vb.per_chg, vb.volume")
    _BFRM = "volume_breakouts vb LEFT JOIN stock_sectors ss ON ss.symbol = vb.symbol"

    FILTERS: dict[str, dict] = {
        # ── FINANCIALS ─────────────────────────────────────────────────────────
        "fin_pat_growth_20":  {"label": "PAT Growth > 20%",         "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND pat_growth_pct > 20 ORDER BY pat_growth_pct DESC LIMIT 100"},
        "fin_pat_growth_50":  {"label": "PAT Growth > 50%",         "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND pat_growth_pct > 50 ORDER BY pat_growth_pct DESC LIMIT 100"},
        "fin_pat_growth_100": {"label": "PAT Growth > 100%",        "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND pat_growth_pct > 100 ORDER BY pat_growth_pct DESC LIMIT 100"},
        "fin_rev_growth_15":  {"label": "Revenue Growth > 15%",     "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND revenue_growth_pct > 15 ORDER BY revenue_growth_pct DESC LIMIT 100"},
        "fin_rev_growth_30":  {"label": "Revenue Growth > 30%",     "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND revenue_growth_pct > 30 ORDER BY revenue_growth_pct DESC LIMIT 100"},
        "fin_ebitda_20":      {"label": "EBITDA Margin > 20%",      "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND ebitda_margin_pct > 20 ORDER BY ebitda_margin_pct DESC LIMIT 100"},
        "fin_ebitda_30":      {"label": "EBITDA Margin > 30%",      "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND ebitda_margin_pct > 30 ORDER BY ebitda_margin_pct DESC LIMIT 100"},
        "fin_turnaround":     {"label": "Turnaround (Loss→Profit)", "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND pat_growth_pct > 200 AND pat_cr > 0 ORDER BY broadcast_dt DESC LIMIT 100"},
        "fin_revenue_100":    {"label": "Revenue > ₹100 Cr",        "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND revenue_cr > 100 ORDER BY revenue_cr DESC LIMIT 100"},
        "fin_revenue_500":    {"label": "Revenue > ₹500 Cr",        "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND revenue_cr > 500 ORDER BY revenue_cr DESC LIMIT 100"},
        "fin_recent_7d":      {"label": "Results This Week",         "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE {_FIN_EX} AND DATE(broadcast_dt) >= DATE('now','-7 days') ORDER BY broadcast_dt DESC LIMIT 100"},
        "fin_annual":         {"label": "Annual Results (FY)",       "type": "financials",
            "sql": f"SELECT {_FIN} FROM financial_results WHERE period_type='annual' ORDER BY broadcast_dt DESC LIMIT 100"},
        # ── ORDER WINS ─────────────────────────────────────────────────────────
        "ord_defence_100": {"label": "Defence Orders > ₹100 Cr", "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 100 AND (LOWER(subject) LIKE '%defence%' OR LOWER(subject) LIKE '%military%' OR LOWER(subject) LIKE '%army%' OR LOWER(subject) LIKE '%navy%' OR LOWER(subject) LIKE '%drdo%' OR LOWER(sector_tags) LIKE '%defence%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_defence_500": {"label": "Defence Orders > ₹500 Cr", "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 500 AND (LOWER(subject) LIKE '%defence%' OR LOWER(subject) LIKE '%military%' OR LOWER(subject) LIKE '%army%' OR LOWER(sector_tags) LIKE '%defence%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_railway_50":  {"label": "Railway Orders > ₹50 Cr",  "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 50 AND (LOWER(subject) LIKE '%railway%' OR LOWER(subject) LIKE '%rail%' OR LOWER(sector_tags) LIKE '%railway%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_power_50":    {"label": "Power Sector Orders > ₹50 Cr", "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 50 AND (LOWER(subject) LIKE '%power%' OR LOWER(subject) LIKE '%energy%' OR LOWER(subject) LIKE '%solar%' OR LOWER(subject) LIKE '%transmission%' OR LOWER(sector_tags) LIKE '%power%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_mega_1000":   {"label": "Mega Orders > ₹1,000 Cr",  "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 1000 ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_large_500":   {"label": "Large Orders > ₹500 Cr",   "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 500 ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_infra_100":   {"label": "Infra Orders > ₹100 Cr",   "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 100 AND (LOWER(subject) LIKE '%infrastructure%' OR LOWER(subject) LIKE '%infra%' OR LOWER(subject) LIKE '%road%' OR LOWER(subject) LIKE '%highway%' OR LOWER(sector_tags) LIKE '%infra%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_all_7d":      {"label": "All Orders This Week",      "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 0 AND DATE(broadcast_dt) >= DATE('now','-7 days') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_govt":        {"label": "Govt / PSU Orders",         "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 0 AND (LOWER(subject) LIKE '%government%' OR LOWER(subject) LIKE '%govt%' OR LOWER(subject) LIKE '%ministry%' OR LOWER(subject) LIKE '%municipal%' OR LOWER(subject) LIKE '%nhai%' OR LOWER(subject) LIKE '%ntpc%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_export":      {"label": "Export Orders",             "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 0 AND (LOWER(subject) LIKE '%export%' OR LOWER(subject) LIKE '%international%' OR LOWER(subject) LIKE '%overseas%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_water":       {"label": "Water / Irrigation Orders", "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 0 AND (LOWER(subject) LIKE '%water%' OR LOWER(subject) LIKE '%irrigation%' OR LOWER(subject) LIKE '%sewage%' OR LOWER(subject) LIKE '%jal%') ORDER BY order_value_cr DESC LIMIT 100"},
        "ord_epc":         {"label": "EPC / Construction Orders", "type": "orders",
            "sql": f"SELECT {_ANN} FROM announcements WHERE order_value_cr > 0 AND (LOWER(subject) LIKE '%epc%' OR LOWER(subject) LIKE '%construction%' OR LOWER(subject) LIKE '%turnkey%') ORDER BY order_value_cr DESC LIMIT 100"},
        # ── CORPORATE EVENTS ───────────────────────────────────────────────────
        "evt_dividend":      {"label": "Dividend Announcements",   "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE LOWER(subject) LIKE '%dividend%' ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_buyback":       {"label": "Buyback Announcements",    "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%buyback%' OR LOWER(subject) LIKE '%buy-back%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_bonus_split":   {"label": "Bonus / Stock Split",      "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%bonus%' OR LOWER(subject) LIKE '%stock split%') AND LOWER(subject) NOT LIKE '%revenue%' ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_fundraising":   {"label": "QIP / Fundraising",        "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%qip%' OR LOWER(subject) LIKE '%preferential%' OR LOWER(subject) LIKE '%rights issue%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_acquisition":   {"label": "Acquisitions / Mergers",   "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%acquisition%' OR LOWER(subject) LIKE '%merger%' OR LOWER(subject) LIKE '%takeover%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_mgmt_change":   {"label": "Management Changes",       "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%appointment%' OR LOWER(subject) LIKE '%resignation%' OR LOWER(subject) LIKE '%ceo%' OR LOWER(subject) LIKE '%managing director%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_credit_rating": {"label": "Credit Rating Changes",    "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%credit rating%' OR LOWER(subject) LIKE '%rating upgrade%' OR LOWER(subject) LIKE '%rating downgrade%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_all_7d":        {"label": "All Events This Week",     "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE DATE(broadcast_dt) >= DATE('now','-7 days') AND (order_value_cr IS NULL OR order_value_cr = 0) ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_insolvency":    {"label": "NCLT / Insolvency",        "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%nclt%' OR LOWER(subject) LIKE '%insolvency%' OR LOWER(subject) LIKE '%cirp%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "evt_concall":       {"label": "Investor / Analyst Meets", "type": "events",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(subject) LIKE '%analyst%' OR LOWER(subject) LIKE '%investor meet%' OR LOWER(subject) LIKE '%concall%') ORDER BY broadcast_dt DESC LIMIT 100"},
        # ── BREAKOUTS ──────────────────────────────────────────────────────────
        "brk_all_14d":  {"label": "All Breakouts (14 days)",    "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE vb.signal_date >= DATE('now','-14 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_defence":  {"label": "Defence Breakouts",          "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE (LOWER(COALESCE(vb.sector,'')) LIKE '%defence%' OR LOWER(COALESCE(ss.sector,'')) LIKE '%defence%') AND vb.signal_date >= DATE('now','-30 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_pharma":   {"label": "Pharma Breakouts",           "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE (LOWER(COALESCE(vb.sector,'')) LIKE '%pharma%' OR LOWER(COALESCE(ss.sector,'')) LIKE '%pharma%') AND vb.signal_date >= DATE('now','-30 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_smallcap": {"label": "Smallcap Breakouts",         "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE LOWER(vb.marketcap) = 'small cap' AND vb.signal_date >= DATE('now','-14 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_midcap":   {"label": "Midcap Breakouts",           "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE LOWER(vb.marketcap) = 'mid cap' AND vb.signal_date >= DATE('now','-14 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_it":       {"label": "IT / Tech Breakouts",        "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE (LOWER(COALESCE(vb.sector,'')) LIKE '%software%' OR LOWER(COALESCE(vb.sector,'')) LIKE '%tech%' OR LOWER(COALESCE(ss.sector,'')) LIKE '%informat%') AND vb.signal_date >= DATE('now','-30 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_infra":    {"label": "Infrastructure Breakouts",   "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE (LOWER(COALESCE(vb.sector,'')) LIKE '%infra%' OR LOWER(COALESCE(ss.sector,'')) LIKE '%infra%') AND vb.signal_date >= DATE('now','-30 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        "brk_30d":      {"label": "Breakouts (30 days)",        "type": "breakouts",
            "sql": f"SELECT {_BRK} FROM {_BFRM} WHERE vb.signal_date >= DATE('now','-30 days') ORDER BY vb.signal_date DESC LIMIT 100"},
        # ── SECTORS ────────────────────────────────────────────────────────────
        "sec_defence":   {"label": "Defence Sector",    "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE LOWER(sector_tags) LIKE '%defence%' ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_pharma":    {"label": "Pharma Sector",     "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE LOWER(sector_tags) LIKE '%pharma%' ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_it":        {"label": "IT / Tech Sector",  "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(sector_tags) LIKE '%informat%' OR LOWER(sector_tags) LIKE '%software%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_auto":      {"label": "Auto Sector",       "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE LOWER(sector_tags) LIKE '%auto%' ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_banking":   {"label": "Banking & Finance",  "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(sector_tags) LIKE '%bank%' OR LOWER(sector_tags) LIKE '%financ%' OR LOWER(sector_tags) LIKE '%nbfc%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_realty":    {"label": "Real Estate",       "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(sector_tags) LIKE '%real%' OR LOWER(sector_tags) LIKE '%realt%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_fmcg":      {"label": "FMCG Sector",       "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE (LOWER(sector_tags) LIKE '%fmcg%' OR LOWER(sector_tags) LIKE '%consumer%') ORDER BY broadcast_dt DESC LIMIT 100"},
        "sec_chemicals": {"label": "Chemicals Sector",  "type": "sectors",
            "sql": f"SELECT {_ANN} FROM announcements WHERE LOWER(sector_tags) LIKE '%chemical%' ORDER BY broadcast_dt DESC LIMIT 100"},
    }

    f = FILTERS.get(filter_id)
    if not f:
        raise HTTPException(status_code=404, detail=f"Unknown filter: {filter_id}")

    ck = f"screen:{filter_id}"
    hit = _c_get(ck)
    if hit is not None:
        return hit

    conn = _sqlite3.connect(str(cfg.db_path))
    try:
        rows = conn.execute(f["sql"]).fetchall()
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=str(exc))
    conn.close()

    rtype = f["type"]
    if rtype == "financials":
        data = {
            "type": rtype, "label": f["label"], "count": len(rows),
            "rows": [{"symbol": r[0], "company": r[1] or "", "period": r[2] or "",
                      "revenue_cr": r[3], "pat_cr": r[4], "pat_growth_pct": r[5],
                      "revenue_growth_pct": r[6], "ebitda_margin_pct": r[7],
                      "broadcast_dt": (r[8] or "")[:10]} for r in rows],
        }
    elif rtype in ("orders", "events", "sectors"):
        data = {
            "type": rtype, "label": f["label"], "count": len(rows),
            "rows": [{"symbol": r[0], "company": r[1] or "", "subject": r[2] or "",
                      "order_value_cr": r[3], "sector_tags": r[4] or "",
                      "broadcast_dt": (r[5] or "")[:10]} for r in rows],
        }
    else:
        # breakouts
        data = {
            "type": rtype, "label": f["label"], "count": len(rows),
            "rows": [{"symbol": r[0], "company": r[1] or "", "signal_date": r[2] or "",
                      "marketcap": r[3] or "", "sector": r[4] or "",
                      "close": r[5], "per_chg": r[6], "volume": r[7]} for r in rows],
        }
    _c_set(ck, data, ttl=600)   # screener results: 10 min cache
    return data


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
