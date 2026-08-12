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

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from api.chat_handler import ChatHandler

import os as _os
_USE_AGENT   = _os.getenv("USE_AGENT",   "0").lower() in ("1", "true", "yes")
_WEB_SEARCH  = _os.getenv("WEB_SEARCH",  "0").lower() in ("1", "true", "yes")

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

    _log.info("=== Auto-update complete for %s ===", date_str)


async def _scheduler() -> None:
    """
    1. On startup: catch up any missing dates between last DB entry and today.
    2. Then: every day at 22:00 local time, update for today.
    """
    today     = date.today()
    last_date = _last_db_date()

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
    task = asyncio.create_task(_scheduler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App setup ─────────────────────────────────────────────────────────────────

app     = FastAPI(title="EP News AI", version="2.0", lifespan=lifespan)
handler = ChatHandler(chroma_path=cfg.chroma_path, db_path=cfg.db_path)

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


class ChatRequest(BaseModel):
    message: str
    n:       int        = 8
    history: list[dict] = []   # [{"role": "user"/"assistant", "content": "..."}]


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting EP News AI on http://localhost:8080")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=False)
