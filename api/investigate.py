"""
Breakout Investigation Agent
Three sequential steps: DB context → web news → LLM synthesis (streaming).
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator

_OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


# ── Step 1: DB context ────────────────────────────────────────────────────────

def _db_context(db_path: Path, symbol: str, signal_date: str) -> dict:
    conn = sqlite3.connect(str(db_path))

    anns = conn.execute("""
        SELECT subject, company, broadcast_dt, order_value_cr, sector_tags
        FROM announcements
        WHERE symbol = ? AND DATE(broadcast_dt) >= DATE(?, '-90 days')
        ORDER BY broadcast_dt DESC LIMIT 15
    """, (symbol, signal_date)).fetchall()

    fins = conn.execute("""
        SELECT period, revenue_cr, pat_cr, pat_growth_pct, revenue_growth_pct,
               ebitda_margin_pct, broadcast_dt
        FROM financial_results
        WHERE symbol = ?
          AND period_type NOT IN ('order_win','acquisition','restructuring',
                                  'credit_rating','cirp','fundraising','buyback','open_offer')
        ORDER BY broadcast_dt DESC LIMIT 4
    """, (symbol,)).fetchall()

    orders = conn.execute("""
        SELECT subject, order_value_cr, broadcast_dt
        FROM announcements
        WHERE symbol = ? AND LOWER(subject) LIKE '%order%'
          AND DATE(broadcast_dt) >= DATE(?, '-90 days')
        ORDER BY broadcast_dt DESC LIMIT 5
    """, (symbol, signal_date)).fetchall()

    prev_signals = conn.execute("""
        SELECT signal_date FROM volume_breakouts
        WHERE symbol = ? ORDER BY signal_date DESC LIMIT 8
    """, (symbol,)).fetchall()

    conn.close()

    company = anns[0][1] if anns and anns[0][1] else ""
    sector  = anns[0][4] if anns and anns[0][4] else ""

    return {
        "company":       company,
        "sector":        sector,
        "announcements": anns,
        "financials":    fins,
        "orders":        orders,
        "prev_signals":  [r[0] for r in prev_signals],
    }


# ── Step 2: Web news (Google News RSS, no API key) ────────────────────────────

def _web_news(symbol: str, company: str) -> list[dict]:
    try:
        query = f"{symbol} {company} NSE BSE India"
        url   = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({
                "q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en",
            })
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())

        results = []
        for item in root.findall(".//item")[:5]:
            title  = item.findtext("title") or ""
            pub    = item.findtext("pubDate") or ""
            src_el = item.find("source")
            src    = src_el.text if src_el is not None else ""
            # Strip Google's " - Source" suffix from titles
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            results.append({"title": title[:160], "published": pub[:30], "source": src})
        return results
    except Exception:
        return []


# ── Step 3: LLM prompt ────────────────────────────────────────────────────────

def _build_prompt(
    symbol: str, company: str, signal_date: str, ctx: dict, news: list,
) -> str:
    parts = [
        "You are a senior Indian equity analyst. Investigate this HIGH VOLUME BREAKOUT signal.",
        f"Stock: {symbol} ({company})  |  Breakout date: {signal_date}",
        "",
        "Write a structured report with EXACTLY these sections:",
        "## Verdict",
        "First line must be: CONFIRMED, PARTIAL, or UNCONFIRMED",
        "Then one sentence stating the main catalyst (or lack of one).",
        "## Key Catalyst",
        "2–3 bullet points on what drove or could drive the move.",
        "## Financial Health",
        "2–3 bullet points on revenue, PAT, and growth trend from recent results.",
        "## Risks",
        "1–2 bullet points on key risks or red flags.",
        "## Signal History",
        "One line: how often does this stock show breakout signals? Reliable or noisy?",
        "",
        "Rules: Be specific. Use ₹ Cr for amounts. No generic filler sentences.",
        "",
        "=== DATA ===",
    ]

    anns = ctx["announcements"]
    if anns:
        parts.append("\nRECENT ANNOUNCEMENTS (last 90 days):")
        for subj, co, dt, oval, sec in anns:
            val = f" — ₹{oval:,.0f} Cr" if oval and oval > 0 else ""
            parts.append(f"  [{str(dt)[:10]}] {subj}{val}")

    orders = ctx["orders"]
    if orders:
        parts.append("\nORDER WINS:")
        for subj, oval, dt in orders:
            val = f" — ₹{oval:,.0f} Cr" if oval and oval > 0 else ""
            parts.append(f"  [{str(dt)[:10]}] {subj}{val}")

    fins = ctx["financials"]
    if fins:
        parts.append("\nFINANCIAL RESULTS (latest quarters):")
        for period, rev, pat, pat_g, rev_g, ebm, dt in fins:
            bits = []
            if rev and rev > 0:        bits.append(f"Rev ₹{rev:,.0f} Cr")
            if pat:                    bits.append(f"PAT ₹{pat:,.0f} Cr")
            if pat_g and pat_g > -900: bits.append(f"PAT {pat_g:+.1f}%")
            if rev_g and rev_g > -900: bits.append(f"Rev {rev_g:+.1f}%")
            if ebm and ebm > 0:        bits.append(f"EBITDA {ebm:.1f}%")
            parts.append(f"  [{period}] " + " | ".join(bits))

    prev = ctx["prev_signals"]
    if len(prev) > 1:
        parts.append(f"\nPREVIOUS BREAKOUT SIGNALS ({len(prev)} total): {', '.join(prev[:6])}")
    elif prev:
        parts.append(f"\nPREVIOUS BREAKOUT SIGNALS: {prev[0]} (first occurrence in DB)")
    else:
        parts.append("\nPREVIOUS BREAKOUT SIGNALS: None in DB")

    if news:
        parts.append("\nWEB NEWS HEADLINES:")
        for n in news:
            src = f"[{n['source']}] " if n['source'] else ""
            parts.append(f"  {src}{n['title']}")

    parts += ["", "=== END DATA ===", "", "Write the investigation report now:"]
    return "\n".join(parts)


# ── Main generator ─────────────────────────────────────────────────────────────

def investigate_stream(
    symbol: str,
    signal_date: str,
    db_path: Path,
    model: str = _OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """Yields JSON-encoded SSE payload strings (without the 'data: ' prefix)."""

    def ev(d: dict) -> str:
        return json.dumps(d)

    # ── Step 1: DB ────────────────────────────────────────────────────────────
    yield ev({"type": "step", "step": "db", "status": "loading",
              "msg": "Searching announcements & financials…"})
    try:
        ctx = _db_context(db_path, symbol, signal_date)
    except Exception as e:
        yield ev({"type": "step", "step": "db", "status": "error", "msg": str(e)})
        yield ev({"type": "done"})
        return

    yield ev({
        "type": "step", "step": "db", "status": "done",
        "msg": (
            f"{len(ctx['announcements'])} announcements · "
            f"{len(ctx['financials'])} results · "
            f"{len(ctx['orders'])} orders"
        ),
        "company": ctx["company"],
    })

    # ── Step 2: Web news ──────────────────────────────────────────────────────
    yield ev({"type": "step", "step": "web", "status": "loading",
              "msg": "Fetching latest headlines…"})
    news = _web_news(symbol, ctx["company"])
    yield ev({
        "type": "step", "step": "web", "status": "done",
        "msg": f"{len(news)} headlines found" if news else "No web results — using DB data only",
    })

    # ── Step 3: LLM ───────────────────────────────────────────────────────────
    yield ev({"type": "step", "step": "llm", "status": "loading",
              "msg": "Starting analysis…"})

    prompt = _build_prompt(symbol, ctx["company"], signal_date, ctx, news)

    try:
        import requests
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.3, "num_predict": 700},
            },
            stream=True,
            timeout=120,
        )
        yield ev({"type": "step", "step": "llm", "status": "streaming", "msg": "Analysing…"})

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield ev({"type": "token", "text": token})
                if chunk.get("done"):
                    break
            except Exception:
                continue

    except Exception as e:
        yield ev({"type": "error", "msg": f"LLM error: {e}"})

    yield ev({"type": "done"})
