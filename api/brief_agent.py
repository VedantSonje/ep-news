"""
Daily Market Brief Agent
Generates a structured markdown brief from the day's announcements + results.
Called by the scheduler after ingestion; stored in daily_briefs table.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

_OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


# ── DB setup ──────────────────────────────────────────────────────────────────

def ensure_table(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_briefs (
            brief_date   TEXT PRIMARY KEY,
            content      TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            ann_count    INTEGER DEFAULT 0,
            highlights   TEXT DEFAULT '[]'
        )
    """)
    conn.commit()
    conn.close()


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch_day_data(conn: sqlite3.Connection, target_date: str) -> dict:
    orders = conn.execute("""
        SELECT ow.symbol, ow.company,
               COALESCE(a.subject, 'Bagging/Receiving of orders/contracts') AS subject,
               ow.order_value_cr,
               a.sector_tags,
               ow.client_name,
               ow.order_sector
        FROM order_wins ow
        LEFT JOIN announcements a ON a.attachment = ow.source_url
        WHERE DATE(ow.broadcast_dt) = ?
        ORDER BY COALESCE(ow.order_value_cr, 0) DESC
        LIMIT 20
    """, (target_date,)).fetchall()

    fins = conn.execute("""
        SELECT symbol, company, period, revenue_cr, pat_cr,
               pat_growth_pct, revenue_growth_pct, ebitda_margin_pct
        FROM financial_results
        WHERE DATE(broadcast_dt) = ?
          AND period_type NOT IN ('order_win','acquisition','restructuring',
                                  'credit_rating','cirp','fundraising','buyback','open_offer',
                                  'summary_only','press_release')
          AND (revenue_cr IS NOT NULL OR pat_cr IS NOT NULL)
          AND NOT (revenue_cr > 100000 AND pat_cr IS NULL)
        ORDER BY COALESCE(revenue_growth_pct, -999) DESC
        LIMIT 20
    """, (target_date,)).fetchall()

    # Other notable announcements by type
    others = conn.execute("""
        SELECT symbol, company, subject, sector_tags
        FROM announcements
        WHERE DATE(broadcast_dt) = ?
          AND LOWER(subject) NOT LIKE '%order%'
        ORDER BY broadcast_dt DESC
        LIMIT 60
    """, (target_date,)).fetchall()

    # Sector activity counts
    sectors = conn.execute("""
        SELECT COALESCE(sector_tags,'Other') as sector, COUNT(*) as n
        FROM announcements
        WHERE DATE(broadcast_dt) = ?
        GROUP BY sector ORDER BY n DESC LIMIT 8
    """, (target_date,)).fetchall()

    total = conn.execute(
        "SELECT COUNT(*) FROM announcements WHERE DATE(broadcast_dt) = ?",
        (target_date,),
    ).fetchone()[0]

    return {
        "orders":  orders,
        "fins":    fins,
        "others":  others,
        "sectors": sectors,
        "total":   total,
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(target_date: str, data: dict) -> str:
    parts = [
        f"You are a senior Indian equity analyst writing the Daily Market Brief for {target_date}.",
        "Write a concise, data-driven brief using ONLY the data below.",
        "Use exactly these markdown sections (omit a section only if no data applies):",
        "",
        "## 🏆 Top Stories",
        "3 bullet points — the most significant events of the day, company + catalyst + number.",
        "",
        "## 📦 Order Wins",
        "One bullet per order win using ONLY symbols from the DATA section below.",
        "Format: **SYMBOL** — [value from DATA e.g. ₹194 Cr or 'value undisclosed'] from [client if known]",
        "Skip this section entirely if DATA shows no order wins.",
        "",
        "## 📊 Results Season",
        "One bullet per result using ONLY symbols from the DATA section below.",
        "Format: **SYMBOL** [period] — Rev ₹[value] Cr | PAT ₹[value] Cr",
        "Include growth % only if it appears in the DATA. Skip this section entirely if DATA shows no results.",
        "",
        "## 💼 Corporate Actions",
        "Notable fundraising, board decisions, acquisitions. 2-4 bullets max. Skip if none.",
        "",
        "## 🏭 Sector Watch",
        "1 line per active sector: which sectors saw the most activity and why.",
        "",
        "CRITICAL RULES:",
        "- ONLY mention companies, symbols, and numbers that appear in the DATA section below.",
        "- DO NOT invent or assume any company names, order values, or financial figures.",
        "- If an order win has no ₹ value in the DATA, write the bullet WITHOUT any ₹ amount.",
        "- If a result has no revenue or PAT in the DATA, write the bullet WITHOUT those figures.",
        "- If a section has no data, write 'No [section name] today.' or omit the section.",
        "- Max 450 words total. No filler sentences.",
        "",
        "=== DATA ===",
    ]

    orders = data["orders"]
    if orders:
        # Deduplicate by symbol — keep highest-value row per symbol
        seen: dict[str, tuple] = {}
        for row in orders:
            sym = row[0]
            oval = row[3]
            if sym not in seen or (oval or 0) > (seen[sym][3] or 0):
                seen[sym] = row
        orders = list(seen.values())
        parts.append(f"\nORDER WINS ({len(orders)} today):")
        for sym, co, subj, oval, sec, client, osec in orders:
            val = f"₹{oval:,.0f} Cr" if oval and oval >= 1 else "value undisclosed"
            sector = osec or sec or ""
            _boilerplate_kw = ("not disclosed", "confidential", "cannot disclose",
                               "commercial issue", "not be disclosed", "withheld",
                               "order(s)/contract(s)")
            _is_boilerplate = not client or any(k in client.lower() for k in _boilerplate_kw)
            client_str = f" | from {client[:60]}" if not _is_boilerplate else ""
            sector_str = f" | {sector}" if sector else ""
            parts.append(f"  {sym} ({co or ''}) — {val}{client_str}{sector_str}")

    fins = data["fins"]
    if fins:
        parts.append(f"\nFINANCIAL RESULTS ({len(fins)} today):")
        for sym, co, period, rev, pat, pat_g, rev_g, ebm in fins:
            bits = []
            if rev and rev > 0:        bits.append(f"Rev ₹{rev:,.0f} Cr")
            if pat:                    bits.append(f"PAT ₹{pat:,.0f} Cr")
            if pat_g and pat_g > -900: bits.append(f"PAT {pat_g:+.1f}%")
            if rev_g and rev_g > -900: bits.append(f"Rev {rev_g:+.1f}%")
            parts.append(f"  {sym} [{period}] — " + " | ".join(bits))

    others = data["others"]
    # Group by broad type
    fundraise = [r for r in others if any(k in (r[2] or '').lower()
                 for k in ('qip','allotment','preferential','fundrais','ncd','debenture'))]
    board_mtg  = [r for r in others if 'board' in (r[2] or '').lower()]
    acq        = [r for r in others if any(k in (r[2] or '').lower()
                 for k in ('acqui','merger','amalgam','takeover'))]

    if fundraise:
        parts.append(f"\nFUNDRAISING ({len(fundraise)}):")
        for r in fundraise[:5]:
            parts.append(f"  {r[0]} — {r[2]}")
    if acq:
        parts.append(f"\nACQUISITIONS/MERGERS ({len(acq)}):")
        for r in acq[:3]:
            parts.append(f"  {r[0]} — {r[2]}")
    if board_mtg:
        parts.append(f"\nBOARD MEETINGS ({len(board_mtg)} outcomes today)")

    sectors = data["sectors"]
    if sectors:
        parts.append("\nSECTOR ACTIVITY:")
        for sec, n in sectors:
            parts.append(f"  {sec}: {n} announcements")

    parts.append(f"\nTOTAL ANNOUNCEMENTS TODAY: {data['total']}")
    parts += ["", "=== END DATA ===", "", "Write the Daily Market Brief now:"]
    return "\n".join(parts)


# ── Highlight extractor ───────────────────────────────────────────────────────

def _extract_highlights(markdown: str) -> list[str]:
    """Pull first 3 bullet points from the Top Stories section."""
    highlights = []
    in_top = False
    for line in markdown.splitlines():
        if "Top Stories" in line or "top stories" in line.lower():
            in_top = True
            continue
        if in_top and line.startswith("#"):
            break
        if in_top:
            stripped = re.sub(r'^[\*\-\•\d\.]\s*', '', line).strip()
            if stripped:
                highlights.append(stripped[:120])
        if len(highlights) >= 3:
            break
    # Fallback: first 3 bullets anywhere
    if not highlights:
        for line in markdown.splitlines():
            stripped = re.sub(r'^[\*\-\•]\s*', '', line).strip()
            if stripped and len(stripped) > 20:
                highlights.append(stripped[:120])
            if len(highlights) >= 3:
                break
    return highlights


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_daily_brief(
    db_path: Path,
    target_date: str | None = None,
    model: str = _OLLAMA_MODEL,
) -> dict | None:
    """
    Generate and store a daily brief.
    Returns the brief dict on success, None if no data or error.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    ensure_table(db_path)
    conn = sqlite3.connect(str(db_path))

    data = _fetch_day_data(conn, target_date)
    if data["total"] == 0:
        conn.close()
        print(f"[brief] No announcements for {target_date} — skipping")
        return None

    print(f"[brief] Generating brief for {target_date} "
          f"({data['total']} ann, {len(data['orders'])} orders, {len(data['fins'])} results)")

    # No financial data → store a plain canned brief, skip LLM entirely
    if not data["orders"] and not data["fins"]:
        content = (
            f"**{target_date} Daily Market Brief**\n\n"
            f"No financial results or order wins filed today "
            f"({data['total']} general announcements received).\n\n"
        )
        if data["others"]:
            content += "## 📋 Other Announcements\n"
            for sym, co, subj, _sec in data["others"][:8]:
                content += f"- **{sym}** — {subj}\n"
        highlights = []
        if data["others"]:
            for sym, co, subj, _sec in data["others"][:3]:
                highlights.append(f"**{sym}** — {subj}")
        now = date.today().isoformat() + "T" + __import__("datetime").datetime.now().strftime("%H:%M:%S")
        conn.execute("""
            INSERT OR REPLACE INTO daily_briefs (brief_date, content, generated_at, ann_count, highlights)
            VALUES (?, ?, ?, ?, ?)
        """, (target_date, content, now, data["total"], json.dumps(highlights)))
        conn.commit()
        conn.close()
        print(f"[brief] Stored canned brief for {target_date} (no orders/results)")
        return {"brief_date": target_date, "content": content, "highlights": highlights}

    prompt = _build_prompt(target_date, data)

    try:
        import requests
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.2, "num_predict": 900},
            },
            timeout=180,
        )
        content = resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[brief] LLM error: {e}")
        conn.close()
        return None

    if not content:
        conn.close()
        return None

    highlights = _extract_highlights(content)
    now        = date.today().isoformat() + "T" + __import__("datetime").datetime.now().strftime("%H:%M:%S")

    conn.execute("""
        INSERT OR REPLACE INTO daily_briefs (brief_date, content, generated_at, ann_count, highlights)
        VALUES (?, ?, ?, ?, ?)
    """, (target_date, content, now, data["total"], json.dumps(highlights)))
    conn.commit()
    conn.close()

    print(f"[brief] Stored brief for {target_date} ({len(content)} chars, {len(highlights)} highlights)")
    return {"date": target_date, "content": content, "highlights": highlights, "ann_count": data["total"]}


def get_latest_brief(db_path: Path, for_date: str | None = None) -> dict | None:
    """Fetch the most recent stored brief (today or yesterday as fallback)."""
    ensure_table(db_path)
    conn = sqlite3.connect(str(db_path))

    if for_date:
        row = conn.execute(
            "SELECT brief_date, content, generated_at, ann_count, highlights FROM daily_briefs WHERE brief_date = ?",
            (for_date,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT brief_date, content, generated_at, ann_count, highlights FROM daily_briefs ORDER BY brief_date DESC LIMIT 1"
        ).fetchone()

    conn.close()
    if not row:
        return None
    return {
        "date":         row[0],
        "content":      row[1],
        "generated_at": row[2],
        "ann_count":    row[3],
        "highlights":   json.loads(row[4] or "[]"),
    }
