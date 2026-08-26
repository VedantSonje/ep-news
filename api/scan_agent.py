"""
Scan Agent — "power prompt" handlers for high-value analytical queries.

Each of the 11 power prompts:
  1. Is detected by regex when the user's message arrives
  2. Runs a focused SQL scan (20–30 rows)
  3. Compresses results to ~2 KB of context text
  4. Streams to the LLM for impact ranking / analysis

Detection happens BEFORE the heavy-query filter in api_server.py so
these queries are answered rather than blocked.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Generator

from api.llm_client import llm_stream, RateLimitError


# ── 11 Power Prompt definitions ───────────────────────────────────────────────

_PP: list[dict] = [
    # 1 ─ Merger & Acquisition
    {
        "id":   "merger_acq",
        "name": "M&A Scan",
        "trigger": re.compile(
            r'\b(merger|acquisition|takeover|amalgam|demerger)\b', re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND (LOWER(subject) LIKE '%merger%'
                OR LOWER(subject) LIKE '%acquisition%'
                OR LOWER(subject) LIKE '%takeover%'
                OR LOWER(subject) LIKE '%amalgam%'
                OR LOWER(subject) LIKE '%demerger%')
            ORDER BY broadcast_dt DESC LIMIT 25
        """,
        "analysis_role": (
            "You are an M&A analyst at an Indian equity research firm. "
            "Below are recent M&A/corporate-restructuring announcements from BSE/NSE (last 30 days). "
            "Rank the top 5 by likely stock price impact. For each: company, what's happening, "
            "is this POSITIVE/NEGATIVE/NEUTRAL for acquirer vs target, and key risk. "
            "Cite rupee values and % stakes where the data provides them."
        ),
    },

    # 2 ─ QIP / Preferential Allotment / Fundraising
    {
        "id":   "qip_fundraise",
        "name": "QIP / Fundraising Scan",
        "trigger": re.compile(
            r'\b(qip|preferential\s+allot|rights\s+issue|fpo|ncd\b|debenture\s+issu)\b', re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND (LOWER(subject) LIKE '%qip%'
                OR LOWER(subject) LIKE '%preferential%'
                OR LOWER(subject) LIKE '%rights issue%'
                OR LOWER(subject) LIKE '% fpo%'
                OR LOWER(subject) LIKE '%ncd%')
            ORDER BY broadcast_dt DESC LIMIT 25
        """,
        "analysis_role": (
            "You are a capital-markets analyst. Below are recent fundraising announcements "
            "(QIP, preferential allotment, rights issues, NCDs) from BSE/NSE (last 30 days). "
            "For each, assess: dilution risk to existing shareholders, likely use of proceeds, "
            "and overall POSITIVE/NEGATIVE/NEUTRAL impact on the stock. Rank the top 5 most significant."
        ),
    },

    # 3 ─ Pharma / USFDA Approvals
    {
        "id":   "pharma_approval",
        "name": "Pharma Regulatory Approvals",
        "trigger": re.compile(
            r'\b(usfda|fda\s+approv|anda\b|drug\s+approv|tentative\s+approv|eir\b|cdsco)\b', re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND (LOWER(subject) LIKE '%usfda%'
                OR LOWER(subject) LIKE '%fda%'
                OR LOWER(subject) LIKE '%anda%'
                OR LOWER(subject) LIKE '%drug%approv%'
                OR LOWER(subject) LIKE '%tentative%approv%'
                OR LOWER(subject) LIKE '%eir%'
                OR LOWER(subject) LIKE '%cdsco%')
            ORDER BY broadcast_dt DESC LIMIT 25
        """,
        "analysis_role": (
            "You are a pharma equity analyst. Below are recent regulatory approval announcements "
            "from Indian pharma/biotech companies (last 30 days). "
            "For each: what drug/product was approved, which market (US/India/EU), "
            "and whether USFDA approvals typically drive 3–10% stock moves. "
            "Rank the top 5 by estimated stock price impact."
        ),
    },

    # 4 ─ Big Order Wins (broad scan, not company-specific)
    {
        "id":   "big_orders",
        "name": "Big Order Wins",
        "trigger": re.compile(
            r'\b(big|large|major|mega|significant)\s+orders?\b'
            r'|\borders?\s+(above|over|greater\s+than)\s*(rs\.?|₹)?\s*\d'
            r'|\ball\s+.*orders?\s+this'
            r'|\btop\s+order\s+wins?\b',
            re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, order_value_cr, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND order_value_cr >= 50
            ORDER BY order_value_cr DESC LIMIT 25
        """,
        "analysis_role": (
            "You are an equity analyst reviewing order wins. Below are recent significant order "
            "wins (Rs.50 Cr+) from BSE/NSE companies (last 30 days), sorted by order value. "
            "For the top 5: client/sector, revenue visibility impact, order-to-revenue ratio if "
            "estimable, and POSITIVE/NEUTRAL signal. Highlight any surprising or unusually large deals."
        ),
    },

    # 5 ─ Defence Sector
    {
        "id":   "defence_orders",
        "name": "Defence Sector Scan",
        "trigger": re.compile(
            r'\bdefence\s+(sector|stocks?|companies|orders?|contracts?)\b'
            r'|\bdefense\s+(sector|orders?)\b'
            r'|\b(drdo|hal\b|bhel.*defence|ordnance\s+factor)',
            re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, order_value_cr, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-60 days')
              AND (LOWER(sector_tags) LIKE '%defence%'
                OR LOWER(subject) LIKE '%defence%'
                OR LOWER(subject) LIKE '%defense%'
                OR LOWER(subject) LIKE '%ministry of defence%'
                OR LOWER(subject) LIKE '%drdo%'
                OR LOWER(subject) LIKE '%indian army%'
                OR LOWER(subject) LIKE '%indian navy%'
                OR LOWER(subject) LIKE '%indian air%')
            ORDER BY COALESCE(order_value_cr, 0) DESC, broadcast_dt DESC LIMIT 25
        """,
        "analysis_role": (
            "You are a defence-sector equity analyst. Below are recent defence-related announcements "
            "from BSE/NSE (last 60 days). For each: what was announced, contract type, and estimated "
            "revenue significance. Rank the top 5 by strategic and financial impact."
        ),
    },

    # 6 ─ Capex / Capacity Expansion
    {
        "id":   "capex_expansion",
        "name": "Capex / Expansion Plans",
        "trigger": re.compile(
            r'\b(capex|capacity\s+expan|new\s+plant|greenfield|brownfield|expansion\s+plan)\b', re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND (LOWER(subject) LIKE '%capex%'
                OR LOWER(subject) LIKE '%capacity%expan%'
                OR LOWER(subject) LIKE '%new%plant%'
                OR LOWER(subject) LIKE '%greenfield%'
                OR LOWER(subject) LIKE '%brownfield%'
                OR LOWER(subject) LIKE '%expansion%')
            ORDER BY broadcast_dt DESC LIMIT 25
        """,
        "analysis_role": (
            "You are a manufacturing-sector analyst. Below are recent capex/expansion announcements "
            "from BSE/NSE companies (last 30 days). For each: investment size, capacity addition "
            "expected, timeline, and long-term revenue impact. Rank the top 5 most significant "
            "expansions and explain why they matter for the stock."
        ),
    },

    # 7 ─ Bonus / Dividend / Stock Split
    {
        "id":   "bonus_dividend",
        "name": "Bonus & Dividend Scan",
        "trigger": re.compile(
            r'\bbonus\s+(issue|share|stock)\b'
            r'|\bdividend\s+(announc|declar|record)\b'
            r'|\bstock\s+split\b'
            r'|\brecord\s+date\b',
            re.I),
        "sql": """
            SELECT symbol, company, subject, broadcast_dt, sector_tags
            FROM announcements
            WHERE DATE(broadcast_dt) >= DATE('now', '-30 days')
              AND (LOWER(subject) LIKE '%bonus%'
                OR LOWER(subject) LIKE '%dividend%'
                OR LOWER(subject) LIKE '%stock split%'
                OR LOWER(subject) LIKE '%record date%')
            ORDER BY broadcast_dt DESC LIMIT 30
        """,
        "analysis_role": (
            "You are an equity analyst covering corporate actions. Below are recent bonus issue, "
            "dividend, and stock split announcements from BSE/NSE (last 30 days). "
            "List all of them with key details (ratio for bonus/split, Rs./share for dividends). "
            "Note any unusually generous corporate actions — these signal financial confidence "
            "and typically trigger a short-term positive sentiment move."
        ),
    },

    # 8 ─ PAT Growth Leaders
    {
        "id":   "pat_leaders",
        "name": "PAT Growth Leaders",
        "trigger": re.compile(
            r'\bpat\s+growth\b|\bprofit\s+growth\b|\bearnings\s+growth\b'
            r'|\bhighest\s+profit\b|\bbest\s+(quarterly\s+)?results?\b',
            re.I),
        "sql": """
            SELECT symbol, company, period, revenue_cr, pat_cr,
                   pat_growth_pct, ebitda_margin_pct, broadcast_dt
            FROM financial_results
            WHERE pat_growth_pct IS NOT NULL
              AND pat_growth_pct > -900
              AND period_type NOT IN
                ('order_win','acquisition','restructuring','credit_rating',
                 'cirp','fundraising','buyback','open_offer')
              AND DATE(broadcast_dt) >= DATE('now', '-90 days')
            ORDER BY pat_growth_pct DESC LIMIT 20
        """,
        "analysis_role": (
            "You are an equity analyst reviewing quarterly earnings. Below are companies ranked "
            "by PAT (net profit) growth YoY from the most recent quarter (last 90 days). "
            "For the top 5: revenue trend, margin quality, and sustainability of growth — "
            "is this genuine operational improvement or a low base effect? "
            "Flag any with very high growth that looks unsustainable."
        ),
    },

    # 9 ─ Revenue Growth Leaders
    {
        "id":   "revenue_leaders",
        "name": "Revenue Growth Leaders",
        "trigger": re.compile(
            r'\brevenue\s+growth\b|\bsales\s+growth\b|\btop.?line\s+growth\b'
            r'|\bhighest\s+revenue\b|\bfastest.grow\b',
            re.I),
        "sql": """
            SELECT symbol, company, period, revenue_cr, pat_cr,
                   revenue_growth_pct, ebitda_margin_pct, broadcast_dt
            FROM financial_results
            WHERE revenue_growth_pct IS NOT NULL
              AND revenue_growth_pct > -900
              AND period_type NOT IN
                ('order_win','acquisition','restructuring','credit_rating',
                 'cirp','fundraising','buyback','open_offer')
              AND DATE(broadcast_dt) >= DATE('now', '-90 days')
            ORDER BY revenue_growth_pct DESC LIMIT 20
        """,
        "analysis_role": (
            "You are an equity analyst. Below are companies ranked by revenue (topline) growth YoY "
            "from the most recent quarter (last 90 days). For the top 5: assess whether growth is "
            "margin-accretive (check EBITDA margin), driven by volume vs price, and if PAT growth "
            "kept pace. Flag any with strong topline but weak profit conversion — that's a quality concern."
        ),
    },

    # 10 ─ High EBITDA Margin Companies
    {
        "id":   "high_margins",
        "name": "High EBITDA Margin Leaders",
        "trigger": re.compile(
            r'\bebitda\s+margin\b|\boperating\s+margin\b'
            r'|\bhigh.?margin\b|\bbest\s+.*margin\b|\bmargin\s+leader',
            re.I),
        "sql": """
            SELECT symbol, company, period, revenue_cr, pat_cr,
                   ebitda_margin_pct, pat_growth_pct, broadcast_dt
            FROM financial_results
            WHERE ebitda_margin_pct IS NOT NULL
              AND ebitda_margin_pct > 0
              AND period_type NOT IN
                ('order_win','acquisition','restructuring','credit_rating',
                 'cirp','fundraising','buyback','open_offer')
              AND DATE(broadcast_dt) >= DATE('now', '-90 days')
            ORDER BY ebitda_margin_pct DESC LIMIT 20
        """,
        "analysis_role": (
            "You are a quality-focused equity analyst. Below are companies with the highest EBITDA "
            "margins from the most recent quarter (last 90 days). For the top 5: margin %, revenue "
            "scale, what drives such margins (pricing power, IP moat, low raw-material cost), "
            "and whether margins improved YoY. These are typically high-quality compounders worth tracking."
        ),
    },

    # 11 ─ Volume Breakout + News Catalyst
    {
        "id":   "breakout_catalyst",
        "name": "Breakout Stocks with Catalyst",
        "trigger": re.compile(
            r'\bbreakout\b.{0,40}\bcatalyst\b'
            r'|\bcatalyst\b.{0,40}\bbreakout\b'
            r'|\bvolume\s+spike\b.{0,30}\breason\b'
            r'|\bwhy\b.{0,20}\bbreaking\s*out\b',
            re.I),
        "sql": """
            SELECT vb.symbol, vb.company, vb.signal_date,
                   vb.close, vb.per_chg, vb.volume,
                   vb.marketcap, vb.sector,
                   a.subject  AS catalyst,
                   a.order_value_cr,
                   DATE(a.broadcast_dt) AS catalyst_dt
            FROM volume_breakouts vb
            LEFT JOIN announcements a
              ON a.symbol = vb.symbol
             AND DATE(a.broadcast_dt)
                 BETWEEN DATE(vb.signal_date, '-5 days')
                     AND DATE(vb.signal_date, '+1 day')
            WHERE DATE(vb.signal_date) >= DATE('now', '-14 days')
            ORDER BY vb.per_chg DESC LIMIT 25
        """,
        "analysis_role": (
            "You are an equity analyst examining volume-breakout stocks. Below are stocks with "
            "significant volume spikes in the last 14 days, paired with any nearby BSE/NSE "
            "announcements as potential catalysts. For each with a catalyst: does the announcement "
            "justify the price move? Rank the top 5 most compelling catalyst-driven breakouts. "
            "Flag any where the move looks technical / noise with no fundamental reason."
        ),
    },
]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_power_prompt(message: str) -> dict | None:
    """Return the first matching power prompt config, or None."""
    for pp in _PP:
        if pp["trigger"].search(message):
            return pp
    return None


def list_power_prompts() -> list[dict]:
    """Return id + name for all 11 prompts (used by /api/scan/prompts)."""
    return [{"id": p["id"], "name": p["name"]} for p in _PP]


def run_power_prompt(
    message: str,
    db_path:  Path,
    pp:       dict,
) -> Generator[str, None, None]:
    """
    Run a power prompt:  SQL scan → compress → LLM analysis stream.
    Yields text tokens; raises RateLimitError if all LLM providers are busy.
    """
    # Run SQL
    try:
        conn    = sqlite3.connect(str(db_path))
        cursor  = conn.execute(pp["sql"])
        col_names = [d[0] for d in cursor.description]
        rows    = cursor.fetchall()
        conn.close()
    except Exception as exc:
        yield f"⚠️ Database error: {exc}"
        return

    if not rows:
        yield (
            f"No results found for **{pp['name']}** in the selected timeframe.\n\n"
            "Our database may not have data for this period yet. "
            "Check back after the next daily sync."
        )
        return

    # Compress each row to a short one-liner (~80 chars)
    lines: list[str] = []
    for row in rows:
        r    = dict(zip(col_names, row))
        parts: list[str] = []

        if r.get("symbol"):    parts.append(r["symbol"])
        if r.get("company"):   parts.append(str(r["company"])[:40])
        if r.get("period"):    parts.append(str(r["period"]))
        if r.get("subject"):   parts.append(str(r["subject"])[:100])
        if r.get("broadcast_dt") or r.get("signal_date"):
            parts.append((r.get("broadcast_dt") or r.get("signal_date") or "")[:10])
        if r.get("catalyst_dt"):   parts.append(str(r["catalyst_dt"])[:10])

        ov = r.get("order_value_cr")
        if ov and float(ov) > 0:    parts.append(f"Order Rs.{float(ov):,.0f}Cr")

        rev = r.get("revenue_cr")
        if rev and float(rev) > 0:  parts.append(f"Rev Rs.{float(rev):,.0f}Cr")

        pat = r.get("pat_cr")
        if pat is not None:          parts.append(f"PAT Rs.{float(pat):,.0f}Cr")

        pg = r.get("pat_growth_pct")
        if pg is not None and float(pg) > -900:
            parts.append(f"PAT {float(pg):+.1f}%")

        rg = r.get("revenue_growth_pct")
        if rg is not None and float(rg) > -900:
            parts.append(f"Rev {float(rg):+.1f}%")

        em = r.get("ebitda_margin_pct")
        if em is not None and float(em) > 0:
            parts.append(f"EBITDA {float(em):.1f}%")

        pc = r.get("per_chg")
        if pc is not None:           parts.append(f"Price {float(pc):+.1f}%")

        cat = r.get("catalyst")
        if cat:                      parts.append(f"[{str(cat)[:70]}]")

        if parts:
            lines.append(" | ".join(parts))

    context = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))

    messages = [
        {"role": "system", "content": pp["analysis_role"]},
        {
            "role": "user",
            "content": (
                f"User query: {message}\n\n"
                f"Database records ({len(rows)} found):\n\n"
                f"{context}\n\n"
                "Provide a clear, structured analysis. "
                "Use Indian number format (Rs. Cr). "
                "If no catalyst/value is shown for a row, note 'value not disclosed'."
            ),
        },
    ]

    try:
        yield from llm_stream(messages)
    except RateLimitError:
        raise
    except Exception as exc:
        yield f"\n\n⚠️ LLM error: {exc}\n\nRaw data:\n{context}"
