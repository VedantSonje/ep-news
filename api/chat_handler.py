"""
ChatHandler — two-stage RAG pipeline:

  Stage 1  Retrieve: keyword intent → metadata filter + ChromaDB cosine search
  Stage 2  Generate: retrieved context + conversation history → Ollama llama3:latest

The retrieval stage is instant (~50ms, no LLM).
The generation stage streams tokens so the UI feels like ChatGPT.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Generator

from financial.vector_store import VectorStore
from retrieval.query_classifier import QueryClassifier, QueryIntent
from retrieval.hybrid_search import CandidateDoc
from retrieval.reranker import Reranker

# ── LangSmith — zero-cost no-op when LANGSMITH_API_KEY is not set ─────────────
import os as _os

def _traceable(**_kw):
    def _inner(fn): return fn
    return _inner

if _os.getenv("LANGSMITH_API_KEY"):
    try:
        from langsmith import traceable as _traceable  # type: ignore[assignment]
        if not _os.getenv("LANGSMITH_PROJECT"):
            _os.environ["LANGSMITH_PROJECT"] = "ep-news-ai"
    except ImportError:
        pass


from api.retrieval_graph import build_retrieval_graph, RetrievalState
from api.llm_client import llm_complete, llm_stream, ACTIVE_PROVIDER, ACTIVE_MODEL

_OLLAMA_MODEL = ACTIVE_MODEL   # kept for LangSmith metadata compat

_SYSTEM_PROMPT = """You are an equity research analyst at an Indian stockbroker.
Answer questions about BSE/NSE corporate announcements using ONLY the data provided in the context below.

Rules:
- Be direct and specific — cite exact company names, rupee values, and dates from the context
- The context has already been filtered — ALL results shown MEET the user's criteria (sector, value threshold, etc.)
- NEVER exclude a result that is clearly marked as meeting the criteria in the context
- When the context says "There are N companies" or "IMPORTANT: include ALL N", you MUST list every single one — no exceptions, no skipping, no summarising
- For numerical comparisons ("above Rs.100 Cr"): if the context shows "Order value: Rs.X Cr" and X > 100, that result IS above 100 Cr — include it
- For comparisons or rankings, calculate directly from the numbers given
- If multiple results, use a structured format: bullet list or markdown table
- If data is missing from context, say "Not available in current results"
- Never invent numbers — use only what the context provides
- Use Indian number format (Rs. Cr, not millions)
- Keep answers concise — 3-10 lines for single-company queries; for list queries include ALL entries from the context
- DATE RANGES: The context header already shows the exact date range of results (e.g. "Announced between 2026-07-01 and 2026-07-31"). Use those exact dates if you need to mention a period — do NOT compute or guess date ranges yourself
- OFF-TOPIC: If the question is completely unrelated to Indian stocks, NSE/BSE, financial results, order wins, or investments, reply ONLY with: "I'm specialized for Indian equity research. Please ask me about NSE/BSE stocks, financial results, order wins, or market announcements." Do NOT answer the off-topic question.
- VAGUE STOCK QUERY: If the question is about stocks but has no specific criterion (e.g. "top 5 stocks", "best stocks", "which stocks to buy", "hot stocks", "good companies"), reply ONLY with:
"To find the right stocks, please tell me what you're looking for:

📦 **Order wins** — e.g. "Defence order wins above Rs.100 Cr this month"
📈 **Financial results** — e.g. "Companies with PAT growth above 20% this quarter"
⚡ **Volume breakouts** — e.g. "Breakout stocks in pharma sector last 14 days"
🏭 **By sector** — e.g. "Top order wins in railways or defence"
📊 **EBITDA / margins** — e.g. "Companies with EBITDA margin above 15%"

Just type your question and I'll search our BSE/NSE database!" Do NOT guess or rank stocks without a clear criterion."""


# ── Intent detection ──────────────────────────────────────────────────────────

_ORDER_KW = {
    "order", "orders", "contract", "contracts", "win", "wins",
    "bagged", "awarded", "l1", "work order", "supply order", "loa",
}
_BREAKOUT_KW = {
    "breakout", "breakouts", "hvy", "volume", "surge", "signal", "signals",
    "5%", "move", "momentum", "spike", "highest volume",
}
_FINANCIAL_KW = {
    "revenue", "profit", "pat", "quarterly", "quarter", "results",
    "financial", "earnings", "ebitda", "margin", "sales", "income",
    "eps", "dividend", "annual", "q1", "q2", "q3", "q4", "fy",
}
_ANN_SUBJECT_MAP = {
    "acquisition":    "Acquisition",
    "acquisitions":   "Acquisition",
    "merger":         "Amalgamation/Merger",
    "amalgamation":   "Amalgamation/Merger",
    "demerger":       "Demerger",
    "insolvency":     "Corporate Insolvency Resolution Process",
    "nclt":           "Corporate Insolvency Resolution Process",
    "cirp":           "Corporate Insolvency Resolution Process",
    "buyback":        "Buyback",
    "qip":            "Qualified Institutional Placement",
    "rights issue":   "Rights Issue",
    "open offer":     "Public Announcement-Open Offer",
    "credit rating":  "Credit Rating- Revision",
    # management / personnel changes
    "management":     "Change in Management",
    "appointment":    "Appointment",
    "appointments":   "Appointment",
    "appointed":      "Appointment",
    "resignation":    "Resignation of Director/KMP/SMP",
    "resigned":       "Resignation of Director/KMP/SMP",
    "director":       "Change in Director(s)",
    "directors":      "Change in Director(s)",
    "ceo":            "Change in Management",
    "cfo":            "Change in Management",
    "cto":            "Change in Management",
    "md":             "Change in Management",
    "kmp":            "Change in Management",
    # fundraising
    "fundraising":    "Fund Raising - Preferential Issue",
    "allotment":      "Allotment of Securities",
    "preferential":   "Fund Raising - Preferential Issue",
    # dividends
    "dividend":       "Dividend",
    "dividends":      "Dividend",
}
_SECTOR_KW = {
    # ── CSV-mapped sectors (used for stock_sectors JOIN in financial queries) ──
    "bank":                   ["bank", "banking", "banks", "psu bank", "private bank"],
    "financials":             ["nbfc", "insurance", "amc", "asset management", "wealth management"],
    "healthcare":             ["healthcare", "pharma", "pharmaceutical", "hospital", "drug",
                               "medicine", "biotech", "diagnostic"],
    "i.t":                    ["information technology", "it sector", "software stocks",
                               "tech stocks", "saas", "it companies"],
    "fmcg":                   ["fmcg", "fast moving", "consumer goods", "fmcg stocks",
                               "hul", "dabur", "nestle", "britannia"],
    "auto":                   ["auto sector", "automobile", "automotive", "two-wheeler",
                               "electric vehicle", "ev sector", "auto stocks"],
    "metals & mining":        ["metals", "mining", "steel", "iron ore", "aluminium",
                               "copper", "zinc", "metal stocks"],
    "energy":                 ["oil", "gas", "petroleum", "refinery", "ongc",
                               "energy sector", "crude"],
    "chemicals":              ["chemical", "chemicals", "specialty chemicals",
                               "pesticide", "agrochemical", "fertilizer"],
    "realty":                 ["realty", "real estate", "property", "builder",
                               "housing", "developer", "real-estate"],
    "telecom":                ["telecom", "5g", "fibre", "fiber", "broadband", "telecom stocks"],
    "industrials":            ["industrial", "industrials", "manufacturing", "engineering",
                               "capital goods", "heavy industry"],
    "consumer discretionary": ["consumer discretionary", "retail", "hotel", "restaurant",
                               "apparel", "fashion", "discretionary"],
    "aerospace & defence":    ["aerospace", "defence", "defense", "military", "army",
                               "navy", "hal", "bel", "drdo"],
    "textiles":               ["textile", "textiles", "garment", "clothing", "yarn", "fabric"],
    "transportation":         ["transport", "logistics", "shipping", "aviation",
                               "airline", "port", "freight"],
    "media":                  ["media", "entertainment", "film", "television", "ott",
                               "broadcasting"],
    "power & utilities":      ["power sector", "electricity", "utility", "solar energy",
                               "wind energy", "ntpc", "renewable energy"],
    "building materials":     ["cement", "tiles", "paint", "glass", "sanitaryware",
                               "building material", "plywood"],
    # ── Order-win sector keywords (used for semantic context only) ──
    "railways":               ["railway", "railways", "rail", "metro", "rvnl", "ircon"],
    "roads":                  ["road", "roads", "highway", "nhai", "expressway", "bridge"],
    "water":                  ["water", "irrigation", "sewage", "pipeline"],
}

_VALUE_RE = re.compile(
    r"(?:above|over|more\s+than|greater\s+than|>|≥|atleast|at\s+least)\s*"
    r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d,]+(?:\.\d+)?)\s*"
    r"(crore|cr\.?|lakh|lakhs|million|mn)?",
    re.IGNORECASE,
)


def _detect_intent(text: str) -> str:
    # Split on ALL non-alphanumeric chars so "financial/earnings" → {"financial","earnings"}
    tl   = set(re.split(r'\W+', text.lower()))
    full = text.lower()
    if tl & _BREAKOUT_KW and not (tl & _ORDER_KW) and not (tl & _FINANCIAL_KW):
        return "volume_breakout"
    if tl & _ORDER_KW or any(k in full for k in ["work order", "order win"]):
        if not (tl & _FINANCIAL_KW):
            return "order_win"
    if tl & _FINANCIAL_KW and not (tl & _ORDER_KW):
        return "financials"
    if any(k in full for k in _ANN_SUBJECT_MAP) and not (tl & _ORDER_KW) and not (tl & _FINANCIAL_KW):
        return "announcements"
    return "all"


_ANNUAL_KW  = {"annual", "yearly", "fy2026", "fy2025", "fy2027", "full year", "full-year"}
_QUARTER_KW = {"quarterly", "quarter", "q1", "q2", "q3", "q4"}


def _extract_period_type(text: str) -> str | None:
    """Return 'annual' or 'quarterly' if the query explicitly requests one, else None."""
    tl = set(re.split(r'\W+', text.lower()))
    if tl & _ANNUAL_KW or "full year" in text.lower():
        return "annual"
    if tl & _QUARTER_KW:
        return "quarterly"
    return None


def _extract_min_cr(text: str) -> float | None:
    m = _VALUE_RE.search(text)
    if not m:
        return None
    val  = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "cr").lower().strip(".")
    if "lakh" in unit:
        val /= 100
    elif "million" in unit or unit == "mn":
        val /= 10
    return val


_SPECIFIC_DATE_RE = re.compile(
    r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'
    r'|\b(\d{4})-(\d{2})-(\d{2})\b'
    r'|\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?'
    r'|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'\s+(\d{4})\b',
    re.IGNORECASE,
)
_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


_TODAY_RE       = re.compile(r'\btoday\b', re.I)
_YESTERDAY_RE   = re.compile(r'\by\w{4,10}day\b', re.I)   # catches "yesterday" + typos
_THIS_WEEK_RE   = re.compile(r'\b(?:this|current)\s*week\b', re.I)
_LAST_WEEK_RE   = re.compile(r'\blast\s*week\b', re.I)
_THIS_MONTH_RE  = re.compile(r'\b(?:this|current)\s*month\b', re.I)
_LAST_MONTH_RE  = re.compile(r'\blast\s*month\b', re.I)    # matches "last month" and "lastmonth"
_THIS_YEAR_RE   = re.compile(r'\b(?:this|current)\s*year\b', re.I)
_LAST_YEAR_RE   = re.compile(r'\blast\s*year\b', re.I)
_LAST_N_DAYS_RE = re.compile(r'\blast\s+(\d+)\s+days?\b', re.I)


def _extract_date_range(text: str) -> tuple[date | None, date | None]:
    """Detect temporal references and return (from_date, to_date). Returns (None, None) if no date found."""
    today = date.today()  # noqa: used below

    if _TODAY_RE.search(text):
        return today, today
    if _YESTERDAY_RE.search(text):
        d = today - timedelta(days=1)
        return d, d
    if _THIS_WEEK_RE.search(text):
        start = today - timedelta(days=today.weekday())
        return start, today
    if _LAST_WEEK_RE.search(text):
        end = today - timedelta(days=today.weekday() + 1)
        return end - timedelta(days=6), end
    if _THIS_MONTH_RE.search(text):
        return today.replace(day=1), today
    if _LAST_MONTH_RE.search(text):
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end
    if _THIS_YEAR_RE.search(text):
        return today.replace(month=1, day=1), today
    if _LAST_YEAR_RE.search(text):
        return today.replace(year=today.year - 1, month=1, day=1), \
               today.replace(year=today.year - 1, month=12, day=31)

    # "last N days"
    m = _LAST_N_DAYS_RE.search(text)
    if m:
        return today - timedelta(days=int(m.group(1))), today

    # Specific date in text (e.g. "26 july", "july 26", "26-07-2026")
    m = _SPECIFIC_DATE_RE.search(text)
    if m:
        try:
            g = m.groups()
            if g[6]:  # day month_word year
                day, mon_str, yr = int(g[6]), g[7][:3].lower(), int(g[8])
                mon = _MONTH_MAP.get(mon_str)
                if mon:
                    d = date(yr, mon, day)
                    return d, d
            elif g[3]:  # YYYY-MM-DD
                d = date(int(g[3]), int(g[4]), int(g[5]))
                return d, d
            elif g[0]:  # DD/MM/YYYY
                d = date(int(g[2]), int(g[1]), int(g[0]))
                return d, d
        except (ValueError, TypeError):
            pass

    return None, None


def _extract_sector(text: str) -> str | None:
    tl = text.lower()
    for sec, kws in _SECTOR_KW.items():
        if any(k in tl for k in kws):
            return sec
    return None


# ── Citation grounding (Loop 2) ───────────────────────────────────────────────

_GROUNDING_SKIP = frozenset({
    "I", "A", "AT", "IN", "OF", "OR", "AND", "THE", "FOR", "BY", "PAT",
    "EPS", "FY", "Q1", "Q2", "Q3", "Q4", "YOY", "QOQ", "RS", "CR", "LTD",
    "INC", "PLC", "CO", "EBITDA", "CAGR", "NSE", "BSE", "IPO", "NPA",
    "AUM", "ROE", "ROA", "PE", "PB", "NA", "NO", "ALL", "TOP", "EP",
    "MF", "ETF", "NFO", "AGM", "EGM", "OFS", "QIP", "FPO",
})
_RS_VAL_RE         = re.compile(r'Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|Crore)\b', re.I)
_SYM_IN_ANSWER_RE  = re.compile(r'\b([A-Z]{3,10})\b')


def _check_grounding(
    answer: str,
    results: list[dict],
    known_symbols: set[str],
) -> list[str]:
    """
    Returns up to 3 unverified claims in the LLM answer — Rs.X Cr values or
    company symbols that are absent from the retrieved source context.
    """
    if not answer or not results:
        return []

    # Collect all numeric values and symbols present in retrieved context
    context_values: list[float] = []
    context_symbols: set[str] = set()
    for r in results:
        sym = (r.get("symbol") or "").upper()
        if sym:
            context_symbols.add(sym)
        for field in ("revenue_cr", "pat_cr", "order_value_cr", "ebitda_cr", "eps"):
            v = r.get(field)
            if v:
                try:
                    fv = float(v)
                    if fv > 0:
                        context_values.append(fv)
                except (ValueError, TypeError):
                    pass
        # Also parse Rs. values embedded in snippet text
        for m in _RS_VAL_RE.finditer(r.get("_snippet", "") or ""):
            try:
                context_values.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass

    # Threshold keywords — when "above Rs.100 Cr" appears, Rs.100 is not a data value
    _THRESHOLD_RE = re.compile(
        r'\b(?:above|over|below|under|more\s+than|less\s+than|exceeding|minimum|minimum\s+of|at\s+least|upto|up\s+to|greater\s+than|smaller\s+than)\s+Rs',
        re.I,
    )
    threshold_positions: set[int] = set()
    for tm in _THRESHOLD_RE.finditer(answer):
        threshold_positions.add(tm.end())  # position right before "Rs."

    unverified: list[str] = []

    # Check every Rs.X Cr value cited in the answer
    for m in _RS_VAL_RE.finditer(answer):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if val < 1:   # skip sub-crore (EPS, dividends, small percentages)
            continue
        # Skip values used as comparison thresholds ("above Rs.100 Cr"), not data points
        if any(abs(m.start() - pos) <= 5 for pos in threshold_positions):
            continue
        matched = any(abs(val - cv) / max(cv, 1.0) < 0.06 for cv in context_values)
        label = f"Rs.{val:,.0f} Cr"
        if not matched and label not in unverified:
            unverified.append(label)

    # Check company symbols cited in answer that weren't in retrieved context
    for m in _SYM_IN_ANSWER_RE.finditer(answer):
        sym = m.group(1)
        if (sym not in _GROUNDING_SKIP
                and sym in known_symbols
                and sym not in context_symbols
                and sym not in unverified):
            unverified.append(sym)

    return unverified[:3]


# ── Company name → symbol expansion ──────────────────────────────────────────

_NAME_STOPWORDS = frozenset({
    "limited", "ltd", "pvt", "private", "india", "industries", "industry",
    "solutions", "services", "enterprises", "holdings", "group", "corp",
    "technologies", "technology", "corporation", "company", "international",
    "infrastructure", "finance", "financial", "capital", "asset", "assets",
    "management", "power", "energy", "trading", "investment", "investments",
    "chemicals", "chemical", "products", "product", "systems", "system",
})


def _build_name_to_symbol(conn: sqlite3.Connection) -> dict[str, str]:
    """
    Build word → symbol index from company names in DB.
    Only stores unambiguous (one word → one symbol) mappings, used as a
    second-pass lookup when exact symbol matching fails in a query.
    """
    word_to_syms: dict[str, list[str]] = {}
    rows = conn.execute(
        "SELECT DISTINCT symbol, company FROM announcements WHERE company IS NOT NULL"
    ).fetchall()

    for sym, company in rows:
        if not sym or not company:
            continue
        name = company.lower()
        for suffix in (" limited", " ltd.", " ltd", " pvt ltd", " pvt. ltd"):
            name = name.replace(suffix, "")
        name = name.strip()

        # Index each significant word
        for word in re.split(r'[^a-z]+', name):
            if len(word) >= 4 and word not in _NAME_STOPWORDS:
                word_to_syms.setdefault(word, [])
                if sym not in word_to_syms[word]:
                    word_to_syms[word].append(sym)

        # Index the full stripped name too
        if len(name) >= 4:
            word_to_syms.setdefault(name, [])
            if sym not in word_to_syms[name]:
                word_to_syms[name].append(sym)

    # Keep only unambiguous (1:1) word→symbol mappings
    return {w: syms[0] for w, syms in word_to_syms.items() if len(syms) == 1}


def _extract_subject(text: str) -> str | None:
    tl = text.lower()
    for kw, subj in _ANN_SUBJECT_MAP.items():
        if kw in tl:
            return subj
    return None


# ── Reranker helpers ──────────────────────────────────────────────────────────

def _docs_to_candidates(results: list[dict]) -> list[CandidateDoc]:
    """Convert VectorStore result dicts to CandidateDoc for cross-encoder reranking."""
    out: list[CandidateDoc] = []
    for i, r in enumerate(results):
        doc = CandidateDoc(
            id=r.get("_id", f"{r.get('symbol', '')}_{i}"),
            symbol=r.get("symbol", ""),
            company=r.get("company", ""),
            subject=r.get("subject", "") or r.get("period", ""),
            details=r.get("_snippet", "") or "",
            score=int(r.get("score") or 0),
            broadcast_dt=str(r.get("broadcast_dt", ""))[:10],
            rrf_score=float(r.get("_rrf_score") or 0.0),
            sources=["bm25", "vector"],   # VectorStore always fuses both
        )
        doc.evidence["order_value_cr"] = r.get("order_value_cr")
        doc.evidence["sector_tags"]    = r.get("sector") or r.get("sector_tags") or ""
        doc.evidence["_orig"]          = r
        out.append(doc)
    return out


def _candidates_to_dicts(ranked: list[CandidateDoc]) -> list[dict]:
    """Recover original VectorStore dicts from reranked CandidateDocs."""
    return [
        doc.evidence.get("_orig", {
            "symbol": doc.symbol, "company": doc.company,
            "subject": doc.subject, "broadcast_dt": doc.broadcast_dt,
            "_snippet": doc.details,
        })
        for doc in ranked
    ]


# ── Direct list formatter (bypasses LLM for pure listing queries) ─────────────

_LIST_QUERY_RE = re.compile(
    r'\b(all|full|complete|every|list\s+all|give\s+(me\s+)?all|show\s+(me\s+)?all'
    r'|give\s+(me\s+)?a\s+list|list\s+of|show\s+(me\s+)?a\s+list)\b',
    re.I,
)

def _is_listing_query(message: str) -> bool:
    """True when the user wants every result listed, not just a summary or analysis."""
    return bool(_LIST_QUERY_RE.search(message))


def _format_direct_list(results: list[dict], intent: str, min_cr: float | None = None) -> str:
    """
    Build a complete, deterministic formatted answer directly from results.
    Called for listing queries so the LLM never gets to pick which items to include.
    """
    if not results:
        return "No matching records found in the database."

    order_dates = sorted({str(r.get("broadcast_dt", ""))[:10] for r in results if r.get("broadcast_dt")})
    date_range = (
        f"{order_dates[0]} to {order_dates[-1]}" if len(order_dates) >= 2
        else (order_dates[0] if order_dates else "")
    )

    header = f"Companies that received orders ({date_range}):\n"
    if min_cr and min_cr > 0:
        header = f"Companies that received orders ≥ Rs.{min_cr:,.0f} Cr ({date_range}):\n"

    rows = []
    for i, r in enumerate(results, 1):
        sym  = r.get("symbol", "")
        co   = r.get("company", "")
        ov   = r.get("order_value_cr")
        dt   = str(r.get("broadcast_dt", ""))[:10]
        val  = f"Rs.{ov:,.2f} Cr" if ov and ov > 0 else "value not disclosed"
        rows.append(f"{i:>2}. {sym} — {co}  |  {val}  |  {dt}")

    return header + "\n".join(rows)


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(results: list[dict], intent: str, min_cr: float | None = None) -> str:
    if not results:
        return "No matching records found in the database."

    lines: list[str] = []

    if intent == "order_win":
        order_dates = sorted({str(r.get("broadcast_dt", ""))[:10] for r in results if r.get("broadcast_dt")})
        date_note = (
            f" | Announced between {order_dates[0]} and {order_dates[-1]}" if len(order_dates) >= 2
            else (f" | Announced on {order_dates[0]}" if order_dates else "")
        )
        header = f"ORDER WIN ANNOUNCEMENTS (BSE/NSE database){date_note}"
        if min_cr and min_cr > 0:
            header += f" — pre-filtered: ALL results below have order value ≥ Rs.{min_cr:,.0f} Cr"
        lines.append(header + ":")
        lines.append("NOTE: 'Date' below is the BSE/NSE announcement date. Use these exact dates in your answer.")
        if len(results) > 5:
            lines.append(f"IMPORTANT: There are {len(results)} companies below. You MUST include ALL {len(results)} of them in your answer — do NOT skip or omit any entry.")
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('symbol')} — {r.get('company')}")
            ov  = r.get("order_value_cr", -1)
            sec = r.get("sector", "")
            ct  = r.get("client_type", "")
            if ov and ov > 0:
                meets = f" ✓ above Rs.{min_cr:,.0f} Cr" if (min_cr and ov >= min_cr) else ""
                lines.append(f"   Order value : Rs.{ov:,.2f} Cr{meets}")
            if sec:
                lines.append(f"   Sector      : {sec}")
            if ct:
                lines.append(f"   Client type : {ct}")
            lines.append(f"   Date        : {str(r.get('broadcast_dt',''))[:10]}")
            snip = r.get("_snippet", "")
            if snip and len(results) <= 15:
                lines.append(f"   Detail      : {snip[:150]}")

    elif intent == "financials":
        ann_dates = sorted(set(str(r.get("broadcast_dt",""))[:10] for r in results if r.get("broadcast_dt")))
        date_note = f" | Announced between {ann_dates[0]} and {ann_dates[-1]}" if len(ann_dates) >= 2 else (f" | Announced on {ann_dates[0]}" if ann_dates else "")
        lines.append(f"FINANCIAL RESULTS (BSE/NSE database){date_note}:")
        lines.append("NOTE: 'Date' below is the BSE/NSE announcement date, NOT the financial period end date.")
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('symbol')} — {r.get('company')} [{r.get('period')}]")
            rev = r.get("revenue_cr", -1)
            pat = r.get("pat_cr", -1)
            rg  = r.get("revenue_growth_pct", -999)
            pg  = r.get("pat_growth_pct", -999)
            ebm = r.get("ebitda_margin_pct", -1)
            eps = r.get("eps", -1)
            if rev and rev > 0:
                rg_s = f" ({rg:+.1f}% YoY)" if rg and rg > -900 else ""
                lines.append(f"   Revenue      : Rs.{rev:,.2f} Cr{rg_s}")
            if pat and pat > 0:
                pg_s = f" ({pg:+.1f}% YoY)" if pg and pg > -900 else ""
                lines.append(f"   PAT          : Rs.{pat:,.2f} Cr{pg_s}")
            if ebm and ebm > 0:
                lines.append(f"   EBITDA margin: {ebm:.1f}%")
            if eps and eps > 0:
                lines.append(f"   EPS          : Rs.{eps:.2f}")
            lines.append(f"   Date         : {str(r.get('broadcast_dt',''))[:10]}")
            snip = r.get("_snippet", "")
            if snip and len(results) <= 15:
                after_colon = snip[snip.find(":")+1:].strip()
                if len(after_colon) > 20:
                    lines.append(f"   Summary      : {after_colon[:150]}")

    elif intent == "multi_hop":
        cond_labels = results[0].get("_conditions", []) if results else []
        lines.append(
            f"MULTI-HOP SCREENER — {len(results)} stocks matching ALL conditions: "
            + " + ".join(f"[{c}]" for c in cond_labels)
        )
        lines.append("IMPORTANT: List every stock below. Each one satisfies ALL conditions.")
        lines.append("")
        for r in results:
            lines.append(f"Stock: {r['symbol']} ({r.get('company', '')})")
            for cond in cond_labels:
                data = r.get(f"_data_{cond}", {})
                if cond == "breakout":
                    lines.append(f"  Breakout signal : {data.get('signal_date','?')} | {data.get('sector','')} | {data.get('marketcap','')}")
                elif cond == "order":
                    oval = data.get("order_value_cr")
                    val  = f" Rs.{oval:,.0f} Cr" if oval and oval > 0 else ""
                    lines.append(f"  Order win       :{val} on {str(data.get('broadcast_dt',''))[:10]}")
                elif cond == "financial":
                    rev  = data.get("revenue_cr")
                    rg   = data.get("revenue_growth_pct")
                    pat  = data.get("pat_cr")
                    pg   = data.get("pat_growth_pct")
                    bits = []
                    if rev and rev > 0:       bits.append(f"Rev Rs.{rev:,.0f}Cr")
                    if rg and rg > -900:      bits.append(f"Rev {rg:+.1f}%")
                    if pat:                   bits.append(f"PAT Rs.{pat:,.0f}Cr")
                    if pg and pg > -900:      bits.append(f"PAT {pg:+.1f}%")
                    lines.append(f"  Financials      : " + " | ".join(bits) + f" [{data.get('period','')}]")
            lines.append("")

    elif intent == "volume_breakout":
        sectors  = sorted({r.get("sector", "") for r in results if r.get("sector")})
        caps     = sorted({r.get("marketcap", "") for r in results if r.get("marketcap")})
        dates    = sorted({str(r.get("signal_date", ""))[:10] for r in results if r.get("signal_date")})
        date_rng = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "")
        lines.append(f"VOLUME BREAKOUT SIGNALS — {len(results)} stocks | {date_rng}")
        lines.append(f"Sectors: {', '.join(sectors[:8])} | Caps: {', '.join(caps)}")
        lines.append("")
        for r in results:
            lines.append(
                f"  {r['symbol']:14} {r.get('marketcap',''):10} {r.get('sector',''):30} {str(r.get('signal_date',''))[:10]}"
            )

    else:  # announcements or all
        ann_dates2 = sorted({str(r.get("broadcast_dt", ""))[:10] for r in results if r.get("broadcast_dt")})
        date_note2 = (
            f" | Announced between {ann_dates2[0]} and {ann_dates2[-1]}" if len(ann_dates2) >= 2
            else (f" | Announced on {ann_dates2[0]}" if ann_dates2 else "")
        )
        label = f"ANNOUNCEMENTS (BSE/NSE database){date_note2}:"
        lines.append(label)
        lines.append("NOTE: Use these exact announcement dates in your answer.")
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('symbol')} — {r.get('company')}")
            lines.append(f"   Subject : {r.get('subject','')}")
            lines.append(f"   Date    : {str(r.get('broadcast_dt',''))[:10]}")
            sec = r.get("sector_tags", "")
            if sec:
                lines.append(f"   Sectors : {sec}")
            ov = r.get("order_value_cr", -1)
            if ov and ov > 0:
                lines.append(f"   Value   : Rs.{ov:,.2f} Cr")
            snip = r.get("_snippet", "")
            if snip:
                after = snip[snip.find(".")+1:].strip()
                if len(after) > 20:
                    lines.append(f"   Detail  : {after[:250]}")

    return "\n".join(lines)


# ── ChatHandler ───────────────────────────────────────────────────────────────

class ChatHandler:

    _SUGGESTIONS = [
        "Defence order wins above Rs.100 Cr",
        "Companies with revenue growth this quarter",
        "Recent acquisitions in pharma sector",
        "NCLT insolvency cases",
        "Railway infrastructure orders above Rs.200 Cr",
        "Buyback announcements",
        "Compare EBITDA margins across results",
        "Which company got the largest order this month?",
    ]

    def __init__(self, chroma_path: Path | str, db_path: Path | str) -> None:
        self._store      = VectorStore(chroma_path)
        self._db_path    = Path(db_path)
        self._classifier = QueryClassifier()
        self._reranker   = Reranker()   # warms up CrossEncoder on startup
        # Cache known symbols + company-name → symbol index for query expansion
        _conn = sqlite3.connect(str(self._db_path))
        self._known_symbols: set[str] = {
            r[0] for r in _conn.execute("SELECT DISTINCT symbol FROM announcements").fetchall()
        }
        self._name_to_symbol: dict[str, str] = _build_name_to_symbol(_conn)
        _conn.close()
        self._graph = build_retrieval_graph(self)

    def _extract_company_symbol(self, text: str) -> str | None:
        """
        Return the best matching stock symbol for a company reference in the query.

        Pass 1 — exact symbol match: looks for uppercase tokens that are known symbols.
        Pass 2 — name expansion: matches query words against company name word index
                  (e.g. "infosys" → INFY, "larsen" → LT, "nestle" → NESTLEIND).
                  Tries bigrams before unigrams to prefer more specific matches.
        """
        # Pass 1: exact uppercase token match
        for token in re.split(r'\W+', text):
            if token.upper() in self._known_symbols:
                return token.upper()

        # Pass 2: company name word expansion
        # Try n-grams longest-first so "indian renewable energy development agency"
        # matches before any shorter ambiguous sub-phrase.
        words = [w for w in re.split(r'\W+', text.lower()) if w]
        max_n = min(len(words), 6)
        for n in range(max_n, 1, -1):          # 6-gram down to bigram
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if ngram in self._name_to_symbol:
                    return self._name_to_symbol[ngram]
        # Unigrams last
        for word in words:
            if len(word) >= 4 and word in self._name_to_symbol:
                return self._name_to_symbol[word]
        return None

    # ── SQL financial retrieval ───────────────────────────────────────────────

    def _sql_financials_by_symbol(self, symbol: str, n: int = 10) -> list[dict]:
        """Return all financial results for a single company, most recent first."""
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute("""
            SELECT symbol, company, period, period_type,
                   revenue_cr, revenue_growth_pct, pat_cr, pat_growth_pct,
                   ebitda_margin_pct, eps, key_highlights, raw_summary, broadcast_dt,
                   source_url
            FROM financial_results
            WHERE symbol = ?
              AND period_type NOT IN ('credit_rating','cirp','fundraising','buyback','open_offer','restructuring')
            ORDER BY broadcast_dt DESC
            LIMIT ?
        """, (symbol, n)).fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({
                "symbol": r[0], "company": r[1] or "",
                "period": r[2] or "", "period_type": r[3] or "",
                "revenue_cr": r[4], "revenue_growth_pct": r[5],
                "pat_cr": r[6], "pat_growth_pct": r[7],
                "ebitda_margin_pct": r[8], "eps": r[9],
                "_snippet": (r[11] or "")[:300],
                "broadcast_dt": (r[12] or "")[:10],
                "source_url": r[13] or "",
            })
        return results

    def _sql_financials_by_date(
        self,
        from_dt:     date,
        to_dt:       date,
        n:           int = 100,
        period_type: str | None = None,
        symbol:      str | None = None,
        sector:      str | None = None,
    ) -> list[dict]:
        """Query financial_results directly by date range — returns ALL companies, not just top-n."""
        conn = sqlite3.connect(str(self._db_path))
        join = ""
        clauses = [
            "DATE(fr.broadcast_dt) BETWEEN ? AND ?",
            "fr.period_type NOT IN ('credit_rating','cirp','fundraising','buyback','open_offer','restructuring')",
        ]
        params: list = [from_dt.isoformat(), to_dt.isoformat()]
        if sector:
            join = "JOIN stock_sectors ss ON fr.symbol = ss.symbol"
            if sector == "telecom":
                clauses.append("ss.sector IN ('telecom', 'telecom-service')")
            else:
                clauses.append("ss.sector = ?")
                params.append(sector)
        if period_type == "annual":
            clauses.append("fr.period_type = 'annual'")
        elif period_type == "quarterly":
            clauses.append("fr.period_type IN ('quarterly','financial_results')")
        if symbol:
            clauses.append("fr.symbol = ?")
            params.append(symbol.upper())
        params.append(n)
        sql = f"""
            SELECT fr.symbol, fr.company, fr.period, fr.period_type,
                   fr.revenue_cr, fr.pat_cr, fr.revenue_growth_pct, fr.pat_growth_pct,
                   fr.ebitda_margin_pct, fr.eps, fr.raw_summary, fr.broadcast_dt
            FROM financial_results fr {join}
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(fr.revenue_cr, -1) DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "", "period": r[2] or "",
                "period_type": r[3] or "",
                "revenue_cr": r[4], "pat_cr": r[5],
                "revenue_growth_pct": r[6], "pat_growth_pct": r[7],
                "ebitda_margin_pct": r[8], "eps": r[9],
                "_snippet": r[10] or "", "broadcast_dt": (r[11] or "")[:10],
            }
            for r in rows
        ]

    def _sql_financials_by_sector(self, sector: str, n: int = 30) -> list[dict]:
        """Return the most recent quarterly result per company in the given sector."""
        conn = sqlite3.connect(str(self._db_path))
        sec_filter = "ss.sector IN ('telecom', 'telecom-service')" if sector == "telecom" \
                     else "ss.sector = ?"
        params: list = [] if sector == "telecom" else [sector]
        params.append(n)
        sql = f"""
            SELECT sub.symbol, sub.company, sub.period, sub.period_type,
                   sub.revenue_cr, sub.pat_cr, sub.revenue_growth_pct, sub.pat_growth_pct,
                   sub.ebitda_margin_pct, sub.eps, sub.raw_summary, sub.broadcast_dt
            FROM (
                SELECT fr.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY fr.symbol
                           ORDER BY fr.broadcast_dt DESC
                       ) AS rn
                FROM financial_results fr
                JOIN stock_sectors ss ON fr.symbol = ss.symbol
                WHERE {sec_filter}
                  AND fr.period_type NOT IN (
                      'credit_rating','cirp','fundraising','buyback','open_offer','restructuring'
                  )
            ) sub
            WHERE sub.rn = 1
            ORDER BY COALESCE(sub.revenue_cr, 0) DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "", "period": r[2] or "",
                "period_type": r[3] or "",
                "revenue_cr": r[4], "pat_cr": r[5],
                "revenue_growth_pct": r[6], "pat_growth_pct": r[7],
                "ebitda_margin_pct": r[8], "eps": r[9],
                "_snippet": r[10] or "", "broadcast_dt": (r[11] or "")[:10],
            }
            for r in rows
        ]

    # Subjects that represent genuine order announcements — any value extracted
    # from these PDFs is trusted regardless of size.
    def _sql_order_wins_by_date(
        self, from_dt: date, to_dt: date, n: int = 100, min_cr: float | None = None,
        sector: str | None = None,
    ) -> list[dict]:
        """Query order_wins table by date range — optionally filtered by company sector."""
        conn = sqlite3.connect(str(self._db_path))
        params: list = [from_dt.isoformat(), to_dt.isoformat()]
        join = ""
        sector_filter = ""
        _ORDER_WIN_ONLY_SECTORS = {"railways", "roads", "water"}
        if sector and sector not in _ORDER_WIN_ONLY_SECTORS:
            join = "JOIN stock_sectors ss ON ow.symbol = ss.symbol"
            if sector == "telecom":
                sector_filter = "AND ss.sector IN ('telecom', 'telecom-service')"
            else:
                sector_filter = "AND ss.sector = ?"
                params.append(sector)
        min_filter = ""
        if min_cr and min_cr > 0:
            # Strict: require a disclosed value that actually meets the threshold.
            # The old IS NULL clause caused every undisclosed-value order to pass,
            # flooding results with irrelevant PDFs as sources.
            min_filter = "AND ow.order_value_cr >= ?"
            params.append(min_cr)
        rows = conn.execute(f"""
            SELECT ow.symbol, ow.company, ow.order_value_cr, ow.client_name, ow.client_type,
                   ow.order_sector, ow.execution_months, ow.raw_summary, ow.broadcast_dt,
                   ow.from_genuine_order, ow.description, ow.source_url
            FROM order_wins ow {join}
            WHERE DATE(ow.broadcast_dt) BETWEEN ? AND ?
              {sector_filter}
              {min_filter}
            ORDER BY DATE(ow.broadcast_dt) DESC, ow.order_value_cr DESC NULLS LAST
        """, params).fetchall()
        conn.close()
        seen: set[str] = set()
        results = []
        for r in rows:
            sym = r[0]
            if sym in seen:
                continue
            seen.add(sym)
            order_val = r[2]
            # Discard suspiciously large values from non-genuine sources (press releases,
            # annual reports) — they are usually market sizes or revenue figures, not orders.
            if order_val and not r[9] and order_val > 500:
                order_val = None
            snippet = r[10] or r[7] or ""
            results.append({
                "symbol":          sym,
                "company":         r[1] or "",
                "period_type":     "order_win",
                "order_value_cr":  order_val,
                "client_name":     r[3] or "",
                "client_type":     r[4] or "",
                "order_sector":    r[5] or "",
                "execution_months": r[6],
                "_snippet":        snippet[:300],
                "broadcast_dt":    (r[8] or "")[:10],
                "source_url":      r[11] or "",
            })
        results.sort(key=lambda x: x["order_value_cr"] or 0, reverse=True)
        return results[:n]

    def _sql_order_wins_by_symbol(self, symbol: str, n: int = 10) -> list[dict]:
        """Return order wins for a specific company, most recent first."""
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute("""
            SELECT symbol, company, broadcast_dt, order_value_cr,
                   client_name, client_type, order_sector, raw_summary, source_url
            FROM order_wins
            WHERE symbol = ?
            ORDER BY broadcast_dt DESC
            LIMIT ?
        """, (symbol, n)).fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({
                "symbol":         r[0],
                "company":        r[1] or "",
                "period_type":    "order_win",
                "broadcast_dt":   (r[2] or "")[:10],
                "order_value_cr": r[3],
                "client_name":    r[4] or "",
                "client_type":    r[5] or "",
                "sector":         r[6] or "",
                "_snippet":       (r[7] or "")[:300],
                "source_url":     r[8] or "",
            })
        return results

    def _sql_announcements_by_date(
        self, from_dt: date, to_dt: date, n: int = 50, subject: str | None = None
    ) -> list[dict]:
        """Query announcements directly by date range, optionally filtered by subject."""
        conn = sqlite3.connect(str(self._db_path))
        # Build subject-family filter: map the single matched subject to its sibling subjects
        _SUBJECT_FAMILIES: dict[str, list[str]] = {
            "Acquisition": ["Acquisition"],
            "Change in Management": ["Change in Management", "Appointment", "Change in Director(s)"],
            "Appointment": ["Appointment", "Change in Management", "Change in Director(s)"],
            "Resignation of Director/KMP/SMP": ["Resignation of Director/KMP/SMP", "Resignation", "Cessation"],
            "Change in Director(s)": ["Change in Director(s)", "Change in Management", "Appointment", "Resignation of Director/KMP/SMP"],
            "Credit Rating- Revision": ["Credit Rating- Revision", "Credit Rating- New"],
            "Rights Issue": ["Rights Issue", "Fund Raising - Preferential Issue", "Qualified Institutional Placement"],
            "Fund Raising - Preferential Issue": ["Rights Issue", "Fund Raising - Preferential Issue", "Qualified Institutional Placement", "Allotment of Securities"],
            "Allotment of Securities": ["Allotment of Securities", "Rights Issue", "Fund Raising - Preferential Issue"],
            "Dividend": ["Dividend", "Record Date"],
            "Buyback": ["Buyback"],
        }
        subj_list = _SUBJECT_FAMILIES.get(subject or "", [subject]) if subject else []
        if subj_list:
            ph = ",".join("?" * len(subj_list))
            sql = f"""
                SELECT symbol, company, subject, details, broadcast_dt, score
                FROM announcements
                WHERE DATE(broadcast_dt) BETWEEN ? AND ? AND subject IN ({ph})
                ORDER BY score DESC, broadcast_dt DESC
                LIMIT ?
            """
            params = (from_dt.isoformat(), to_dt.isoformat(), *subj_list, n)
        else:
            sql = """
                SELECT symbol, company, subject, details, broadcast_dt, score
                FROM announcements
                WHERE DATE(broadcast_dt) BETWEEN ? AND ?
                ORDER BY score DESC, broadcast_dt DESC
                LIMIT ?
            """
            params = (from_dt.isoformat(), to_dt.isoformat(), n)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "",
                "subject": r[2] or "", "_snippet": r[3] or "",
                "broadcast_dt": (r[4] or "")[:10], "score": r[5],
            }
            for r in rows
        ]

    def _sql_announcements_by_symbol(self, symbol: str, n: int = 20) -> list[dict]:
        """Return recent announcements for a specific company, most recent first."""
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute("""
            SELECT symbol, company, subject, details, broadcast_dt, score
            FROM announcements
            WHERE symbol = ?
            ORDER BY broadcast_dt DESC
            LIMIT ?
        """, (symbol.upper(), n)).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "",
                "subject": r[2] or "", "_snippet": r[3] or "",
                "broadcast_dt": (r[4] or "")[:10], "score": r[5] or 0,
            }
            for r in rows
        ]

    # ── Volume breakout queries ───────────────────────────────────────────────

    def _sql_volume_breakouts(
        self,
        sector:    str | None = None,
        marketcap: str | None = None,
        symbol:    str | None = None,
        from_dt:   str | None = None,
        to_dt:     str | None = None,
        n:         int = 50,
    ) -> list[dict]:
        """Return volume breakout signals filtered by sector/cap/symbol/date."""
        conn   = sqlite3.connect(str(self._db_path))
        where  = []
        params: list = []
        if symbol:
            where.append("symbol = ?");               params.append(symbol.upper())
        if sector:
            where.append("LOWER(sector) LIKE ?");     params.append(f"%{sector.lower()}%")
        if marketcap:
            where.append("LOWER(marketcap) = ?");     params.append(marketcap.lower())
        if from_dt:
            where.append("signal_date >= ?");         params.append(str(from_dt))
        if to_dt:
            where.append("signal_date <= ?");         params.append(str(to_dt))
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(f"""
            SELECT symbol, signal_date, marketcap, sector
            FROM volume_breakouts
            {where_sql}
            ORDER BY signal_date DESC
            LIMIT ?
        """, params + [n]).fetchall()
        conn.close()
        return [
            {
                "symbol":       r[0],
                "signal_date":  r[1],
                "marketcap":    r[2],
                "sector":       r[3],
                "_type":        "volume_breakout",
                "broadcast_dt": r[1],
            }
            for r in rows
        ]

    # ── Multi-hop screener ────────────────────────────────────────────────────

    # Condition-type keyword sets (distinct from the single-intent sets above)
    _MULTI_COND = [
        ("breakout",   {"breakout", "breakouts", "hvy", "volume", "surge", "spike", "momentum"}),
        ("order",      {"order", "orders", "contract", "win", "wins", "bagged", "awarded", "loa"}),
        ("financial",  {"revenue", "profit", "pat", "growth", "results", "earnings", "financial",
                        "q1", "q2", "q3", "q4", "fy", "quarterly"}),
        ("capex",      {"capacity", "expansion", "plant", "capex", "invest", "manufacturing",
                        "greenfield", "brownfield", "facility"}),
        ("fundraise",  {"qip", "fundrais", "allotment", "preferential", "ncd", "debenture",
                        "rights issue"}),
    ]
    _COMPOUND_SIGNALS = {"and", "both", "also", "plus", "along", "combined", "while", "who"}

    def _is_compound(self, message: str) -> bool:
        tl = set(re.split(r'\W+', message.lower()))
        matched = [name for name, kws in self._MULTI_COND if tl & kws]
        # Compound if 2+ distinct condition types present
        return len(matched) >= 2

    def _sql_order_wins_multi(
        self,
        sector:    str | None = None,
        marketcap: str | None = None,
        from_dt:   str | None = None,
        to_dt:     str | None = None,
        n:         int = 500,
    ) -> list[dict]:
        conn   = sqlite3.connect(str(self._db_path))
        where  = ["LOWER(subject) LIKE '%order%'"]
        params: list = []
        if sector:
            where.append("(LOWER(sector_tags) LIKE ? OR LOWER(subject) LIKE ?)")
            params += [f"%{sector.lower()}%", f"%{sector.lower()}%"]
        if from_dt:
            where.append("DATE(broadcast_dt) >= ?"); params.append(str(from_dt))
        if to_dt:
            where.append("DATE(broadcast_dt) <= ?"); params.append(str(to_dt))
        rows = conn.execute(f"""
            SELECT a.symbol, a.company, a.broadcast_dt, a.order_value_cr, a.sector_tags,
                   vb.marketcap
            FROM announcements a
            LEFT JOIN volume_breakouts vb ON vb.symbol = a.symbol
            WHERE {" AND ".join(where)}
            GROUP BY a.symbol ORDER BY a.broadcast_dt DESC LIMIT ?
        """, params + [n]).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "",
                "broadcast_dt": r[2], "order_value_cr": r[3],
                "sector_tags": r[4] or "", "marketcap": r[5] or "",
                "_type": "order_win",
                "_cond": "order",
            }
            for r in rows
        ]

    def _sql_financials_multi(
        self,
        sector:          str | None = None,
        min_rev_growth:  float | None = None,
        min_pat_growth:  float | None = None,
        from_dt:         str | None = None,
        to_dt:           str | None = None,
        n:               int = 500,
    ) -> list[dict]:
        conn   = sqlite3.connect(str(self._db_path))
        where  = [
            "period_type NOT IN ('order_win','acquisition','restructuring',"
            "'credit_rating','cirp','fundraising','buyback','open_offer')"
        ]
        params: list = []
        if sector:
            where.append("LOWER(COALESCE(fr.sector_tags,'')) LIKE ?")
            params.append(f"%{sector.lower()}%")
        if min_rev_growth is not None:
            where.append("revenue_growth_pct >= ?"); params.append(min_rev_growth)
        if min_pat_growth is not None:
            where.append("pat_growth_pct >= ?");     params.append(min_pat_growth)
        if from_dt:
            where.append("DATE(broadcast_dt) >= ?"); params.append(str(from_dt))
        if to_dt:
            where.append("DATE(broadcast_dt) <= ?"); params.append(str(to_dt))
        rows = conn.execute(f"""
            SELECT fr.symbol, fr.company, fr.period,
                   fr.revenue_cr, fr.pat_cr, fr.revenue_growth_pct, fr.pat_growth_pct,
                   fr.broadcast_dt
            FROM financial_results fr
            WHERE {" AND ".join(where)}
            ORDER BY broadcast_dt DESC LIMIT ?
        """, params + [n]).fetchall()
        conn.close()
        return [
            {
                "symbol": r[0], "company": r[1] or "", "period": r[2] or "",
                "revenue_cr": r[3], "pat_cr": r[4],
                "revenue_growth_pct": r[5], "pat_growth_pct": r[6],
                "broadcast_dt": r[7], "_type": "financials", "_cond": "financial",
            }
            for r in rows
        ]

    @staticmethod
    def _extract_pct_threshold(message: str) -> float | None:
        m = re.search(
            r'(?:>|greater than|more than|above|over|atleast|at least)\s*(\d+(?:\.\d+)?)\s*%',
            message, re.I,
        )
        return float(m.group(1)) if m else None

    def _execute_multi_hop(
        self,
        message:   str,
        sector:    str | None,
        from_dt,
        to_dt,
        n:         int = 50,
    ) -> list[dict]:
        tl        = set(re.split(r'\W+', message.lower()))
        pct_thres = self._extract_pct_threshold(message)

        # Detect marketcap filter
        marketcap = None
        if tl & {"smallcap", "small", "smallcaps"}:  marketcap = "smallcap"
        elif tl & {"midcap", "mid", "midcaps"}:       marketcap = "midcap"
        elif tl & {"largecap", "large", "largecaps"}:  marketcap = "largecap"

        fdt = str(from_dt) if from_dt else None
        tdt = str(to_dt)   if to_dt   else None

        pools: dict[str, list[dict]] = {}

        if tl & self._MULTI_COND[0][1]:  # breakout
            pools["breakout"] = self._sql_volume_breakouts(
                sector=sector, marketcap=marketcap, from_dt=fdt, to_dt=tdt, n=500,
            )
        if tl & self._MULTI_COND[1][1]:  # order
            pools["order"] = self._sql_order_wins_multi(
                sector=sector, from_dt=fdt, to_dt=tdt, n=500,
            )
        if tl & self._MULTI_COND[2][1]:  # financial
            rev_g = pct_thres if (tl & {"revenue", "rev"}) else None
            pat_g = pct_thres if (tl & {"profit", "pat"})  else None
            pools["financial"] = self._sql_financials_multi(
                sector=sector, min_rev_growth=rev_g, min_pat_growth=pat_g,
                from_dt=fdt, to_dt=tdt, n=500,
            )
        if tl & self._MULTI_COND[3][1]:  # capex
            pools["capex"] = self._sql_order_wins_multi(
                sector=sector, from_dt=fdt, to_dt=tdt, n=500,
            )  # reuse as proxy — announcements with capacity keywords

        if len(pools) < 2:
            return []

        # Intersect by symbol
        sym_sets = [set(r["symbol"] for r in rs) for rs in pools.values()]
        common   = sym_sets[0].intersection(*sym_sets[1:])
        if not common:
            return []

        cond_labels = list(pools.keys())
        combined: list[dict] = []
        for sym in sorted(common):
            entry: dict = {
                "symbol":       sym,
                "_type":        "multi_hop",
                "_conditions":  cond_labels,
                "broadcast_dt": "",
            }
            for cond, rs in pools.items():
                for r in rs:
                    if r["symbol"] == sym:
                        entry[f"_data_{cond}"] = r
                        if not entry.get("company"):
                            entry["company"] = r.get("company", "")
                        if r.get("broadcast_dt", "") > entry["broadcast_dt"]:
                            entry["broadcast_dt"] = r["broadcast_dt"]
                        break
            combined.append(entry)

        return combined[:n]

    # ── Reflection helper ────────────────────────────────────────────────────

    def _reflect_params(
        self,
        message:  str,
        intent:   str,
        sector:   str | None,
        from_dt,
        to_dt,
        min_cr:   float | None,
    ) -> dict | None:
        """
        When a temporal SQL query returns 0 rows, ask Ollama to suggest corrected
        parameters (wider date range, fixed sector name, lower threshold).
        Returns a dict with corrected params + reason, or None on failure.
        Capped at 10 s to keep the UI responsive.
        """
        import json as _json
        available_sectors = list(_SECTOR_KW.keys())
        prompt = (
            f'A BSE/NSE database query returned 0 results. Suggest corrected parameters.\n\n'
            f'User query: "{message}"\n'
            f'Detected intent: {intent}\n'
            f'Current params:\n'
            f'  sector   : {sector or "None"}\n'
            f'  from_dt  : {from_dt or "None"}\n'
            f'  to_dt    : {to_dt or "None"}\n'
            f'  min_cr   : {min_cr or "None"}\n\n'
            f'Available sectors: {", ".join(available_sectors)}\n'
            f'Today: {date.today().isoformat()}\n\n'
            f'Common fixes:\n'
            f'1. Sector mismatch — e.g. "defence" → "aerospace & defence", "banking" → "bank"\n'
            f'2. Date range too narrow — try widening by 30 days\n'
            f'3. min_cr threshold too high — halve it or set to null\n\n'
            f'Reply ONLY with a JSON object:\n'
            f'{{"sector":"corrected_or_null","from_dt":"YYYY-MM-DD_or_null",'
            f'"to_dt":"YYYY-MM-DD_or_null","min_cr":number_or_null,"reason":"brief explanation"}}'
        )
        try:
            content = llm_complete(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            ).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return None
            data = _json.loads(m.group())
            # Normalise nulls expressed as strings
            for k in ("sector", "from_dt", "to_dt"):
                if str(data.get(k, "")).lower() in ("null", "none", ""):
                    data[k] = None
            if str(data.get("min_cr", "")).lower() in ("null", "none", ""):
                data["min_cr"] = None
            return data
        except Exception:
            return None

    # ── Retrieval (Stage 1) ───────────────────────────────────────────────────

    @_traceable(
        run_type="retriever",
        name="ep-retrieve",
        process_inputs=lambda inp: {
            "message": inp.get("message", ""),
            "n": inp.get("n", 8),
        },
    )
    def retrieve(self, message: str, n: int = 8, reflect: bool = True) -> tuple[list[dict], str, float | None]:
        """
        Run the retrieval StateGraph: classify → SQL / reflect / vector → rerank.
        Returns (results, intent_string, min_cr). Fast when no reflection needed (~50 ms).
        """
        initial: RetrievalState = {
            "message":          message,
            "n":                n,
            "do_reflect":       reflect,
            # Placeholder values — node_classify_params overwrites all of these
            "intent":           "all",
            "min_cr":           None,
            "sector":           None,
            "subject":          None,
            "period_type":      None,
            "company_sym":      None,
            "from_dt":          None,
            "to_dt":            None,
            "is_multi_hop":     False,
            "results":          [],
            "reflection_count": 0,
            "early_return":     False,
            "needs_rerank":     False,
        }
        final = self._graph.invoke(initial)
        return final["results"], final["intent"], final["min_cr"]

    # ── Generation (Stage 2) ─────────────────────────────────────────────────

    @_traceable(
        run_type="llm",
        name="ep-generate",
        metadata={"model": _OLLAMA_MODEL},
        process_inputs=lambda inp: {
            "message": inp.get("message", ""),
            "intent": inp.get("intent", ""),
            "n_results": len(inp.get("results") or []),
            "top_symbols": [r.get("symbol") for r in (inp.get("results") or [])[:5]],
            "min_cr": inp.get("min_cr"),
        },
    )
    def stream_response(
        self,
        message:  str,
        results:  list[dict],
        intent:   str,
        history:  list[dict],
        min_cr:   float | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream Ollama tokens given retrieved context + conversation history.
        Yields raw text tokens as they arrive from llama3:latest.
        """
        # For pure listing queries ("give me all", "list all") bypass the LLM entirely.
        # The LLM is non-deterministic even at temperature=0 and will randomly skip
        # entries from a long list. The direct formatter is 100% deterministic.
        if _is_listing_query(message) and intent in ("order_win", "financials") and results:
            direct = _format_direct_list(results, intent, min_cr=min_cr)
            yield direct
            return

        context  = _build_context(results, intent, min_cr=min_cr)
        messages = self._build_messages(message, context, history)

        _RESPONSE_CHAR_LIMIT = 6_000  # hard cap — anything beyond is LLM padding
        _TRUNCATION_RE = re.compile(
            r'[a-zA-Z0-9₹,\.]\s*$'   # ends mid-word / mid-number with no sentence terminator
        )
        _SENTENCE_END_RE = re.compile(r'[.!?)\]"\']\s*$')

        full_response = ""
        try:
            for token in llm_stream(messages, temperature=0, max_tokens=1500):
                full_response += token
                yield token
                if len(full_response) > _RESPONSE_CHAR_LIMIT:
                    yield "\n\n*(response truncated — ask a more specific question)*"
                    return
        except Exception as e:
            yield f"\n\n⚠️ LLM error ({ACTIVE_PROVIDER}): {e}\n"
            yield "\nFalling back to direct results:\n\n"
            yield context
            return   # skip grounding check on error path

        # Output guardrail: detect truncated response (ends mid-sentence)
        stripped = full_response.rstrip()
        if stripped and not _SENTENCE_END_RE.search(stripped) and len(stripped) > 100:
            if _TRUNCATION_RE.search(stripped):
                yield "\n\n⚠️ *Response may be incomplete — the model stopped mid-sentence.*"

        # Loop 2: citation grounding — warn if LLM cited values/symbols
        # that are absent from the retrieved source context.
        warnings = _check_grounding(full_response, results, self._known_symbols)
        if warnings:
            yield (
                f"\n\n⚠️ *Grounding note: Could not verify in source data — "
                f"{', '.join(warnings)}*"
            )

    @staticmethod
    def _build_messages(
        message:  str,
        context:  str,
        history:  list[dict],
    ) -> list[dict]:
        from datetime import date as _date
        today_str = _date.today().strftime("%d %B %Y")  # e.g. "31 July 2026"
        system = _SYSTEM_PROMPT + f"\n\nToday's date is {today_str}. Use this when interpreting 'last 7 days', 'today', 'this week', etc."
        msgs = [{"role": "system", "content": system}]
        # last 4 history turns (2 user + 2 assistant)
        for turn in history[-4:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                msgs.append({"role": turn["role"], "content": str(turn["content"])[:1000]})
        # current question with fresh context
        msgs.append({
            "role": "user",
            "content": f"DATABASE CONTEXT:\n{context}\n\nQUESTION: {message}",
        })
        return msgs

    # ── Stats / suggestions ───────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        counts  = self._store.counts()
        conn    = sqlite3.connect(str(self._db_path))
        ann_sql = conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
        fin_sql = conn.execute("SELECT COUNT(*) FROM financial_results").fetchone()[0]
        dr      = conn.execute(
            "SELECT MIN(broadcast_dt), MAX(broadcast_dt) FROM announcements"
        ).fetchone()
        conn.close()
        return {
            "sql_announcements":     ann_sql,
            "sql_financial_results": fin_sql,
            "chroma_announcements":  counts["ep_announcements"],
            "chroma_financials":     counts["ep_financials"],
            "date_from": str(dr[0] or "")[:10],
            "date_to":   str(dr[1] or "")[:10],
        }

    @property
    def suggestions(self) -> list[str]:
        return self._SUGGESTIONS
