"""
LocalExtractor — zero-cost PDF extraction pipeline.

1. pdfplumber  : PDF bytes → plain text                (local, free, no ML)
2. Ollama      : plain text → JSON via format="json"   (local, free, llama3)
3. Pydantic    : JSON dict → typed model + validation  (zero-cost, type-safe)

Handles all extraction types: financial_results, order_win, acquisition,
restructuring, fundraising, buyback, open_offer, credit_rating, cirp.

Falls back to None if PDF is scanned/unreadable (Docling returns no text).

NOTE: Structured output uses Ollama's JSON mode + Pydantic validation rather
than PydanticAI tool-calling. llama3.1 supports function calling natively —
the ToolAgent (api/tool_agent.py) uses it. For PDF extraction we keep JSON
mode because constrained decoding is more reliable than tool-calling for
deeply nested financial schemas.
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


# ── Period normalization ───────────────────────────────────────────────────────

_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

def _normalize_period(raw: str) -> str:
    """Normalize any period string to 'Qn FYyyyy' canonical form."""
    if not raw:
        return raw
    # Strip trailing qualifiers: "(Consolidated)", "(Standalone)", etc.
    clean = re.sub(r"\s*\((?:Consolidated|Standalone|Audited|Unaudited)\)\s*$", "", raw, flags=re.IGNORECASE).strip()
    # Already canonical "Q1 FY2027"
    m = re.match(r"(Q[1-4])\s*FY(\d{4})$", clean, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} FY{m.group(2)}"
    # Half-year "H1 FY2027"
    m = re.match(r"(H[12])\s*FY(\d{4})$", clean, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} FY{m.group(2)}"
    # "Quarter ended June 30, 2026" / "Quarter ended June 2026"
    m = re.search(r"quarter\s+ended\s+(\w+)\s+(?:\d{1,2}[,. ]+)?(\d{4})", clean, re.IGNORECASE)
    if m:
        mon = _MONTH_NUM.get(m.group(1).lower())
        yr  = int(m.group(2))
        if mon:
            if   mon in (4, 5, 6):      return f"Q1 FY{yr + 1}"
            elif mon in (7, 8, 9):      return f"Q2 FY{yr + 1}"
            elif mon in (10, 11, 12):   return f"Q3 FY{yr + 1}"
            else:                       return f"Q4 FY{yr}"
    # "Half year ended September 2026"
    m = re.search(r"(?:six\s+months|half\s+year)\s+ended\s+(\w+)\s+(?:\d{1,2}[,. ]+)?(\d{4})", clean, re.IGNORECASE)
    if m:
        mon = _MONTH_NUM.get(m.group(1).lower())
        yr  = int(m.group(2))
        if mon:
            return f"H1 FY{yr + 1}" if mon in (7, 8, 9) else f"H2 FY{yr}"
    # "Year ended March 2027"
    m = re.search(r"year\s+ended\s+\w+\s+(?:\d{1,2}[,. ]+)?(\d{4})", clean, re.IGNORECASE)
    if m:
        return f"FY{m.group(1)}"
    return clean


# ── Audit-report-only pre-filter ──────────────────────────────────────────────

_AUDIT_PHRASES = frozenset([
    "independent auditor", "limited review report", "we have reviewed",
    "statutory auditor", "management's responsibility",
    "basis of review", "qualified conclusion", "emphasis of matter",
    "auditor's responsibility",
])
_PNL_PHRASES = frozenset([
    "revenue from operations", "net sales", "turnover",
    "profit after tax", "profit before tax", "total income",
    "earnings per share", "basic eps",
])

def _is_audit_report_only(text: str) -> bool:
    """True when text looks like an auditor's review letter with no P&L table."""
    t = text.lower()
    audit_hits = sum(1 for p in _AUDIT_PHRASES if p in t)
    has_pnl    = any(p in t for p in _PNL_PHRASES)
    return audit_hits >= 2 and not has_pnl

from pydantic import BaseModel, Field, ValidationError

from financial.models import FinancialResult
from financial.rule_extractor import try_rule_extract, extract_order_win_from_table
from financial.financial_table_extractor import extract_financial_table


_OLLAMA_MODEL   = "llama3.1:latest"
_MAX_TEXT_CHARS = 10_000   # chars per section sent to Ollama
_MAX_CHUNK_CHARS = 80_000  # max raw text retained for section-aware chunking

# ── Section detection for long financial PDFs ─────────────────────────────────

_SEC_PNL = re.compile(
    r"(?im)"
    r"(?:(?:standalone|consolidated)\s+)?"
    r"(?:statement\s+of\s+profit\s+(?:and|&)\s+loss"
    r"|profit\s+(?:and|&)\s+loss\s+(?:account|statement)"
    r"|income\s+(?:and\s+expenditure\s+)?statement)",
)
_SEC_BALANCE = re.compile(
    r"(?im)(?:(?:standalone|consolidated)\s+)?balance\s+sheet"
    r"|statement\s+of\s+(?:financial\s+)?position",
)
_SEC_CASHFLOW = re.compile(
    r"(?im)cash\s+flow\s+statement|statement\s+of\s+cash\s+flows?",
)
_SEC_NOTES = re.compile(
    r"(?im)notes?\s+(?:to|forming\s+part\s+of)\s+(?:the\s+)?(?:financial\s+statements?|accounts?)",
)

_ALL_SECTIONS: list[tuple[str, re.Pattern]] = [
    ("pnl",           _SEC_PNL),
    ("balance_sheet", _SEC_BALANCE),
    ("cash_flow",     _SEC_CASHFLOW),
    ("notes",         _SEC_NOTES),
]


def _best_section_pos(text: str, pattern: re.Pattern) -> int | None:
    """
    Return the start position of the preferred section match.
    When multiple matches exist (standalone + consolidated statements):
      1. Prefer the match whose text contains 'consolidated'.
      2. Fall back to the last match — consolidated typically appears second
         in SEBI dual-statement filings (Standalone first, Consolidated after).
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    if len(matches) > 1:
        for m in matches:
            if "consolidated" in m.group().lower():
                return m.start()
        return matches[-1].start()
    return matches[0].start()


def _split_financial_sections(text: str) -> dict[str, str]:
    """
    Split a long financial PDF at standard Indian financial statement headers.
    Returns {section_name: text}.  Leading content is stored under 'header'.
    When both Standalone and Consolidated statements are present, the
    Consolidated section is preferred for 'pnl' and 'balance_sheet'.
    """
    boundaries: list[tuple[int, str]] = []
    for name, pat in _ALL_SECTIONS:
        pos = _best_section_pos(text, pat)
        if pos is not None:
            boundaries.append((pos, name))
    boundaries.sort()

    if not boundaries:
        return {"header": text}

    sections: dict[str, str] = {}
    if boundaries[0][0] > 100:
        sections["header"] = text[: boundaries[0][0]]

    for i, (start, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections[name] = text[start:end]

    return sections


def _merge_financial_dicts(results: list[dict]) -> dict:
    """
    Merge multiple Ollama extraction dicts (P&L first = highest priority).
    Scalar fields: first non-null wins.  key_highlights: deduplicated union.
    """
    if not results:
        return {}
    merged = dict(results[0])

    scalar_fields = [
        "period", "period_type",
        "revenue_cr", "revenue_growth_pct", "prior_revenue_cr",
        "ebitda_cr", "ebitda_margin_pct",
        "pat_cr", "pat_growth_pct", "pbt_cr",
        "depreciation_cr", "finance_costs_cr",
        "eps", "dividend_per_share",
        "order_value_cr", "execution_months",
        "client_name", "client_type", "order_sector",
        "guidance", "raw_summary",
    ]
    for extra in results[1:]:
        for f in scalar_fields:
            if not merged.get(f) and extra.get(f):
                merged[f] = extra[f]
        seen = set(merged.get("key_highlights") or [])
        for h in extra.get("key_highlights") or []:
            if h not in seen:
                merged.setdefault("key_highlights", []).append(h)
                seen.add(h)

    return merged

# Wrap Ollama financial call with LangSmith tracing when enabled
def _maybe_traceable(fn):
    import os
    if os.getenv("LANGSMITH_TRACING", "").lower() not in ("true", "1"):
        return fn
    try:
        from langsmith import traceable
        return traceable(
            name="ollama_financial_extraction",
            run_type="llm",
            tags=["ollama", "financial"],
        )(fn)
    except ImportError:
        return fn


# ── Pydantic extraction models (validation layer on top of Ollama JSON) ───────
#
# These models validate and coerce the raw JSON dict returned by Ollama, giving
# us type safety and consistent nulls without needing tool-calling support.

class FinancialExtraction(BaseModel):
    period:             str            = Field("", description="Quarter/year label e.g. 'Q1 FY2027'")
    period_type:        str            = "quarterly"
    revenue_cr:         Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    prior_revenue_cr:   Optional[float] = None   # same quarter last year — used to compute growth
    ebitda_cr:          Optional[float] = None
    ebitda_margin_pct:  Optional[float] = None
    pat_cr:             Optional[float] = None
    pat_growth_pct:     Optional[float] = None
    pbt_cr:             Optional[float] = None   # profit before tax — used to compute EBITDA
    depreciation_cr:    Optional[float] = None   # used to compute EBITDA
    finance_costs_cr:   Optional[float] = None   # used to compute EBITDA
    eps:                Optional[float] = None
    dividend_per_share: Optional[float] = None
    key_highlights:     list[str]       = Field(default_factory=list)
    guidance:           str             = ""
    raw_summary:        str             = ""


class OrderWinExtraction(BaseModel):
    period:           str            = ""
    order_value_cr:   Optional[float]= None
    client_name:      str            = ""
    client_type:      str            = ""
    order_sector:     str            = ""
    execution_months: Optional[int]  = None
    description:      str            = ""
    key_highlights:   list[str]      = Field(default_factory=list)
    raw_summary:      str            = ""


class GeneralExtraction(BaseModel):
    period:           str            = ""
    period_type:      str            = "press_release"
    order_value_cr:   Optional[float]= None
    client_name:      str            = ""
    client_type:      str            = ""
    order_sector:     str            = ""
    execution_months: Optional[int]  = None
    key_highlights:   list[str]      = Field(default_factory=list)
    raw_summary:      str            = ""


class ConcallExtraction(BaseModel):
    period:           str       = ""
    revenue_guidance: str       = ""
    margin_guidance:  str       = ""
    key_highlights:   list[str] = Field(default_factory=list)
    raw_summary:      str       = ""


# ── Extraction prompts (used with ollama JSON mode) ──────────────────────────

_FINANCIAL_PROMPT = """You are a senior equity analyst. Extract financial metrics from the Indian BSE/NSE quarterly results document below.

═══ UNIT CONVERSION (critical) ═══
Step 1: Find the unit declaration at the top of the P&L table. Look for:
  - "Rs. in Lakhs" or "₹ in Lakhs" → divide ALL values by 100 to get Crores
  - "Rs. in Millions" or "₹ in Millions" → divide ALL values by 10 to get Crores
  - "Rs. in Crores" or "₹ in Crores" → use values as-is
  - If unclear: numbers like 2,94,270 (Indian lakh format X,XX,XXX) → assume Lakhs → divide by 100
  - Numbers like 29,427 (small, 5-digit) → likely Crores already
ALWAYS output final values in Rs. Crore.

═══ COLUMN SELECTION ═══
Indian results PDFs have 4 columns: [Current Quarter, Previous Quarter, Same Quarter Last Year, YTD].
Extract ONLY the FIRST (leftmost) data column = Current Quarter.
DO NOT average columns. DO NOT add columns.

═══ NEGATIVE NUMBERS ═══
(3,900) or (3900) with parentheses = NEGATIVE → -39.0 Cr (if in Lakhs)

═══ CONSOLIDATED vs STANDALONE ═══
If both are present, prefer "Consolidated" results. Label period as "Q1 FY2027 (Consolidated)".

═══ PERIOD DETECTION ═══
Output period as "Qn FYyyyy" format only — never "Quarter ended …".
Indian FY starts April 1:  Apr-Jun = Q1,  Jul-Sep = Q2,  Oct-Dec = Q3,  Jan-Mar = Q4
"Quarter ended June 2026"      → "Q1 FY2027"
"Quarter ended September 2026" → "Q2 FY2027"
"Quarter ended December 2026"  → "Q3 FY2027"
"Quarter ended March 2027"     → "Q4 FY2027"
If Consolidated results are present, append " (Consolidated)": "Q1 FY2027 (Consolidated)"

═══ ROW LABELS TO FIND ═══
Current Quarter column (leftmost data column):
- revenue_cr      : "Revenue from Operations" / "Net Revenue" / "Turnover" / "Total Income"
- pat_cr          : "Profit After Tax" / "Net Profit" / "PAT" / "Profit for the period"
- pbt_cr          : "Profit Before Tax" / "PBT" / "Profit Before Exceptional Items and Tax"
- depreciation_cr : "Depreciation" / "Depreciation and Amortisation" (standalone line item)
- finance_costs_cr: "Finance Costs" / "Interest Expense" / "Borrowing Costs"
- ebitda_cr       : "EBITDA" row — extract directly if labelled; otherwise leave null
- eps             : "Basic EPS" / "Earnings Per Share (Basic)" — NEVER scale EPS by unit factor

Same Quarter Last Year column (3rd column in standard 4-column Indian results table):
- prior_revenue_cr: revenue from same quarter last year — apply the same unit conversion

═══ UNIT CONVERSION RULES ═══

Apply the unit factor from the table header to EVERY numeric cell before writing JSON:
  "Rs. in Lakhs"   header → divide each cell by 100    (e.g. X,XX,XXX Lakhs → X,XX,XXX ÷ 100 Crores)
  "Rs. in Millions" header → divide each cell by 10    (e.g. X,XXX Millions → X,XXX ÷ 10 Crores)
  "Rs. in Crores"  header → use values as-is           (no division)
  Absolute rupees (₹X,XX,XX,XXX format, 8+ digits) → divide by 1,00,00,000

EPS: NEVER apply the unit divisor to EPS — always use the exact EPS value from the EPS row.
Negative: cell showing (X.X) with parentheses → output as −X.X
Growth pct: only fill if BOTH current and prior period are visible in the table; else null

IMPORTANT: Do NOT use any example numbers from these instructions. Compute and extract ONLY values that appear in the document text below.

═══ OUTPUT FORMAT ═══
Return ONLY valid JSON, no extra text or markdown:
{{
  "period": "<period from document>",
  "period_type": "quarterly",
  "revenue_cr": <number from document or null>,
  "revenue_growth_pct": <number or null>,
  "ebitda_cr": <number from document or null>,
  "ebitda_margin_pct": <number or null>,
  "pat_cr": <number from document or null>,
  "pat_growth_pct": <number or null>,
  "pbt_cr": <number from document or null>,
  "depreciation_cr": <number from document or null>,
  "finance_costs_cr": <number from document or null>,
  "prior_revenue_cr": <same quarter last year revenue or null>,
  "eps": <number from document or null>,
  "dividend_per_share": null,
  "key_highlights": ["<fact from document>"],
  "guidance": "<quote from document or null>",
  "raw_summary": "<1-2 sentences using only numbers found in document>",
  "confidence": "high"
}}

confidence must be one of: "high" (3+ numeric fields extracted), "medium" (1-2 fields), "low" (0 fields, text only)

Company: {company} ({symbol})
Filing date: {broadcast_dt}

DOCUMENT:
{text}"""


_GENERAL_PROMPT = """You are a senior equity analyst. Read the corporate announcement below and extract key facts.

{instructions}

RULES:
- All monetary values in Rs. Crore. Convert USD at ~84 rate, Lakhs ÷100.
- Return null or "not disclosed" for missing facts.
- Return ONLY valid JSON, no extra text.

Return this exact JSON structure:
{{
  "period": "{broadcast_dt}",
  "key_highlights": ["fact 1", "fact 2", "fact 3", "fact 4"],
  "raw_summary": "2-sentence plain English summary."
}}

Company: {company} ({symbol})
Filing date: {broadcast_dt}

DOCUMENT:
{text}"""


_CONCALL_PROMPT = """You are a senior equity analyst reviewing a concall transcript or investor presentation.
Extract management guidance and key messages from the document.

RULES:
- Focus on forward-looking statements, guidance, and management commentary
- Extract specific numbers where mentioned; use null if not stated
- Return ONLY valid JSON, no extra text

Return this exact JSON structure:
{{
  "period": "Q1 FY2027 concall",
  "revenue_guidance": "Revenue expected to grow 12-15% in FY2027 to Rs.X Cr",
  "margin_guidance": "EBITDA margin expected at 18-20% for FY2027",
  "key_highlights": [
    "Management commentary point 1",
    "Order pipeline / demand outlook",
    "Margin or cost commentary",
    "Capex or expansion plans",
    "Key risk or challenge mentioned"
  ],
  "raw_summary": "2-sentence summary of management's key message and business outlook."
}}

Company: {company} ({symbol})
Filing date: {broadcast_dt}

CONCALL / INVESTOR MEET DOCUMENT:
{text}"""


_INSTRUCTIONS: dict[str, str] = {
    "order_win": (
        "Extract: order value (Rs. Cr), client name and type (govt/PSU/private), "
        "goods or services being supplied, contract duration, sector/end-market (defence/railways/power etc.). "
        "IMPORTANT: order_value_cr must be the VALUE OF THE SPECIFIC NEW ORDER RECEIVED, "
        "NOT the cumulative order book size or total backlog. "
        "If the text says 'Order Book crosses Rs.25,750 Cr' but the new order is Rs.960 Cr, "
        "use 960, not 25750."
    ),
    # Press releases vary widely — try to identify what type of announcement this is
    # and extract relevant facts. If an order value exists, extract it.
    "press_release": (
        "This may be an order win, a business update, a product launch, a regulatory approval, "
        "or any other corporate development. "
        "If a specific order or contract value is mentioned, extract: order_value_cr (Rs. Cr), "
        "client name, sector/end-market. "
        "IMPORTANT: order_value_cr must be the VALUE OF THE NEW ORDER, not total order book or backlog. "
        "For any announcement, extract 4-5 key facts as highlights. "
        "Set period_type to 'order_win' if there is a clear order/contract value, else 'press_release'."
    ),
    "concall": (
        "Extract management guidance: revenue growth target, margin guidance, order pipeline status, "
        "key themes discussed, important Q&A points. Focus on forward-looking statements with specific numbers."
    ),
    "acquisition": (
        "Extract: target company name and business, deal value (Rs. Cr), "
        "stake % acquired, valuation multiple if stated, strategic rationale, "
        "regulatory approvals needed and timeline."
    ),
    "restructuring": (
        "Extract: companies involved, swap ratio or cash consideration, "
        "type of restructuring (merger/demerger/hive-off), strategic rationale, "
        "NCLT filing status and effective date."
    ),
    "fundraising": (
        "Extract: instrument type (QIP/rights/preferential/NCD), amount raised (Rs. Cr), "
        "issue price per share, named allottees/investors, dilution %, use of proceeds."
    ),
    "buyback": (
        "Extract: buyback size (Rs. Cr), max buyback price per share, method "
        "(open market / tender offer), shares to be bought back, % of paid-up capital, timeline."
    ),
    "open_offer": (
        "Extract: acquirer name, offer price per share, premium to last close, "
        "% stake in open offer, total offer size (Rs. Cr), SEBI timeline."
    ),
    "credit_rating": (
        "Extract: new rating and outlook (e.g. CRISIL AA+/Stable), previous rating if revision, "
        "instrument and total debt amount, key strengths cited, key risks/concerns."
    ),
    "cirp": (
        "Extract: total admitted debt, resolution applicant name and offered amount, "
        "implied haircut %, NCLT bench and key dates, status (admitted/approved/challenged), "
        "impact on equity shareholders."
    ),
}


# ── LocalExtractor ────────────────────────────────────────────────────────────

class LocalExtractor:
    """
    Uses pdfplumber + Ollama (qwen2.5:14b) to extract structured data from PDFs.
    No cloud API calls — runs entirely on local machine. Zero cost.
    pdfplumber handles text-based PDFs (covers most BSE/NSE filings).
    Scanned/image-only PDFs return None and are skipped.
    """

    def __init__(self) -> None:
        pass

    # ── public ────────────────────────────────────────────────────────────────

    def extract(
        self,
        pdf_bytes:       bytes,
        symbol:          str,
        company:         str,
        broadcast_dt:    str,
        source_url:      str,
        extraction_type: str,
    ) -> FinancialResult | None:
        """
        Convert PDF → text via Docling, then extract facts via Ollama.
        Returns None if PDF is unreadable or Ollama call fails.
        """
        self._last_pdf_bytes = pdf_bytes  # stashed for vision fallback if Ollama is skipped
        # ── Fastest path: direct table extraction (no LLM needed) ───────────
        if extraction_type == "order_win":
            result = extract_order_win_from_table(
                pdf_bytes, symbol, company, broadcast_dt, source_url
            )
            if result is not None:
                print(f"    [table] Annexure-A parsed directly", flush=True)
                return result
            print(f"    [table] Annexure-A not found — trying text extraction", flush=True)

        elif extraction_type == "financial_results":
            result = extract_financial_table(
                pdf_bytes, symbol, company, broadcast_dt, source_url
            )
            if result is not None:
                print(f"    [table] P&L table parsed directly", flush=True)
                return result
            print(f"    [table] P&L table not found — falling back to text+Ollama", flush=True)
            # Fall through to text extraction + _call_ollama_financial below

        # ── Press release: regex first, Ollama fallback, reclassify if order ─
        elif extraction_type == "press_release":
            text = self._pdf_to_text(pdf_bytes)
            if not text or len(text.strip()) < 80:
                data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, "press_release")
                if data is None:
                    return None
                etype = "order_win" if data.get("order_value_cr") else "press_release"
                return self._build_result(data, symbol, company, broadcast_dt, source_url, etype)
            text = text[:_MAX_TEXT_CHARS]
            result = try_rule_extract("press_release", text, symbol, company, broadcast_dt, source_url)
            if result is not None:
                print(f"    [rules] order value found in press release", flush=True)
                return result
            print(f"    [ollama] calling {_OLLAMA_MODEL} for press release", flush=True)
            data = self._call_ollama_general(text, symbol, company, broadcast_dt, "press_release")
            if data is None:
                return None
            # Reclassify to order_win if Ollama detected an order value
            etype = "order_win" if data.get("order_value_cr") else "press_release"
            return self._build_result(data, symbol, company, broadcast_dt, source_url, etype)

        # ── Concall: always Ollama (no table/regex patterns for transcripts) ─
        elif extraction_type == "concall":
            text = self._pdf_to_text(pdf_bytes)
            if not text or len(text.strip()) < 80:
                data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, "concall")
                if data is None:
                    return None
                return self._build_result(data, symbol, company, broadcast_dt, source_url, "concall")
            text = text[:_MAX_TEXT_CHARS]
            print(f"    [ollama] calling {_OLLAMA_MODEL} for concall", flush=True)
            data = self._call_ollama_concall(text, symbol, company, broadcast_dt)
            if data is None:
                return None
            return self._build_result(data, symbol, company, broadcast_dt, source_url, "concall")

        text = self._pdf_to_text(pdf_bytes)
        if not text or len(text.strip()) < 80:
            data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
            if data is None:
                return None
            return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)

        _limit = _MAX_CHUNK_CHARS if extraction_type == "financial_results" else _MAX_TEXT_CHARS
        text = text[:_limit]

        # ── Fast path: regex rules for predictable formats ─────────────────
        result = try_rule_extract(extraction_type, text[:_MAX_TEXT_CHARS], symbol, company, broadcast_dt, source_url)
        if result is not None:
            print(f"    [rules] extracted via regex (no LLM needed)", flush=True)
            return result

        # ── Slow path: Ollama for complex / irregular formats ──────────────
        # Guard: if pdfplumber extracted very little text per page, the PDF is likely scanned.
        # Sending sparse text to llama3 causes hallucination — route to Gemini Flash instead.
        # 2500 chars threshold: scanned PDFs extract only boilerplate (1100-2500 chars across
        # 5-22 pages), while text PDFs with actual tables have 5000+ chars of content.
        _MIN_USEFUL_CHARS = 2500
        if len(text.strip()) < _MIN_USEFUL_CHARS:
            print(f"    [ollama] SKIP — only {len(text.strip())} chars extracted (scanned PDF?), trying vision", flush=True)
            data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
            if data is None:
                return None
            return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)

        # Audit-report pre-filter: if text contains auditor language but no P&L keywords
        # the PDF is an auditor's review letter — skip Ollama (which hallucinates on prose)
        # and route straight to Gemini Flash which reads the images on other pages.
        if extraction_type == "financial_results" and _is_audit_report_only(text[:_MAX_TEXT_CHARS]):
            print("    [filter] audit-only text — skipping Ollama, trying vision", flush=True)
            data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
            if data is None:
                return self._build_result(
                    {"period": broadcast_dt[:10], "confidence": "low",
                     "raw_summary": "[AUDIT REPORT] PDF contains only the auditor's review report — no P&L table found."},
                    symbol, company, broadcast_dt, source_url, extraction_type,
                )
            return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)

        print(f"    [ollama] no rules matched — calling {_OLLAMA_MODEL}", flush=True)
        if extraction_type == "financial_results":
            if len(text) > _MAX_TEXT_CHARS:
                data = self._extract_chunked_financial(text, symbol, company, broadcast_dt)
            else:
                data = self._call_ollama_financial(text, symbol, company, broadcast_dt)
        else:
            data = self._call_ollama_general(text[:_MAX_TEXT_CHARS], symbol, company, broadcast_dt, extraction_type)

        if data is None:
            return None

        # ── Post-Ollama hallucination guard ────────────────────────────────
        # If Ollama returned values that match known prompt-example fingerprints,
        # it hallucinated instead of extracting. Try vision fallback if available.
        result = self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)
        if result.confidence == "low":
            _log.warning("[%s] Ollama returned low-confidence result — trying vision fallback", symbol)
            vision_data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
            if vision_data is not None:
                vision_result = self._build_result(vision_data, symbol, company, broadcast_dt, source_url, extraction_type)
                if vision_result is not None:
                    print(f"    [vision] upgraded low-confidence Ollama result", flush=True)
                    return vision_result
        return result

    # ── Two-phase API (used by PDFAgent for parallel pipeline) ───────────────

    def fast_extract(
        self,
        pdf_bytes:       bytes,
        symbol:          str,
        company:         str,
        broadcast_dt:    str,
        source_url:      str,
        extraction_type: str,
    ) -> tuple["FinancialResult | None", "str | None"]:
        """
        Phase 1 — no Ollama.  Runs table parse + regex only.
        Returns (result, None)  — fast path succeeded, no Ollama needed.
        Returns (None,   text)  — fast path missed; caller should enqueue for Ollama.
        Returns (None,   None)  — PDF unreadable (scanned / < 80 chars of text).
        """
        if extraction_type == "order_win":
            result = extract_order_win_from_table(
                pdf_bytes, symbol, company, broadcast_dt, source_url
            )
            if result is not None:
                print(f"    [table] Annexure-A parsed directly", flush=True)
                return result, None

        elif extraction_type == "financial_results":
            result = extract_financial_table(
                pdf_bytes, symbol, company, broadcast_dt, source_url
            )
            if result is not None and result.revenue_cr is not None:
                # Sanity-check the text layer before trusting the table numbers.
                # Garbled PDFs strip decimal points: "289.67" → "28967", giving
                # 100× wrong values after unit conversion. Detect this by checking
                # text quality — if the text is scrambled, route to vision instead.
                _tq_text = self._pdf_to_text(pdf_bytes)
                _tq_reject = self._check_text_quality(_tq_text, symbol) if _tq_text else None
                if not _tq_reject:
                    print(f"    [table] P&L table parsed directly", flush=True)
                    return result, None
                # Garbled text layer — decimal points may have been stripped.
                # Prefer vision fallback over trusting wrong numbers.
                print(f"    [table] P&L hit but text garbled ({_tq_reject}) — trying vision", flush=True)
                vision_data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
                if vision_data is not None:
                    return self._build_result(vision_data, symbol, company, broadcast_dt, source_url, extraction_type), None
                # Vision unavailable — return garbled table result as last resort
                print(f"    [table] vision unavailable — using table result (may have wrong values)", flush=True)
                return result, None
            if result is not None:
                # Partial hit: table structure found but revenue missing (garbled values)
                # Fall through to text+Ollama rather than storing incomplete data
                print(f"    [table] P&L table partial (revenue missing) — falling back to text+Ollama", flush=True)
            else:
                print(f"    [table] P&L table not found — falling back to text+Ollama", flush=True)

        text = self._pdf_to_text(pdf_bytes)
        if not text or len(text.strip()) < 80:
            # Image-based PDF: try vision before giving up
            data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
            if data is not None:
                result = self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)
                return result, None
            return None, None
        _limit = _MAX_CHUNK_CHARS if extraction_type == "financial_results" else _MAX_TEXT_CHARS
        text = text[:_limit]

        # summary_only: no rules apply — skip to Ollama summarizer
        if extraction_type == "summary_only":
            return None, text[:_MAX_TEXT_CHARS]

        # Order book / admin doc guard — same filter applied in ollama_extract(),
        # but must also run here so fast-path regex doesn't pick up order book values.
        _FINANCIAL_TYPES = {"financial_results", "order_win", "acquisition", "fundraising",
                            "restructuring", "buyback", "open_offer"}
        if extraction_type in _FINANCIAL_TYPES and self._is_non_financial_text(text[:_MAX_TEXT_CHARS]):
            print(f"    [skip] non-financial doc detected in fast path (type={extraction_type})", flush=True)
            return None, None

        result = try_rule_extract(extraction_type, text[:_MAX_TEXT_CHARS], symbol, company, broadcast_dt, source_url)
        if result is not None:
            print(f"    [rules] extracted via regex", flush=True)
            return result, None

        return None, text

    # ── Input guardrails ─────────────────────────────────────────────────────

    # Patterns that could manipulate llama3 if injected from PDF text
    _INJECTION_PATTERNS = re.compile(
        r'(===\s*(END|BEGIN|DATA|SYSTEM|INSTRUCTION)|'
        r'<\s*(instructions?|system|prompt)\s*>|'
        r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions?|'
        r'you\s+are\s+(now\s+)?a\s+\w+|'
        r'disregard\s+your)',
        re.IGNORECASE,
    )

    @staticmethod
    def _garble_ratio(text: str, sample: int = 1000) -> float:
        """Fraction of characters that are non-printable / high-unicode (OCR noise)."""
        chunk = text[:sample]
        if not chunk:
            return 1.0
        noise = sum(1 for c in chunk if ord(c) > 127 or (ord(c) < 32 and c not in '\n\r\t'))
        return noise / len(chunk)

    def _sanitize_for_prompt(self, text: str) -> str:
        """Strip prompt-injection patterns and garbled OCR from PDF text."""
        # Remove injection attempts
        text = self._INJECTION_PATTERNS.sub('[removed]', text)
        # Collapse runs of non-ASCII noise characters (common in bad OCR)
        text = re.sub(r'[^\x00-\x7F]{4,}', ' ', text)
        return text

    # Common 2-4 letter English words that appear in virtually every financial filing.
    # A section of text with <3% of these is font-encoding scrambled (ASCII chars but wrong).
    _COMMON_WORDS = frozenset({
        "the", "of", "to", "and", "in", "is", "for", "on", "at", "by",
        "we", "as", "it", "be", "an", "or", "are", "not", "has", "our",
        "its", "was", "but", "had", "from", "with", "this", "that",
        "per", "net", "tax", "all", "any", "may", "was", "the",
    })

    def _check_text_quality(self, text: str, symbol: str) -> str | None:
        """
        Return a rejection reason if the text is too garbled to extract from,
        or None if quality is acceptable.
        """
        ratio = self._garble_ratio(text)
        if ratio > 0.30:
            return f"garble_ratio={ratio:.2f} (>30% non-ASCII noise)"
        # Repetitive-character pattern: GCSL-type corrupted PDFs repeat same bytes
        sample = text[:500].replace(' ', '').replace('\n', '')
        if len(sample) > 50:
            unique_ratio = len(set(sample)) / len(sample)
            if unique_ratio < 0.05:
                return f"unique_char_ratio={unique_ratio:.2f} (<5% — repeated garbage)"
        # Scrambled ASCII check: bad font encoding produces valid ASCII chars in wrong
        # order ("JINDPAOLL Y" instead of "JINDAL POLY"). Detect by measuring common
        # English word density at several points — financial tables appear at different
        # depths depending on document length (cover letter + auditor report vary 1–10 pages).
        # Check at 1/6, 1/4, and 1/3 to cover short docs (table on page 2–3) and
        # long docs (42-page filings where table is on page 8–12 ≈ 1/6 of document).
        if len(text) > 1500:
            for frac in (1/6, 1/4, 1/3):
                offset = int(len(text) * frac)
                mid_sample = text[offset : offset + 2000].lower()
                short_words = re.findall(r'\b[a-z]{2,4}\b', mid_sample)
                if len(short_words) >= 20:
                    common_count = sum(1 for w in short_words if w in self._COMMON_WORDS)
                    common_ratio = common_count / len(short_words)
                    if common_ratio < 0.07:
                        return (f"scrambled_ascii at {frac:.0%} mark "
                                f"(common_word_ratio={common_ratio:.3f} — font encoding issue)")
        return None

    # Keywords that strongly indicate a non-financial PDF (no P&L/order data)
    _NON_FINANCIAL_KEYWORDS = frozenset({
        "resignation", "resignations", "auditor resignation", "record date",
        "export certificate", "star export", "audio link", "conference call link",
        "investor meet", "analyst meet", "postal ballot", "notice of agm",
        "notice of egm", "scrutinizer report", "compliance certificate",
        "trading window", "insider trading", "code of conduct", "whistleblower",
        "change in registered", "change of address", "change in name",
        "disclosure under reg 30", "intimation of appointment",
        "intimation of resignation", "appointment of director",
        # Allotment / warrant documents — board meetings that approve capital actions,
        # not financial results.  Often filed under "Outcome of Board Meeting".
        "allotment of warrants", "allotment of securities", "warrant allotment",
        "board approves allotment", "allotment letter", "convertible warrants",
        "exercise of warrants", "warrant holder",
        # Routine regulatory / admin
        "investor presentation", "analyst presentation",
        "postal ballot notice", "notice of postal",
        # Order book update filings — quarterly order book summaries, not new orders
        "order book position", "order book update", "order book as on",
        "order book status", "order book details", "order book summary",
    })

    def _is_non_financial_text(self, text: str) -> bool:
        """Return True if the text is clearly a non-financial/admin document."""
        sample = text[:1500].lower()
        hits = sum(1 for kw in self._NON_FINANCIAL_KEYWORDS if kw in sample)
        # Also check: very short text with no numeric patterns likely has no data
        has_numbers = bool(re.search(r'\d{2,}[\.,]\d{2}', text[:2000]))
        return hits >= 2 or (hits >= 1 and not has_numbers)

    def _summarize_text(self, text: str, symbol: str, company: str) -> str:
        """Generate a short plain-text summary using Ollama (no JSON schema)."""
        snippet = text[:3000]
        prompt = (
            f"Summarize this NSE filing for {company} ({symbol}) in 2-3 sentences. "
            f"State what type of document it is and the key facts:\n\n{snippet}"
        )
        try:
            import ollama as _ollama
            resp = _ollama.chat(
                model=_OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_gpu": 99, "num_predict": 200},
            )
            try:
                return resp.message.content.strip()
            except AttributeError:
                return resp["message"]["content"].strip()
        except Exception as e:
            return f"[summary failed: {e}]"

    def ollama_extract(
        self,
        text:            str,
        symbol:          str,
        company:         str,
        broadcast_dt:    str,
        source_url:      str,
        extraction_type: str,
        pdf_bytes:       bytes | None = None,
    ) -> "FinancialResult | None":
        """
        Phase 2 — Ollama only.  Called with pre-extracted text from fast_extract().
        Always runs sequentially (single GPU can only process one request at a time).

        Pre-filter: if text looks like a non-financial admin doc (resignation, notice,
        certificate etc.), skip the heavy JSON schema and store a plain summary instead.

        pdf_bytes: optional — when provided, enables vision fallback for scanned/
        scrambled PDFs that cannot be extracted via text alone.
        """
        # Guardrail 1: reject garbled / corrupted text (GCSL-type failures)
        reject_reason = self._check_text_quality(text, symbol)
        if reject_reason:
            print(f"    [guardrail] {symbol}: text quality rejected — {reject_reason}", flush=True)
            if pdf_bytes is not None:
                print(f"    [vision] trying vision fallback after text quality rejection", flush=True)
                data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
                if data is not None:
                    return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)
            return None

        # Guardrail 2: sanitize prompt-injection patterns before any LLM call
        text = self._sanitize_for_prompt(text)

        # Guardrail 3a: subject says "summary_only" — skip financial schema entirely
        if extraction_type == "summary_only":
            summary = self._summarize_text(text, symbol, company)
            print(f"    [summary] subject-routed to plain summary", flush=True)
            data = {
                "period": broadcast_dt[:10],
                "raw_summary": summary,
                "confidence": "medium",
                "key_highlights": [],
            }
            return self._build_result(data, symbol, company, broadcast_dt, source_url, "summary_only")

        # Guardrail 3b: content says non-financial even for a financial/order subject
        # (e.g. "Outcome of Board Meeting" that is actually about warrant allotments)
        _FINANCIAL_TYPES = {"financial_results", "order_win", "acquisition", "fundraising",
                            "restructuring", "buyback", "open_offer"}
        if extraction_type in _FINANCIAL_TYPES and self._is_non_financial_text(text):
            summary = self._summarize_text(text, symbol, company)
            print(f"    [skip] non-financial doc detected (type={extraction_type}) — storing summary only", flush=True)
            data = {
                "period": broadcast_dt[:10],
                "raw_summary": f"[NON-FINANCIAL] {summary}",
                "confidence": "low",
                "key_highlights": [],
            }
            return self._build_result(data, symbol, company, broadcast_dt, source_url, "press_release")

        # Guardrail 3c: audit-only text in the two-phase path — financial table is on a
        # later page (scanned image) that pdfplumber couldn't reach.  Skip Ollama (it
        # only sees the auditor's letter) and route straight to Gemini Flash vision.
        if extraction_type == "financial_results" and _is_audit_report_only(text):
            print(f"    [filter] audit-only text in Ollama phase — trying vision fallback", flush=True)
            if pdf_bytes is not None:
                data = self._vision_fallback(pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type)
                if data is not None:
                    return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)
            return None

        print(f"    [ollama] calling {_OLLAMA_MODEL} for {extraction_type}", flush=True)
        if extraction_type == "financial_results":
            if len(text) > _MAX_TEXT_CHARS:
                data = self._extract_chunked_financial(text, symbol, company, broadcast_dt)
            else:
                data = self._call_ollama_financial(text, symbol, company, broadcast_dt)
        elif extraction_type == "concall":
            data = self._call_ollama_concall(text[:_MAX_TEXT_CHARS], symbol, company, broadcast_dt)
        else:
            data = self._call_ollama_general(text[:_MAX_TEXT_CHARS], symbol, company, broadcast_dt, extraction_type)

        if data is None:
            return None

        etype = extraction_type
        if extraction_type == "press_release":
            etype = "order_win" if data.get("order_value_cr") else "press_release"
        return self._build_result(data, symbol, company, broadcast_dt, source_url, etype)

    # ── Section-aware chunked extraction ──────────────────────────────────────

    def _extract_chunked_financial(
        self,
        text:         str,
        symbol:       str,
        company:      str,
        broadcast_dt: str,
    ) -> dict | None:
        """
        For long financial PDFs (text > _MAX_TEXT_CHARS): split by section header,
        run Ollama on the P&L section first, then Cash Flow to fill EBITDA
        components, then merge results.  Falls back to first _MAX_TEXT_CHARS if
        no section headers are found.
        """
        sections = _split_financial_sections(text)
        results: list[dict] = []

        for section_name in ("pnl", "cash_flow", "header"):
            sec_text = sections.get(section_name, "").strip()
            if not sec_text:
                continue
            chunk = sec_text[:_MAX_TEXT_CHARS]
            print(
                f"    [chunk] '{section_name}' "
                f"({len(sec_text):,} chars → {len(chunk):,} sent to Ollama)",
                flush=True,
            )
            data = self._call_ollama_financial(chunk, symbol, company, broadcast_dt)
            if data:
                results.append(data)

            # P&L gave confident numbers — no need to process further sections
            if results and section_name == "pnl":
                numeric_count = sum(
                    1 for f in ("revenue_cr", "pat_cr", "eps")
                    if results[0].get(f) is not None
                )
                if numeric_count >= 2:
                    print(
                        f"    [chunk] P&L sufficient ({numeric_count}/3 key fields)"
                        f" — skipping remaining sections",
                        flush=True,
                    )
                    break

        if not results:
            return None

        merged = _merge_financial_dicts(results)
        if len(results) > 1:
            print(f"    [chunk] merged {len(results)} section extractions", flush=True)
        return merged

    # ── pdfplumber PDF → text ─────────────────────────────────────────────────

    def _pdf_to_text(self, pdf_bytes: bytes) -> str:
        """Extract plain text from PDF bytes using pdfplumber (no ML, no OCR)."""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                text = "\n\n".join(pages)
            print(f"    [pdfplumber] extracted {len(text)} chars from {len(pdf.pages)} pages", flush=True)
            return text
        except Exception as e:
            print(f"    [pdfplumber] FAILED: {e}", flush=True)
            return ""

    # ── Ollama JSON call ──────────────────────────────────────────────────────

    def _ollama_json(self, prompt: str) -> dict | None:
        """Call Ollama with JSON mode and return the parsed dict, or None on error."""
        try:
            import ollama
            response = ollama.chat(
                model=_OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_gpu": 99},
            )
            try:
                raw = response.message.content
            except AttributeError:
                raw = response["message"]["content"]
            print(f"    [ollama] response {len(raw)} chars", flush=True)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    [ollama] JSON parse error: {e}", flush=True)
            return None
        except Exception as e:
            print(f"    [ollama] ERROR: {e}", flush=True)
            return None

    # ── Ollama calls with Pydantic validation ────────────────────────────────

    def _ollama_json_validated(self, prompt: str, model_cls: type) -> dict | None:
        """
        Call ollama (JSON mode) → parse → validate with Pydantic model → return dict.
        Pydantic coerces types and normalises nulls, eliminating bad-field errors
        even when Ollama returns a float field as a string, etc.
        """
        raw = self._ollama_json(prompt)
        if raw is None:
            return None
        try:
            validated = model_cls.model_validate(raw)
            return validated.model_dump()
        except ValidationError as e:
            print(f"    [pydantic] validation warning (using raw): {e.error_count()} errors", flush=True)
            return raw  # fall back to raw dict if validation fails — better than None

    _MAX_FINANCIAL_CHARS = 8_000  # llama3 context limit guard

    def _call_ollama_financial(self, text: str, symbol: str, company: str, broadcast_dt: str) -> dict | None:
        if len(text) > self._MAX_FINANCIAL_CHARS:
            # Keep first 6 000 chars (header / revenue line) + last 2 000 (P&L table footer)
            text = text[:6_000] + "\n...[truncated]...\n" + text[-2_000:]
            print(f"    [ollama] text truncated to {len(text)} chars for {symbol}", flush=True)
        return self._call_ollama_financial_traced(text, symbol, company, broadcast_dt)

    def _call_ollama_financial_traced(
        self, text: str, symbol: str, company: str, broadcast_dt: str
    ) -> dict | None:
        # Wrapped separately so @traceable can be applied conditionally at import time
        prompt = _FINANCIAL_PROMPT.format(
            company=company, symbol=symbol, broadcast_dt=broadcast_dt, text=text,
        )
        result = self._ollama_json_validated(prompt, FinancialExtraction)
        if result is not None:
            result = self._apply_confidence(result, symbol)
        return result

    # Numeric values that appeared verbatim in previous prompt examples.
    # These exact values should never appear in real financial data; if Ollama
    # returns them it is echoing the old example rather than extracting the document.
    _PROMPT_FINGERPRINTS = frozenset({
        1234.5, 123.4, 234.5, 189.3,          # original prompt examples (v1)
        1266.42, 106.18,                        # second prompt iteration examples (v2)
    })

    def _apply_confidence(self, data: dict, symbol: str) -> dict:
        """Count non-null numeric fields; flag hallucination fingerprints; log if low."""
        _NUMERIC_FIELDS = ("revenue_cr", "pat_cr", "ebitda_cr", "eps")
        filled = sum(1 for f in _NUMERIC_FIELDS if data.get(f) is not None)

        # Detect fingerprint hallucination: value matches a known prompt-example number
        fingerprint_hit = any(
            data.get(f) in self._PROMPT_FINGERPRINTS
            for f in _NUMERIC_FIELDS
        )
        if fingerprint_hit:
            _log.warning("[%s] hallucination fingerprint detected — overriding confidence to low", symbol)
            data["confidence"] = "low"
            summary = data.get("raw_summary") or ""
            if not summary.startswith("[LOW CONFIDENCE]"):
                data["raw_summary"] = f"[LOW CONFIDENCE] {summary}"
            return data

        # Derive confidence from field count, but don't blindly trust model's self-report
        if filled >= 3:
            conf = "high"
        elif filled >= 1:
            conf = "medium"
        else:
            conf = "low"

        data["confidence"] = conf

        if conf == "low":
            _log.warning("[%s] low-confidence extraction — only %d/%d numeric fields filled",
                         symbol, filled, len(_NUMERIC_FIELDS))
            summary = data.get("raw_summary") or ""
            if not summary.startswith("[LOW CONFIDENCE]"):
                data["raw_summary"] = f"[LOW CONFIDENCE] {summary}"

        return data

    def _call_ollama_general(self, text: str, symbol: str, company: str, broadcast_dt: str, extraction_type: str) -> dict | None:
        instructions = _INSTRUCTIONS.get(extraction_type, _INSTRUCTIONS["order_win"])
        prompt = _GENERAL_PROMPT.format(
            instructions=instructions, company=company, symbol=symbol,
            broadcast_dt=broadcast_dt, text=text,
        )
        model_cls = OrderWinExtraction if extraction_type == "order_win" else GeneralExtraction
        return self._ollama_json_validated(prompt, model_cls)

    def _call_ollama_concall(self, text: str, symbol: str, company: str, broadcast_dt: str) -> dict | None:
        prompt = _CONCALL_PROMPT.format(
            company=company, symbol=symbol, broadcast_dt=broadcast_dt, text=text,
        )
        return self._ollama_json_validated(prompt, ConcallExtraction)

    # ── Build FinancialResult from parsed dict ─────────────────────────────────

    # ── Vision fallback (image-based / scanned PDFs) ─────────────────────────

    def _vision_fallback(
        self,
        pdf_bytes: bytes,
        symbol: str,
        company: str,
        broadcast_dt: str,
        source_url: str,
        extraction_type: str,
    ) -> dict | None:
        """Try Gemini Flash vision when pdfplumber found < 80 chars of text."""
        print(f"    [vision] image-based PDF — trying Gemini Flash", flush=True)
        from financial.vision_extractor import extract_with_vision
        return extract_with_vision(
            pdf_bytes, symbol, company, broadcast_dt, source_url, extraction_type
        )

    def _build_result(
        self,
        data: dict,
        symbol: str,
        company: str,
        broadcast_dt: str,
        source_url: str,
        extraction_type: str,
    ) -> FinancialResult:
        def _float(key: str) -> float | None:
            v = data.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        period_norm = _normalize_period(str(data.get("period") or broadcast_dt[:10]))

        # Compute revenue growth from prior period when not stated in document
        rev_growth = _float("revenue_growth_pct")
        if rev_growth is None:
            rev       = _float("revenue_cr")
            prior_rev = _float("prior_revenue_cr")
            if rev is not None and prior_rev and prior_rev != 0:
                rev_growth = round((rev - prior_rev) / prior_rev * 100, 1)

        # Compute EBITDA from P&L components when not labelled in document
        ebitda = _float("ebitda_cr")
        if ebitda is None:
            pbt = _float("pbt_cr")
            dep = _float("depreciation_cr")
            fin = _float("finance_costs_cr")
            if pbt is not None and dep is not None and fin is not None:
                ebitda = round(pbt + dep + fin, 2)

        # Sanity checks: nullify values that are physically impossible (unit errors)
        rev    = _float("revenue_cr")
        pat    = _float("pat_cr")
        eps_v  = _float("eps")
        ebm    = _float("ebitda_margin_pct")
        conf   = data.get("confidence") or None

        suspicious = []
        if rev is not None and rev > 500_000:
            suspicious.append(f"revenue_cr={rev} exceeds max possible Indian company revenue")
            rev = None
        if pat is not None and rev is not None and pat > rev and extraction_type == "financial_results":
            suspicious.append(f"pat_cr={pat} > revenue_cr={rev}")
            pat = None
        if ebm is not None and (ebm > 95 or ebm < -200):
            suspicious.append(f"ebitda_margin_pct={ebm} outside valid range")
            ebm = None
        if eps_v is not None and abs(eps_v) > 100_000:
            suspicious.append(f"eps={eps_v} likely unit error")
            eps_v = None
        if suspicious:
            conf = "low"
            print(f"    [sanity] {symbol}: {'; '.join(suspicious)}", flush=True)

        # Order value — use JSON field if present, else parse from first highlight
        _order_value_cr = _float("order_value_cr")
        if _order_value_cr is None and extraction_type in ("order_win", "press_release"):
            for _hl in (data.get("key_highlights") or []):
                _hm = re.search(
                    r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d+)?)\s*Cr', str(_hl), re.IGNORECASE
                )
                if _hm:
                    try:
                        _order_value_cr = float(_hm.group(1).replace(",", ""))
                    except ValueError:
                        pass
                    break

        return FinancialResult(
            symbol             = symbol,
            company            = company,
            period             = period_norm,
            period_type        = extraction_type,
            source_url         = source_url,
            broadcast_dt       = broadcast_dt,
            revenue_cr         = rev,
            revenue_growth_pct = rev_growth,
            ebitda_cr          = ebitda,
            ebitda_margin_pct  = ebm,
            pat_cr             = pat,
            pat_growth_pct     = _float("pat_growth_pct"),
            eps                = eps_v,
            dividend_per_share = _float("dividend_per_share"),
            key_highlights     = data.get("key_highlights") or [],
            guidance           = data.get("guidance") or None,
            raw_summary        = str(data.get("raw_summary") or ""),
            confidence         = conf,
            order_value_cr     = _order_value_cr,
            client_name        = data.get("client_name") or None,
            client_type        = data.get("client_type") or None,
            order_sector       = data.get("order_sector") or None,
            execution_months   = int(data["execution_months"]) if data.get("execution_months") else None,
        )


# Apply LangSmith tracing to the financial extraction method (no-op when tracing is off)
LocalExtractor._call_ollama_financial_traced = _maybe_traceable(
    LocalExtractor._call_ollama_financial_traced
)
