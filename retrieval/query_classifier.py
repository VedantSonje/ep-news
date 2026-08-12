"""
QueryClassifier — deterministic intent detection for EP news queries.

Routing logic (no LLM):
  STRUCTURED  -> SQL only    — specific symbols, dates, score thresholds
  SEMANTIC    -> vector only — thematic/descriptive, no structured filters
  HYBRID      -> BM25 + vec + RRF fusion — default for natural-language queries
  FINANCIAL   -> financial_results table — revenue/PAT/EPS/growth queries
  COMPOUND    -> hybrid + financial join — "orders AND results" queries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(Enum):
    STRUCTURED = "sql_only"
    SEMANTIC   = "vector_only"
    HYBRID     = "hybrid"
    FINANCIAL  = "financial"
    COMPOUND   = "compound"


@dataclass
class ParsedQuery:
    intent:           QueryIntent
    symbols:          list[str]   = field(default_factory=list)
    date_filter:      str | None  = None   # ISO date "YYYY-MM-DD"
    min_score:        int | None  = None
    order_value_min:  float | None = None  # "above Rs.100 Cr" → 100.0
    themes:           list[str]   = field(default_factory=list)
    raw:              str         = ""


# ── Extraction patterns ───────────────────────────────────────────────────────

_SYMBOL = re.compile(r"\b([A-Z]{2,10})\b")

_DATE_TODAY    = re.compile(r"\btoday\b", re.I)
_DATE_EXPLICIT = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_DMY      = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")

_MIN_SCORE = re.compile(
    r"\b(?:score|ep\s+score)\s*(?:>=?|above|over|min(?:imum)?)\s*(\d+)", re.I
)
_SCORE_HIGH = re.compile(r"\b(?:high|top|best)\s+(?:score[sd]?|rated)\b", re.I)

_ORDER_VALUE_RE = re.compile(
    r"(?:above|over|more\s+than|greater\s+than|>=?|atleast|at\s+least)\s*"
    r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d,]+(?:\.\d+)?)\s*"
    r"(crore|cr\.?|lakh|lakhs|million|mn|thousand)?",
    re.I,
)

_THEME_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdefence|defense|military\b", re.I),                "defence"),
    (re.compile(r"\brailway|rail|metro|train\b", re.I),                "railway"),
    (re.compile(r"\bsolar|renewable|wind\s+energy|BESS\b", re.I),     "solar_renewables"),
    (re.compile(r"\bpower|electricity|transmission\b", re.I),          "power"),
    (re.compile(r"\binfrastructure|EPC|highway|road|bridge\b", re.I), "infrastructure"),
    (re.compile(r"\bpharma|drug|FDA|ANDA|biosimilar\b", re.I),        "pharma_healthcare"),
    (re.compile(r"\bsoftware|digital|cloud|SaaS\b", re.I),            "IT_technology"),
    (re.compile(r"\bPSU|public\s+sector|government\s+contract\b", re.I), "PSU_government"),
    (re.compile(r"\bPLI|Make\s+in\s+India|semiconductor\b", re.I),    "PLI_manufacturing"),
    (re.compile(r"\bcapital\s+goods|engineering|industrial\b", re.I), "capital_goods"),
    (re.compile(r"\breal\s+estate|realty|housing\b", re.I),           "real_estate"),
    (re.compile(r"\bbank|NBFC|microfinance|AUM\b", re.I),             "banking_finance"),
    (re.compile(r"\bFMCG|consumer\s+goods|beverage\b", re.I),         "FMCG_consumer"),
]

_FINANCIAL_SIGNALS = re.compile(
    r"\b(revenue|PAT|EBITDA|EPS|profit|quarterly\s+result|"
    r"annual\s+result|earnings|growth|margin|guidance|dividend)\b",
    re.I,
)
_STRUCTURED_SIGNALS = re.compile(
    r"\b(score|ep\s+score|today|yesterday|\d{4}-\d{2}-\d{2}|"
    r"top\s+\d+|largest|biggest|highest\s+score)\b",
    re.I,
)
_SEMANTIC_SIGNALS = re.compile(
    r"\b(similar\s+to|like|related|about|theme|sector|"
    r"companies\s+(that|which)|stocks\s+(with|having))\b",
    re.I,
)

_NON_SYMBOLS = frozenset({
    "I", "A", "AT", "IN", "OF", "OR", "AND", "THE", "FOR", "BY",
    "TO", "DO", "IS", "IT", "BE", "ON", "NO", "SO", "UP", "AS",
    "EP", "IPO", "NSE", "BSE", "PAT", "NPA", "AUM", "API", "EPC",
    "LTD", "PVT", "CO", "INC", "PLC", "YOY", "QOQ", "TTM",
    "FY", "Q1", "Q2", "Q3", "Q4", "FROM", "WITH", "THAT", "THIS",
    "ALL", "ANY", "NOT", "TOP", "GET", "CAN", "YOU", "ME", "MY",
})


def _extract_symbols(text: str) -> list[str]:
    return [m.group(1) for m in _SYMBOL.finditer(text) if m.group(1) not in _NON_SYMBOLS]


def _extract_date(text: str) -> str | None:
    if _DATE_TODAY.search(text):
        from datetime import date
        return date.today().isoformat()
    m = _DATE_EXPLICIT.search(text)
    if m:
        return m.group(1)
    m = _DATE_DMY.search(text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    return None


def _extract_min_score(text: str) -> int | None:
    m = _MIN_SCORE.search(text)
    if m:
        return int(m.group(1))
    if _SCORE_HIGH.search(text):
        return 8
    return None


def _extract_order_value_min(text: str) -> float | None:
    m = _ORDER_VALUE_RE.search(text)
    if not m:
        return None
    val  = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "cr").lower().strip(".")
    if "lakh" in unit:
        val /= 100.0
    elif unit in ("million", "mn"):
        val /= 10.0
    elif "thousand" in unit:
        val /= 1_000.0
    return val


def _extract_themes(text: str) -> list[str]:
    return [label for pattern, label in _THEME_MAP if pattern.search(text)]


class QueryClassifier:
    """
    Classifies a natural-language query into a routing intent and
    extracts structured filters. Pure Python — no LLM calls.
    """

    def classify(self, query: str) -> ParsedQuery:
        symbols   = _extract_symbols(query)
        date_f    = _extract_date(query)
        min_score = _extract_min_score(query)
        themes    = _extract_themes(query)

        order_val = _extract_order_value_min(query)

        fin   = bool(_FINANCIAL_SIGNALS.search(query))
        struc = bool(_STRUCTURED_SIGNALS.search(query)) or bool(symbols) or (date_f is not None)
        # themes are FILTERS not routing signals — don't force SEMANTIC (which drops BM25)
        sem   = bool(_SEMANTIC_SIGNALS.search(query))

        if fin and (struc or sem):
            intent = QueryIntent.COMPOUND
        elif fin:
            intent = QueryIntent.FINANCIAL
        elif struc and not sem:
            intent = QueryIntent.STRUCTURED
        elif sem and not struc:
            intent = QueryIntent.SEMANTIC
        else:
            intent = QueryIntent.HYBRID   # default — BM25 + vector + RRF

        return ParsedQuery(
            intent          = intent,
            symbols         = symbols,
            date_filter     = date_f,
            min_score       = min_score,
            order_value_min = order_val,
            themes          = themes,
            raw             = query,
        )
