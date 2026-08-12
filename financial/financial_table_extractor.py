"""
FinancialTableExtractor — extracts P&L metrics from quarterly results PDFs.

Two-tier approach:
  Tier 1  pdfplumber extract_tables() — fast, works for PDFs with visible grid lines
  Tier 2  Text-line parser             — works for borderless/light-border tables
           (covers ~95% of Indian BSE/NSE filings)

Both tiers look for the same row labels (Ind-AS Regulation 33 format):
  Revenue From Operations / Net Profit / EPS Basic / Finance Costs / Depreciation …

Unit declared in header: "Rs. in million" / "Rs. in lakhs" / "Rs. in crore" / "INR Million"
Current quarter = FIRST (leftmost) numeric data column.
Negative values: (3,900) means -3,900.
"""
from __future__ import annotations

import io
import re
from financial.models import FinancialResult


# ── Unit detection ─────────────────────────────────────────────────────────────

_UNIT_RE = re.compile(
    r"(?:Rs\.?|₹|INR|Rupees?)\s*[.,]?\s*in\s*(million|mn|lakh|lakhs?|lacs?|crore|cr\.?)"
    r"|All\s+amounts\s+are\s+in\s+INR\s+(million|lakh|lacs?|crore)"
    # "(~ in crore)" / "(in crore)" / "(Rs. in crore)" / "(Amount in million ...)"
    r"|\([^)]*\bin\s+(million|mn|lakh|lacs?|lakhs?|crore|cr\.?)\b[^)]*\)"
    r"|(?:Rs\.?|₹|INR)\s+(million|lakh|lacs?|crore)"
    # Standalone header like "in Lakhs" or "in Crore" at top of P&L page
    r"|\bin\s+(lakh|lakhs?|lacs?|crore|cr\.?|million|mn)\b",
    re.IGNORECASE,
)
_UNIT_MULT: dict[str, float] = {
    "million": 0.1, "mn": 0.1,
    "lakh": 0.01,   "lakhs": 0.01,
    "lac":  0.01,   "lacs":  0.01,   # alternate Indian spelling
    "crore": 1.0,   "cr": 1.0,
}


# Indian lakh number format: "X,XX,XXX" — groups of 2 after the initial digit(s),
# then 3 at the end (e.g. "2,94,270" or "13,00,631"). Distinct from western thousands
# ("294,270") which use groups of 3. Requires at least 2 comma groups so single
# thousands like "12,345" are not misidentified.
_INDIAN_LAKH_NUM_RE = re.compile(r'\b\d{1,2},\d{2},\d{3}(?:\.\d+)?\b')


def _detect_unit(text: str) -> float:
    m = _UNIT_RE.search(text)
    if m:
        unit = next((g for g in m.groups() if g), "crore").lower().strip(".")
        return _UNIT_MULT.get(unit, 1.0)
    # Heuristic: if the page has 7+ numbers in Indian lakh format (X,XX,XXX)
    # and no explicit unit declaration, the values are almost certainly in lakhs.
    # Threshold=7 avoids false positives for large-caps whose annual column totals
    # (4-6 values) happen to be in Indian lakh format even though the unit is crore.
    if len(_INDIAN_LAKH_NUM_RE.findall(text)) >= 7:
        return 0.01
    return 1.0


# ── Number parsing ─────────────────────────────────────────────────────────────

_NUM_PARENS = re.compile(r'\((\d[\d,]*(?:\.\d+)?)\)')          # (3,900) → negative
_NUM_PLAIN  = re.compile(r'(?<![(\d])-?(\d[\d,]*(?:\.\d+)?)')  # 3,900.50

def _parse_num(cell: str | None) -> float | None:
    """Parse Indian-format number from a table cell. None for blank/dash."""
    if not cell:
        return None
    s = str(cell).strip()
    if s in ("", "-", "--", "—", "Nil", "NIL", "NA", "N/A", "Not Applicable"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # Remove commas (thousands separator). Some Indian PDFs use spaces as digit
    # group separators: "2 248 39" = 2,248.39 Cr.
    # Rule: if the LAST space-separated group has exactly 2 digits, treat it as
    # the decimal part; join all prior groups as the integer part.
    # Examples: "412 72"→412.72, "2 248 39"→2248.39, "1 234"→1234 (3 digits→no decimal)
    s = s.replace(",", "")
    parts = s.strip().split()
    if len(parts) >= 2 and re.fullmatch(r'\d{2}', parts[-1]):
        integer_part = "".join(parts[:-1])
        s = f"{integer_part}.{parts[-1]}"
    else:
        s = s.replace(" ", "")
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


def _first_number_after(text: str) -> float | None:
    """
    Return the first meaningful number from `text` (the part of a line
    after the row-label keyword).

    Skips small bare integers (1–99 without comma/decimal) which are
    row-serial-numbers, not financial values.
    """
    for m in re.finditer(
        r'\((\d[\d,]*(?:\.\d+)?)\)|(?<!\d)(-?\d[\d,]*(?:\.\d+)?)',
        text
    ):
        neg_s = m.group(1)
        pos_s = m.group(2)
        if neg_s:
            try:
                return -float(neg_s.replace(",", ""))
            except ValueError:
                continue
        elif pos_s:
            try:
                val = float(pos_s.replace(",", ""))
                # skip integers 0-99 with no comma/dot → serial numbers
                if "," not in pos_s and "." not in pos_s and 0 <= val < 100:
                    continue
                return val
            except ValueError:
                continue
    return None


# ── Row label matchers ────────────────────────────────────────────────────────

_FIELD_ROWS: dict[str, list[str]] = {
    "revenue": [
        "total revenue from operations",   # catches "(I) Total revenue from operations"
        "revenue from operations",
        "revenue from operation",
        "net sales",
        "net revenue from operations",
        "sales and services",
        "income from operations",
        "turnover",
        "revenue from contracts with customers",  # real estate / Ind-AS 115
        "gross premium",                           # insurance
        "premium earned",
        "net revenue",
        "income from revenue",                     # non-standard label
    ],
    # Banks/NBFCs — kept separate so sub-items don't shadow the P&L total line.
    # _build_result uses this only when "revenue" and "total_income" are both absent.
    "interest_income": [
        "net interest income",
        "interest income",
        "interest earned",
    ],
    "total_income": [
        "total income",
        "total revenue",
        "total income from operations",
    ],
    "ebitda": [
        "ebitda",
        "operating profit",
        "profit from operations before finance",
        "earnings before interest",
    ],
    "pbt": [
        "profit before tax",
        "profit/(loss) before tax",
        "profit / (loss) before tax",
        "profit before exceptional",
        "loss before tax",
        "loss/(profit) before tax",
        "profit/(loss) before exceptional items and tax",
        "profit before exceptional items and tax",
        "loss before exceptional items and tax",
    ],
    "pat": [
        "net profit for the period",
        "net profit for the year",
        "profit for the period",
        "profit for the year",
        "profit/(loss) for the period",
        "profit/(loss) for the year",
        "profit after tax",
        "net profit after tax",
        "loss for the period",
        "loss for the year",
        # OCR variants: pdfplumber sometimes drops the leading 'p' from "period"
        "profit for the eriod",    # "period" → "eriod"
        "profit for the ear",      # "year"   → "ear"
        "net profit for the eriod",
        "net profit for the ear",
        "net profit",
        "net loss",
    ],
    "eps_basic": [
        "earnings per share",
        "basic earnings per share",
        "eps - basic",
        "eps (basic)",
        "basic (in rs.)",
        "basic",
    ],
    "eps_diluted": [
        "diluted earnings per share",
        "diluted (in rs.)",
        "diluted",
    ],
    "finance_cost": [
        "finance costs",
        "finance cost",
        "interest expense",
        "interest and finance charges",
        "borrowing costs",
    ],
    "depreciation": [
        "depreciation and amortisation",
        "depreciation and amortization",
        "depreciation & amortisation",
        "depreciation",
        "amortisation",
        "amortization",
    ],
}


_DATE_CONTEXT_RE = re.compile(
    r"\b(march|june|sept|dec|jan|feb|apr|may|jul|aug|oct|nov|fy|quarter|ended|period)\b",
    re.IGNORECASE,
)

# OCR-resilient regex for PAT: handles common substitutions like ")" → "l", "p" → "n"
# Matches "profit/(loss) for the period/year" and OCR variants like "profit/(lossl for the neriod"
_PAT_OCR_FALLBACK_RE = re.compile(
    r'\bprofit[/(]\(?l[o0]ss[l)]\)?\s+for\s+the\s+[pn]eri[o0]d\b'
    r'|\bprofit[/(]\(?l[o0]ss[l)]\)?\s+for\s+the\s+[vy]ear\b',
    re.IGNORECASE,
)


_STANDALONE_RE   = re.compile(r'\bstandalone\b',   re.IGNORECASE)
_CONSOLIDATED_RE = re.compile(r'\bconsolidated\b', re.IGNORECASE)

# Pages that discuss financials but are NOT financial result tables — auditor's reports,
# review reports, and certificates mention "profit" and "income" but contain no actual
# numbers we want to extract.
_AUDIT_PAGE_RE = re.compile(
    r"INDEPENDENT\s+AUDITOR[‘’S]*\s+REPORT"
    r"|LIMITED\s+REVIEW\s+REPORT"
    r"|AUDITOR[‘’S]*\s+CERTIFICATE"
    r"|INDEPENDENT\s+REVIEW\s+REPORT",
    re.IGNORECASE,
)
# Investor KPI summary pages (e.g. L&T "Key Parameters") have reversed column order
# (prior year first, current quarter second) and must be skipped to avoid picking
# the wrong column as current-quarter revenue.
_SUMMARY_PAGE_RE = re.compile(
    r"\bKey\s+Parameters?\b"
    r"|\bGroup\s+Performance\b"
    r"|\bBusiness\s+Performance\b"
    r"|\bPerformance\s+Highlights?\b",
    re.IGNORECASE,
)


def _detect_statement_type(text: str) -> str:
    """Return 'standalone', 'consolidated', or 'unknown' from page heading text."""
    snippet = text[:2000]
    has_sa = bool(_STANDALONE_RE.search(snippet))
    has_co = bool(_CONSOLIDATED_RE.search(snippet))
    if has_sa and not has_co:
        return "standalone"
    if has_co and not has_sa:
        return "consolidated"
    if has_sa and has_co:
        pos_sa = _STANDALONE_RE.search(snippet).start()
        pos_co = _CONSOLIDATED_RE.search(snippet).start()
        return "standalone" if pos_sa < pos_co else "consolidated"
    return "unknown"


def _matches(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in text, also handling space-less OCR variants."""
    t = text.lower().strip()
    t_nospace = re.sub(r'\s+', '', t)
    # OCR-clean: "Pr" → "Pl'" is a common Indian PDF OCR error (e.g. "Pl'ofit" = "Profit")
    t_ocr = re.sub(r"l['’`]", "r", t)
    return any(
        kw in t or kw.replace(' ', '') in t_nospace or kw in t_ocr
        for kw in keywords
    )


# ── Period helpers ────────────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r"(\d{1,2})[.\-/\s]+(\w+)[.\-/\s]+(\d{4})"   # 30 June 2026 / 31-03-2026
    r"|(\w+)\s+(\d{1,2}),?\s*(\d{4})",              # June 30, 2026 / March 31, 2026
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _quarter_from_month_year(month: int, year: int) -> str:
    """Convert quarter-end month/year to 'Q1 FY2027' style."""
    if month == 3:
        return f"Q4 FY{year}"
    elif month == 6:
        return f"Q1 FY{year + 1}"
    elif month == 9:
        return f"Q2 FY{year + 1}"
    elif month == 12:
        return f"Q3 FY{year + 1}"
    return f"FY{year}"


def _extract_period_from_text(text: str) -> str:
    """Try to pull the reporting period from document text."""
    # Look for "quarter ended" / "year ended" near a date
    pattern = re.compile(
        r"(?:quarter|year|half[- ]year)\s+ended(?:\s+on)?\s+([^\n]{5,35})",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text[:3000]):
        raw = m.group(1).strip()
        dm = _DATE_RE.search(raw)
        if dm:
            try:
                if dm.group(1):  # DD MMM YYYY
                    d, mo_s, yr = dm.group(1), dm.group(2), dm.group(3)
                    month = _MONTH_MAP.get(mo_s[:3].lower(), 0)
                else:             # Month DD, YYYY
                    mo_s, d, yr = dm.group(4), dm.group(5), dm.group(6)
                    month = _MONTH_MAP.get(mo_s[:3].lower(), 0)
                if month:
                    q = _quarter_from_month_year(month, int(yr))
                    if "year" in m.group(0).lower() and "quarter" not in m.group(0).lower():
                        return f"FY{yr}"
                    return q
            except (ValueError, TypeError):
                pass
    return ""


def _infer_period_type(period: str) -> str:
    p = period.lower()
    if any(w in p for w in ["fy", "year", "annual", "march", "31-mar"]):
        if "q" not in p:
            return "annual"
    if any(w in p for w in ["h1", "h2", "half"]):
        return "half-yearly"
    return "quarterly"


# ── Tier 1: table-based extraction ────────────────────────────────────────────

_PL_DATE_RE = re.compile(
    r"\b(3[01])[.\-/](0[1-9]|1[0-2])[.\-/](20\d{2})\b"   # 30.06.2026 / 31-03-2026
    r"|\b(march|june|sept|dec|quarter|ended|preceding|corresponding)\b",
    re.IGNORECASE,
)


def _is_pl_table(table: list[list]) -> bool:
    if not table or len(table) < 4:
        return False
    header_text = " ".join(
        str(cell).lower()
        for row in table[:5]
        for cell in (row or [])
        if cell
    )
    body_text = " ".join(
        str(cell).lower()
        for row in table[:6]
        for cell in (row or [])
        if cell
    )
    # Require a date or period keyword in the header rows — prevents picking up
    # notes-to-accounts or segment tables which have "revenue" but no period header.
    has_period   = bool(_PL_DATE_RE.search(header_text))
    has_structure = "particular" in body_text or (
        "ebitda" in body_text and any(kw in body_text for kw in ["revenue", "profit", "pat"])
    )
    has_financial = any(kw in body_text for kw in [
        "revenue", "income", "profit", "loss", "sales", "expenditure", "ebitda"
    ])
    return has_period and has_structure and has_financial


def _find_data_col(table: list[list]) -> int:
    date_re = re.compile(r"\d{2}[.\-/]\w+[.\-/]?\d{2,4}|\bjun\b|\bmar\b|\bsep\b|\bdec\b", re.I)
    for row in table[:5]:
        for i, cell in enumerate(row or []):
            if cell and i > 0 and date_re.search(str(cell)):
                return i
    for row in table[3:8]:
        for i, cell in enumerate(row or []):
            if i > 0 and _parse_num(cell) is not None:
                return i
    return 2


def _get_period_from_table(table: list[list], data_col: int) -> str:
    period_re = re.compile(
        r"(\d{1,2}[\s\-./]\w+[\s\-./]\d{2,4}"
        r"|\w+\s+\d{4}"
        r"|Q[1-4]\s*FY[\s']*\d{2,4}"   # handles "Q1 FY'27" with apostrophe
        r"|\d{4}[-/]\d{2}[-/]\d{2})",
        re.IGNORECASE,
    )
    for row in table[:5]:
        if not row or data_col >= len(row):
            continue
        m = period_re.search(str(row[data_col] or ""))
        if m:
            return m.group(0).strip()
    return ""


def _extract_from_table(pl_table: list[list], unit_mult: float, period: str, broadcast_dt: str) -> dict:
    """Pull field values from a detected pdfplumber table."""
    data_col = _find_data_col(pl_table)
    if not period:
        period = _get_period_from_table(pl_table, data_col)

    # Detect label column: standard NSE has serial# in col 0 and label in col 1;
    # management summary tables have label in col 0 and first value in col 1.
    # Heuristic: if data_col == 1, labels must be in col 0.
    label_col = 0 if data_col <= 1 else 1

    # Sparsity check: pdfplumber sometimes merges all quarterly columns into a
    # single cell (seen in BAJFINANCE). When ≥55% of data rows have an empty
    # label cell, the table structure is unusable — return empty so Tier 2
    # text parsing runs instead.
    data_rows = [r for r in pl_table if r and len(r) > label_col]
    empty_labels = sum(1 for r in data_rows if not str(r[label_col] or "").strip())
    if len(data_rows) > 4 and empty_labels / len(data_rows) >= 0.55:
        return {"fields": {}, "period": period, "unit_mult": unit_mult}

    # Merged-cell check: pdfplumber sometimes concatenates ALL row labels into a
    # single cell for borderless tables (e.g. RALLIS). Any label with 3+ newlines
    # means multiple rows were merged — bail out so Tier 2 text-parse runs instead.
    if any(str(r[label_col] or "").count("\n") >= 3
           for r in data_rows if len(r) > label_col):
        return {"fields": {}, "period": period, "unit_mult": unit_mult}

    extracted: dict[str, float | None] = {}
    for row in pl_table:
        if not row or len(row) <= data_col:
            continue
        raw_label = str(row[label_col] or "")
        # Strip trailing asterisks and footnote markers common in management tables
        particulars = re.sub(r'[\*#†]+$', '', raw_label).strip()
        value_raw   = row[data_col]
        for field, keywords in _FIELD_ROWS.items():
            if field not in extracted and _matches(particulars, keywords):
                extracted[field] = _parse_num(value_raw)
        # OCR-resilient PAT fallback: handles garbled labels like "Profit/(Lossl for the neriod"
        if "pat" not in extracted and _PAT_OCR_FALLBACK_RE.search(particulars):
            extracted["pat"] = _parse_num(value_raw)
        # Exact-abbreviation match for management summary tables (col 0 = "Revenue"/"PAT"/"EBITDA")
        lp = particulars.lower()
        if lp in ("revenue",) and "revenue" not in extracted:
            extracted["revenue"] = _parse_num(value_raw)
        elif lp in ("pat",) and "pat" not in extracted:
            extracted["pat"] = _parse_num(value_raw)
        elif lp in ("ebitda",) and "ebitda" not in extracted:
            extracted["ebitda"] = _parse_num(value_raw)
    return {"fields": extracted, "period": period, "unit_mult": unit_mult}


# ── Tier 2: text-line extraction ──────────────────────────────────────────────

def _is_pl_page(text: str) -> bool:
    """Quick check: does this page contain a P&L table?"""
    tl = text.lower()
    has_particulars = "particular" in tl or "revenue from operation" in tl
    has_financial   = any(kw in tl for kw in [
        "revenue from operations", "net profit", "profit before tax",
        "profit after tax", "net sales", "total income",
    ])
    return has_particulars or has_financial


def _extract_from_text(page_text: str) -> dict[str, float | None]:
    """
    Parse P&L fields from plain text (Tier 2 fallback).
    Scans each line for keyword match, then extracts the first number
    in the text AFTER the matched keyword (= current quarter value).
    """
    extracted: dict[str, float | None] = {}
    lines = page_text.split("\n")

    for i, line in enumerate(lines):
        line_lower = line.lower()

        line_nospace = re.sub(r'\s+', '', line_lower)
        # OCR-clean: replace "l'" with "r" (Indian PDF OCR error: "Pr" → "Pl'")
        line_ocr = re.sub(r"l[''`]", "r", line_lower)

        for field, keywords in _FIELD_ROWS.items():
            if field in extracted:
                continue

            # Find the keyword in this line (also try space-less and OCR-clean variants)
            kw_end = -1
            for kw in keywords:
                pos = line_lower.find(kw)
                if pos >= 0:
                    kw_end = pos + len(kw)
                    break
                # Space-less fallback: "REVENUEFROMOPERATIONS" → "revenue from operations"
                kw_ns = kw.replace(' ', '')
                pos_ns = line_nospace.find(kw_ns)
                if pos_ns >= 0:
                    kw_end = 0
                    break
                # OCR-clean fallback: "Pl'ofit after tax" → "profit after tax"
                if line_ocr != line_lower and kw in line_ocr:
                    kw_end = 0
                    break

            if kw_end < 0:
                continue

            # Text after the keyword on the same line (kw_end=0 → entire line)
            remainder = line[kw_end:]
            val = _first_number_after(remainder)

            # Reject year-like values when they appear in a date-context line.
            # e.g. "Profit for the year ended June 30, 2026" → 2026 is a calendar
            # year, not a financial value. We check the WHOLE line for date words.
            if val is not None and 2020.0 <= val <= 2030.0:
                if _DATE_CONTEXT_RE.search(line):
                    val = None

            # Reject values from comparison-summary lines that inline growth rates
            # like "Net Profit 819 809 1.3% 919 -10.9%". Standard Reg 33 P&L never
            # has inline percentage changes — this pattern means a USD/USD-INR dual
            # table (common in large-cap filings like INFY). Rejecting lets the next
            # occurrence of the same keyword (INR table) be used instead.
            # But NOT for press-release lines with parenthetical annotations like
            # "Net profit of INR 16.5 Crs. (net margin: 13.1%)" — strip parens first.
            if val is not None and re.search(r'\d+\.\d+%', remainder):
                remainder_no_parens = re.sub(r'\([^)]*\)', '', remainder)
                if re.search(r'\d+\.\d+%', remainder_no_parens):
                    val = None

            # If no number on this line, check the next line
            # But NOT if the next line is a sub-item like "(a) Interest income",
            # and NOT if the current line is itself a section header "(a) Revenue
            # from operations" — that header has sub-items below it, not a value.
            current_is_header = bool(re.match(r'^\s*\d*\s*\([a-zA-Z]\)', line))
            if val is None and i + 1 < len(lines) and not current_is_header:
                nxt = lines[i + 1].lstrip()
                # Detect sub-items: "(a) ...", "i. ...", "a) ...", "b. ..."
                is_sub_item = bool(re.match(r'^\([a-zA-Z0-9]{1,3}\)', nxt) or
                                   re.match(r'^[ivxlIVXL]+[\.\)]\s', nxt) or
                                   re.match(r'^[a-zA-Z]\)', nxt))
                if not is_sub_item:
                    nxt_line = lines[i + 1]
                    val = _first_number_after(nxt_line)
                    # Apply same year + comparison-table filters to next-line value
                    if val is not None and 2020.0 <= val <= 2030.0:
                        if _DATE_CONTEXT_RE.search(nxt_line):
                            val = None
                    if val is not None and re.search(r'\d+\.\d+%', nxt_line):
                        nxt_no_parens = re.sub(r'\([^)]*\)', '', nxt_line)
                        if re.search(r'\d+\.\d+%', nxt_no_parens):
                            val = None

            if val is not None:
                extracted[field] = val

    # OCR-resilient fallback: if PAT keyword matching failed (garbled text like
    # "Profit/(Lossl for the neriod"), try regex that tolerates ")" → "l" / "p" → "n"
    if "pat" not in extracted:
        for i, line in enumerate(lines):
            if _PAT_OCR_FALLBACK_RE.search(line):
                remainder = line[_PAT_OCR_FALLBACK_RE.search(line).end():]
                val = _first_number_after(remainder)
                if val is None and i + 1 < len(lines):
                    val = _first_number_after(lines[i + 1])
                if val is not None and not (2020.0 <= val <= 2030.0 and _DATE_CONTEXT_RE.search(line)):
                    extracted["pat"] = val
                break

    return extracted


# ── Result builder (shared) ───────────────────────────────────────────────────

def _build_result(
    extracted:    dict[str, float | None],
    unit_mult:    float,
    period:       str,
    symbol:       str,
    company:      str,
    broadcast_dt: str,
    source_url:   str,
    method:       str,
) -> FinancialResult | None:
    """Convert raw extracted dict → FinancialResult. Returns None if no usable data."""

    def to_cr(val: float | None) -> float | None:
        if val is None:
            return None
        r = round(val * unit_mult, 2)
        return r if r != 0 else None

    revenue_cr = to_cr(
        extracted.get("revenue")
        or extracted.get("total_income")
        or extracted.get("interest_income")  # banks/NBFCs last resort
    )
    pat_raw    = extracted.get("pat") or extracted.get("pbt")
    pat_cr     = to_cr(pat_raw)

    # EBITDA: explicit or calculated = PBT + finance costs + depreciation
    ebitda_cr = to_cr(extracted.get("ebitda"))
    if ebitda_cr is None:
        pbt = extracted.get("pbt")
        fin = extracted.get("finance_cost")
        dep = extracted.get("depreciation")
        if pbt is not None and (fin is not None or dep is not None):
            ebitda_cr = to_cr((pbt or 0) + (fin or 0) + (dep or 0))

    ebitda_margin: float | None = None
    if ebitda_cr and revenue_cr and revenue_cr > 0:
        m = round(ebitda_cr / revenue_cr * 100, 1)
        # Sanity-check: realistic EBITDA margins are -100% to +100%
        if -100.0 <= m <= 100.0:
            ebitda_margin = m

    # EPS — per share, don't scale with unit unless obviously wrong
    eps: float | None = None
    raw_eps = extracted.get("eps_basic") or extracted.get("eps_diluted")
    if raw_eps is not None:
        eps_scaled = raw_eps * unit_mult
        eps = raw_eps if abs(raw_eps) < 500 else eps_scaled

    dividend = extracted.get("dividend")

    if revenue_cr is None and pat_cr is None:
        return None  # no meaningful data — caller falls back to Ollama

    # Reject absurdly large values caused by PDFs reporting in absolute rupees
    # without a unit declaration (e.g. "55,82,57,842" treated as crore).
    # No Indian company has >5 lakh crore quarterly revenue.
    _MAX_CR = 5_00_000
    if (revenue_cr is not None and abs(revenue_cr) > _MAX_CR) or \
       (pat_cr is not None and abs(pat_cr) > _MAX_CR):
        return None

    if not period:
        period = broadcast_dt[:10]
    period_type = _infer_period_type(period)

    highlights: list[str] = []
    if revenue_cr:
        highlights.append(f"Revenue: Rs.{revenue_cr:,.1f} Cr")
    if pat_cr:
        highlights.append(f"PAT: Rs.{pat_cr:,.1f} Cr")
    if ebitda_margin:
        highlights.append(f"EBITDA margin: {ebitda_margin:.1f}%")
    if eps:
        highlights.append(f"EPS: Rs.{eps:.2f}")
    if dividend:
        highlights.append(f"Dividend: Rs.{dividend}/share")
    highlights.append(f"Period: {period} | Unit mult: {unit_mult} | Method: {method}")

    rev_s = f"Rs.{revenue_cr:,.0f}Cr revenue" if revenue_cr else ""
    pat_s = f"Rs.{pat_cr:,.0f}Cr PAT"         if pat_cr    else ""
    raw_summary = (
        f"{company} reported {', '.join(filter(None, [rev_s, pat_s]))} for {period}."
    )

    return FinancialResult(
        symbol             = symbol,
        company            = company,
        period             = period,
        period_type        = period_type,
        source_url         = source_url,
        broadcast_dt       = broadcast_dt,
        revenue_cr         = revenue_cr,
        ebitda_cr          = ebitda_cr,
        ebitda_margin_pct  = ebitda_margin,
        pat_cr             = pat_cr,
        eps                = eps,
        dividend_per_share = dividend,
        key_highlights     = highlights,
        raw_summary        = raw_summary,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def _open_pdf(pdf_bytes: bytes):
    """Open PDF bytes — pdfplumber primary (layout-accurate text), fitz fallback."""
    try:
        import pdfplumber
        return pdfplumber.open(io.BytesIO(pdf_bytes)), "pdfplumber"
    except ImportError:
        pass
    try:
        import fitz  # PyMuPDF
        return fitz.open(stream=pdf_bytes, filetype="pdf"), "fitz"
    except ImportError:
        return None, None


def _iter_pages(pdf, backend: str):
    """Yield (page_text, page_tables) for each page."""
    if backend == "fitz":
        for page in pdf:
            text = page.get_text("text") or ""
            tables = []
            try:
                finder = page.find_tables()
                tables = [t.extract() for t in finder.tables]
            except Exception:
                pass
            yield text, tables
    else:
        # pdfplumber
        for page in pdf.pages:
            text   = page.extract_text() or ""
            tables = page.extract_tables() or []
            yield text, tables


def extract_financial_table(
    pdf_bytes:    bytes,
    symbol:       str,
    company:      str,
    broadcast_dt: str,
    source_url:   str,
) -> FinancialResult | None:
    """
    Try Tier 1 (table grid detection) then Tier 2 (text-line parse).
    Uses PyMuPDF (fitz) for 5-10× faster PDF parsing; falls back to pdfplumber.
    Returns None only if neither tier finds usable P&L data.
    """
    pdf, backend = _open_pdf(pdf_bytes)
    if pdf is None:
        return None

    full_text: list[str] = []
    all_page_tables: list[list] = []   # tables per page
    # Per-page unit declaration — two tiers:
    #  explicit : _UNIT_RE matched → authoritative unit string in the text
    #  heuristic: no _UNIT_RE match but 3+ Indian-lakh-format numbers → inferred
    #
    # Document-level fallback uses FIRST EXPLICIT declaration only (first-wins).
    # Heuristic detections are used per-page, never as the document-level fallback.
    # Reason: large PDFs (INFY, 193 pages) may have pages with annual totals in
    # Indian lakh format BEFORE the explicit crore declaration, causing the heuristic
    # to fire erroneously if it fed the document-level fallback.
    page_explicit_decls:   list[float | None] = []
    page_heuristic_decls:  list[float | None] = []

    try:
        # ── Pre-pass: collect text + tables, detect per-page unit ─────────────
        for page_text, page_tables in _iter_pages(pdf, backend):
            full_text.append(page_text)
            all_page_tables.append(page_tables)
            _has_explicit = bool(_UNIT_RE.search(page_text))
            _raw_decl = _detect_unit(page_text)
            page_explicit_decls.append(_raw_decl if _has_explicit else None)
            page_heuristic_decls.append(_raw_decl if (not _has_explicit and _raw_decl != 1.0) else None)

        # Document-level fallback: first EXPLICIT declaration (crore/lakh/million).
        # Lakh heuristic is intentionally excluded here — applied per-page only.
        unit_mult = next((u for u in page_explicit_decls if u is not None), 1.0)

        def _page_unit(idx: int) -> float:
            """Effective unit for page idx: explicit > heuristic > document fallback."""
            if page_explicit_decls[idx] is not None:
                return page_explicit_decls[idx]
            if page_heuristic_decls[idx] is not None:
                return page_heuristic_decls[idx]
            return unit_mult

        # Extract period from full text — used as fallback in Tier 1
        text_period = _extract_period_from_text("\n".join(full_text))

        # Collect first valid result per statement type: standalone > unknown > consolidated
        candidates: dict[str, FinancialResult] = {}

        # ── Tier 1: bordered table detection ─────────────────────────────────
        for idx, (page_text, page_tables) in enumerate(zip(full_text, all_page_tables)):
            if _AUDIT_PAGE_RE.search(page_text[:800]):
                continue
            if _SUMMARY_PAGE_RE.search(page_text[:600]):
                continue
            stmt_type = _detect_statement_type(page_text)
            if stmt_type in candidates:
                continue
            pu = _page_unit(idx)
            for table in page_tables:
                if _is_pl_table(table):
                    result_data = _extract_from_table(table, pu, text_period, broadcast_dt)
                    r = _build_result(
                        result_data["fields"], result_data["unit_mult"],
                        result_data["period"],
                        symbol, company, broadcast_dt, source_url,
                        method="table",
                    )
                    if r:
                        candidates[stmt_type] = r
                        break

        # ── Tier 2: text-line parser ──────────────────────────────────────────
        all_text = "\n".join(full_text)
        period   = text_period or _extract_period_from_text(all_text)

        for idx, page_text in enumerate(full_text):
            if not _is_pl_page(page_text):
                continue
            # Skip auditor's reports / limited-review reports: they mention "profit" and
            # "income" in passing but contain no actual P&L values to extract.
            if _AUDIT_PAGE_RE.search(page_text[:800]):
                continue
            if _SUMMARY_PAGE_RE.search(page_text[:600]):
                continue
            stmt_type = _detect_statement_type(page_text)
            if stmt_type in candidates:
                continue
            pu = _page_unit(idx)
            extracted = _extract_from_text(page_text)
            r = _build_result(
                extracted, pu, period,
                symbol, company, broadcast_dt, source_url,
                method="text-page",
            )
            if r:
                candidates[stmt_type] = r

        # Prefer standalone > unknown > consolidated
        for preferred in ("standalone", "unknown", "consolidated"):
            if preferred in candidates:
                return candidates[preferred]

        # Last resort: parse full document text
        extracted = _extract_from_text(all_text)
        return _build_result(
            extracted, unit_mult, period,
            symbol, company, broadcast_dt, source_url,
            method="text-full",
        )

    except Exception:
        return None
    finally:
        try:
            pdf.close()
        except Exception:
            pass
