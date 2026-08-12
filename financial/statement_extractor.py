"""
statement_extractor.py
Extracts the FULL P&L statement table from quarterly result PDFs.

Captures all 4 columns:
  col1 = current quarter    (e.g. 30.06.2026 = Q1 FY2027)
  col2 = previous quarter   (e.g. 31.03.2026 = Q4 FY2026)
  col3 = same qtr last year (e.g. 30.06.2025 = Q1 FY2026)  ← YoY base
  col4 = full year          (e.g. 31.03.2026 = FY2026)

Stores every labelled row so no data is lost.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── helpers ────────────────────────────────────────────────────────────────────

_NUM_RE   = re.compile(r'^-?\(?\d[\d,\.\s]*\)?$')   # allow spaces e.g. "8 ,149.00"
_DATE_HDR = re.compile(r'\b(3[01])[.\-/](0[1-9]|1[0-2])[.\-/](20\d{2})\b')
_UNIT_RE  = re.compile(
    r'(?:'
    r'(?:rs?\.?|rupees?|inr|amount|₹)\s*in\s*'   # "Rs in", "Amount in", "₹ in", "INR in"
    r'|in\s+(?:rs?\.?\s*)?'                            # "in Lakhs", "in Rs. Lakhs"
    r')'
    r'(crore|lakh|lacs?|million|thousand)',          # lacs/lac = alternate Indian spelling
    re.I,
)


def _clean_num(s: str) -> float | None:
    if not s:
        return None
    s = s.strip()
    negative = s.startswith('(') and s.endswith(')')
    s = s.strip('()')
    # Some Indian PDFs use comma as decimal separator: "59,339,86" = 59339.86
    # Detect: no period AND last comma-group is exactly 2 digits → treat as decimal.
    # "59,339,86" → "59339.86"; "1,09,483,82" → "109483.82"
    # Normal integers end in 3-digit groups ("1,09,483") so they won't match.
    if '.' not in s:
        _m = re.match(r'^([\d,]+),(\d{2})$', s)
        if _m:
            s = _m.group(1).replace(',', '') + '.' + _m.group(2)
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '')
    # Spaces between digit groups are thousands separators in many
    # Indian PDFs, BUT a trailing 2-digit group is the decimal part:
    # "2 197 80" → 2197.80,  "412 72" → 412.72,  "1 234" → 1234 (3-digit = thousands)
    parts = s.split()
    if len(parts) >= 2 and re.fullmatch(r'\d{2}', parts[-1]):
        integer_part = ''.join(parts[:-1])
        s = f"{integer_part}.{parts[-1]}"
    else:
        s = re.sub(r'\s', '', s)
    try:
        v = float(s)
        return -v if negative else v
    except ValueError:
        return None


def _detect_unit(text: str) -> str:
    m = _UNIT_RE.search(text)
    if m:
        u = m.group(1).lower()
        if 'crore' in u: return 'Cr'
        if 'lakh' in u or u.startswith('lac'): return 'Lakhs'
        if 'million' in u: return 'Mn'
    return 'Cr'


def _to_cr(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    if unit == 'Lakhs':
        return round(value / 100, 4)
    if unit == 'Mn':
        return round(value / 10, 4)
    return value


def _clean_label(s: str) -> str:
    s = re.sub(r'^\d+[a-z]?[\.\)]\s*', '', s.strip())   # strip row numbers
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# ── core extraction ─────────────────────────────────────────────────────────────

@dataclass
class FinancialStatement:
    symbol:       str
    company:      str
    source_url:   str
    broadcast_dt: str
    unit:         str
    col1_label:   str
    col2_label:   str
    col3_label:   str | None
    col4_label:   str | None
    line_items:   dict[str, list[float | None]]   # label → [c1, c2, c3, c4]


def _extract_header_dates(rows: list[list]) -> tuple[str, str, str | None, str | None]:
    """Find date labels from table header rows (supports DD.MM.YYYY and '30 June 2026' formats)."""
    dates = []
    for row in rows[:5]:
        for cell in (row or []):
            if not cell:
                continue
            s = str(cell)
            # Numeric date: 30.06.2026 / 31-03-2026
            m = _DATE_HDR.search(s)
            if m:
                dates.append(m.group(0)[:10])
                continue
            # Word date: "30 June 2026", "31 March 2026 (Audited)", etc.
            m2 = _DATE_WORDS.search(s)
            if m2:
                mm = _MONTHS.get(m2.group(2).lower(), '??')
                dates.append(f"{int(m2.group(1)):02d}.{mm}.{m2.group(3)}")
    # dedupe preserving order
    seen, out = set(), []
    for d in dates:
        if d not in seen:
            seen.add(d); out.append(d)
    while len(out) < 4:
        out.append(None)
    return tuple(out[:4])


def _parse_table(table: list[list], unit: str) -> tuple[dict, tuple]:
    """
    Parse a P&L table into {label: [c1,c2,c3,c4]} dict.
    Returns (line_items, (col1,col2,col3,col4) labels).
    """
    headers = _extract_header_dates(table)
    line_items: dict[str, list] = {}

    for row in table:
        if not row:
            continue
        cells = [str(c).strip() if c else '' for c in row]

        # Find the label column (leftmost non-empty non-numeric cell)
        label = ''
        num_start = 0
        for i, c in enumerate(cells):
            if c and not _NUM_RE.match(c):
                label = _clean_label(c)
                num_start = i + 1
                break

        if not label or len(label) < 3:
            continue

        # Collect numeric values after the label (store in original PDF unit)
        nums = []
        for c in cells[num_start:]:
            v = _clean_num(c)
            nums.append(v)

        # Pad/trim to 4 columns
        while len(nums) < 4:
            nums.append(None)
        nums = nums[:4]

        # Only keep rows that have at least one real number
        if any(v is not None for v in nums):
            line_items[label] = nums

    return line_items, headers


def _is_results_table(table: list[list]) -> bool:
    """Check if this table looks like a P&L results table."""
    if len(table) < 5:
        return False
    # Check date header in first 5 rows only — notes/segment tables have dates deeper down
    header_text = ' '.join(
        str(c).lower() for row in table[:5] for c in (row or []) if c
    )
    body_text = ' '.join(
        str(c).lower() for row in table[:8] for c in (row or []) if c
    )
    has_period = bool(_DATE_HDR.search(header_text)) or any(
        w in header_text for w in ['quarter', 'ended', 'preceding', 'corresponding']
    )
    has_income = any(w in body_text for w in [
        'revenue', 'income', 'interest earned', 'total income', 'profit'
    ])
    return has_period and has_income


def extract_full_statement(
    pdf_bytes: bytes,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialStatement | None:
    """
    Extract the complete P&L table (all rows, all 4 columns) from a quarterly result PDF.
    Returns None if no results table is found.
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None

    unit = 'Cr'
    # Track best candidate per statement type: standalone > unknown > consolidated
    # Each entry: (quality_score, row_count, FinancialStatement)
    best_by_type: dict[str, tuple[int, int, FinancialStatement]] = {}

    try:
        for page in pdf:
            page_text = page.get_text("text") or ""
            u = _detect_unit(page_text)
            if u != 'Cr':
                unit = u

            # Skip pages with no financial keywords
            tl = page_text.lower()
            if not any(w in tl for w in ['revenue', 'income', 'profit', 'interest earned']):
                continue

            # Skip investor KPI summary pages (reversed column order, no full P&L)
            if re.search(r'\bKey\s+Parameters?\b|\bGroup\s+Performance\b|\bBusiness\s+Performance\b|\bPerformance\s+Highlights?\b', page_text[:600], re.I):
                continue

            stmt_type = _detect_statement_type(page_text)

            def _update_best(items: dict, hdrs: tuple) -> None:
                q = _quality_score(items)
                if len(items) < 5:
                    return
                # Require at least 1 P&L anchor match — prevents segment/summary
                # tables (quality=0) from being stored when no real P&L is found.
                if q == 0:
                    return
                prev = best_by_type.get(stmt_type)
                if prev is None or q > prev[0] or (q == prev[0] and len(items) > prev[1]):
                    best_by_type[stmt_type] = (q, len(items), FinancialStatement(
                        symbol=symbol, company=company,
                        source_url=source_url, broadcast_dt=broadcast_dt,
                        unit=unit,
                        col1_label=hdrs[0] or '', col2_label=hdrs[1] or '',
                        col3_label=hdrs[2], col4_label=hdrs[3],
                        line_items=items,
                    ))

            # ── Tier 1: bordered table ────────────────────────────────────────
            try:
                finder = page.find_tables()
                tables = [t.extract() for t in finder.tables]
            except Exception:
                tables = []

            for table in tables:
                if not _is_results_table(table):
                    continue
                items, hdrs = _parse_table(table, unit)
                _update_best(items, hdrs)

            # ── Tier 2: line-by-line text parser (most Indian PDFs) ───────────
            items, hdrs = _parse_text_table(page_text, unit)
            _update_best(items, hdrs)

    finally:
        pdf.close()

    # Prefer standalone > unknown > consolidated
    for preferred_type in ("standalone", "unknown", "consolidated"):
        entry = best_by_type.get(preferred_type)
        if entry:
            return entry[2]
    return None


_NUM_LINE_RE = re.compile(r'^-?\(?\d[\d,\.]*\)?$')
# Matches "30.06.2026" or "30-06-2026"
_DATE_NUM = re.compile(r'\b(3[01]|[012]\d)[.\-/](0[1-9]|1[0-2])[.\-/](20\d{2})\b')
# Matches "30 June 2026", "30th June 2026", "30 ,June 2026" (OCR comma artifact)
_DATE_WORDS = re.compile(
    r'\b(\d{1,2})(?:st|nd|rd|th)?[,\.\s]+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'[,\.\s]+(20\d{2})\b', re.I
)
_MONTHS = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06',
           'july':'07','august':'08','september':'09','october':'10','november':'11','december':'12'}

_NOISE_RE = re.compile(
    r'^(unaudited|audited|refer\s+note|quarter\s+ended|year\s+ended|'
    r'months?\s+ended|preceding|corresponding|for\s+the|standalone|'
    r'consolidated|sr\.?\s*no|sl\.?\s*no|particulars?\s*$)', re.I
)

_STANDALONE_RE   = re.compile(r'\bstandalone\b',   re.IGNORECASE)
_CONSOLIDATED_RE = re.compile(r'\bconsolidated\b', re.IGNORECASE)


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
_SECTION_HDR = re.compile(
    r'^[IVXivx]+\.\s+|^[A-Z]\.\s+|^[a-z]\)\s+|^\d+[a-z]?\s*[\.\)]\s+', re.I
)
# Dot-leader / punctuation-only lines that appear between labels and values
# in some Indian PDFs (e.g. Infosys uses ". . . . ." leader lines).
_DECORATION_RE = re.compile(r'^[\s.:\-–—•·~*|()\'"` ]+$')

# Known P&L anchor labels used to score extraction quality
_PL_ANCHORS = frozenset({
    "revenue from oper", "total income", "profit before tax",
    "profit for the period", "profit after tax", "employee benefit",
    "total expenses", "earnings per equity",
})


def _is_decoration(line: str) -> bool:
    """Return True for dot-leader / punctuation-only lines like '.', '••••', '. I'."""
    alnum = re.sub(r'[^a-zA-Z0-9]', '', line)
    return len(alnum) <= 1 or bool(_DECORATION_RE.match(line))


def _quality_score(items: dict) -> int:
    """Count how many known P&L anchor labels appear as keys (higher = better extraction)."""
    return sum(1 for k in items if any(a in k.lower() for a in _PL_ANCHORS))


def _is_num_line(s: str) -> bool:
    return bool(_NUM_LINE_RE.match(s.strip()))


_TITLE_DATE_RE = re.compile(
    r'financial\s+results?|results?\s+for|quarter\s+ended|year\s+ended',
    re.IGNORECASE,
)
# OCR month-name garbling — e.g. "June" → "Jnnc" (u→n, e→c)
_JUNE_OCR_RE  = re.compile(r'\bJ[a-z]n[a-z]\b', re.I)   # Jnnc, Junc, Jnne → June
_JULY_OCR_RE  = re.compile(r'\bJ[a-z][ln][yv]\b', re.I)  # Jnly, Jnlv → July


def _extract_dates(lines: list[str]) -> tuple[list[str | None], int]:
    """
    Find period date labels from page text.
    Counts occurrences (not unique) — Q1 results have Q4-end = FY-end,
    so the same date appears twice (col2 and col4).
    """
    def _normalize(line: str) -> str:
        """Fix common OCR garbling of short month names."""
        line = _JUNE_OCR_RE.sub('June', line)
        line = _JULY_OCR_RE.sub('July', line)
        return line

    def _line_dates(line: str) -> list[str]:
        line = _normalize(line)
        found = []
        for m in _DATE_NUM.finditer(line):
            found.append(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
        for m in _DATE_WORDS.finditer(line):
            mm = _MONTHS.get(m.group(2).lower(), '??')
            found.append(f"{int(m.group(1)):02d}.{mm}.{m.group(3)}")
        return found

    # First pass: find a line that contains 2+ dates (the column header row).
    # Single-date title lines ("Statement of...ended 30 June 2026") are skipped.
    for line in lines:
        ld = _line_dates(line)
        if len(ld) >= 2:
            all_found = ld
            break
    else:
        # Fallback: collect from all lines, skipping title/description lines
        # that contain exactly one date embedded in financial report text.
        all_found = []
        for line in lines:
            ld = _line_dates(line)
            if ld and _TITLE_DATE_RE.search(line):
                continue  # skip date embedded in "Statement of...year ended..."
            all_found.extend(ld)
            if len(all_found) >= 4:
                break

    # Take the first 4 occurrences as column headers
    while len(all_found) < 4:
        all_found.append(None)
    dates = all_found[:4]
    n_cols = sum(1 for d in dates if d is not None)
    n_cols = max(n_cols, 3)
    return dates, n_cols


def _parse_text_table(text: str, unit: str) -> tuple[dict, tuple]:
    """
    Parse Indian quarterly result PDFs where each cell occupies one line:
      Label text
      1,59,479.28        <- col1  (current quarter)
      1,34,896.40        <- col2  (previous quarter)
      1,29,577.89        <- col3  (same quarter last year = YoY base)
      5,22,668.25        <- col4  (full year)
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    line_items: dict[str, list] = {}

    headers, n_cols = _extract_dates(lines)

    # ── Find where the actual data table starts ────────────────────────────────
    # Look for the first line that has at least 2 date patterns, then skip
    # until we reach a known P&L label (revenue/income/profit/interest earned)
    table_start = 0
    for i, line in enumerate(lines):
        if any(w in line.lower() for w in [
            'revenue from oper', 'total income', 'interest earned',
            'net interest income', 'income from operation',
        ]):
            table_start = i
            break

    data_lines = lines[table_start:]

    # ── Walk line by line ──────────────────────────────────────────────────────
    label_parts: list[str] = []
    num_buf:     list[float | None] = []

    def _flush() -> None:
        if not label_parts or not any(v is not None for v in num_buf):
            return
        # Join multi-line labels, strip serial number prefix
        raw = ' '.join(label_parts)
        label = _clean_label(raw)
        if label and len(label) >= 3:
            row = list(num_buf[:n_cols])
            while len(row) < 4:
                row.append(None)
            line_items[label] = row

    for line in data_lines:
        if _is_num_line(line):
            v = _clean_num(line)
            num_buf.append(v)  # store in original PDF unit
            if len(num_buf) >= n_cols:
                _flush()
                label_parts = []
                num_buf = []
        else:
            # Non-numeric line
            if _NOISE_RE.match(line):
                continue
            if _is_decoration(line):        # skip dot-leaders like '.', '••••', '. I'
                continue
            if _DATE_NUM.search(line) or _DATE_WORDS.search(line):
                continue  # header date line
            if num_buf:
                # Partial numbers then hit text → discard partial (section heading)
                num_buf = []
            # Start or continue label accumulation (cap at 3 lines to avoid blob labels)
            if len(label_parts) < 3:
                label_parts.append(line)
            else:
                # Too many label lines → treat as new label, flush nothing
                label_parts = [line]

    _flush()
    return line_items, tuple(headers)


# ── Storage ────────────────────────────────────────────────────────────────────

class StatementStorage:
    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_statements (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL,
                company      TEXT,
                source_url   TEXT NOT NULL,
                broadcast_dt TEXT,
                unit         TEXT,
                col1_label   TEXT,
                col2_label   TEXT,
                col3_label   TEXT,
                col4_label   TEXT,
                line_items   TEXT,
                extracted_at TEXT DEFAULT (datetime('now')),
                UNIQUE(symbol, source_url)
            )
        """)
        self._conn.commit()

    def save(self, stmt: FinancialStatement) -> bool:
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO financial_statements
                (symbol, company, source_url, broadcast_dt, unit,
                 col1_label, col2_label, col3_label, col4_label, line_items)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                stmt.symbol, stmt.company, stmt.source_url, stmt.broadcast_dt,
                stmt.unit, stmt.col1_label, stmt.col2_label,
                stmt.col3_label, stmt.col4_label,
                json.dumps(stmt.line_items),
            ))
            self._conn.commit()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._conn.close()
