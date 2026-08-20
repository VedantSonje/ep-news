"""
SEBITableExtractor — parses the SEBI Regulation 30 Annexure-A mandatory disclosure table.

All listed companies must file order/contract announcements in this exact table format:
  S.No | Particulars                                        | Description
  1.   | Name of the entity to which order is awarded      | <client name>
  2.   | Domestic / International entity                   | Domestic
  3.   | Significant terms and conditions                  | <description>
  4.   | Time period for execution                         | 18 Weeks / 24 months
  5.   | Broad commercial consideration or size            | Rs.76,20,00,000 / Rs.500 Crore
  6.   | Whether promoter group has interest               | No
  7.   | Related party transaction                         | No

Column headers vary slightly (S.No / Sr.No, Description / Details of contracts/orders)
but the Particulars column content is standardised by SEBI.

Returns a dict of parsed fields, or None if the Annexure-A table is not found.
"""
from __future__ import annotations

import io
import re


# ── Row keyword matchers — maps field name → keywords in Particulars column ──

_ROW_KEYS: dict[str, list[str]] = {
    "client":      ["name of the entity", "awarding the contract", "entity to which order"],
    "domestic":    ["domestic", "international entity"],
    "description": ["significant terms", "terms and conditions", "nature of order", "nature of contract"],
    "duration":    ["time period", "to be executed", "period of execution", "period by which"],
    "value":       ["broad commercial consideration", "consideration or size", "size of the order",
                    "broad consideration", "value of the contract", "value of the order"],
    "related_party": ["related party"],
}

# SEBI form template text that appears in Description column when companies don't fill in the client.
# Some PDFs copy the Particulars row label verbatim into the Description — strip this prefix.
_SEBI_TEMPLATE_CLIENT = re.compile(
    r"^['\"]?order\s*\(s\)\s*/\s*contract\s*\(s\)\s*['\"]?\s*[;,]?\s*",
    re.IGNORECASE,
)

# ── Amount parsing ────────────────────────────────────────────────────────────

# Matches: Rs. 500 Crore / ₹76,20,00,000 / INR 500 Cr / Rs.847.50 Crores
_RS_CRORE = re.compile(
    r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)\s*(crore|cr\.?|lakh|lakhs|lacs?|million|mn)",
    re.IGNORECASE,
)

# Indian numeric format: 76,20,00,000 (crore,lakh,thousand,hundred,ten,unit)
# Last 7 digits = sub-crore portion → just parse as plain int
_INDIAN_NUM = re.compile(r"([\d,]+(?:\.\d+)?)")

# Word form: "Seventy-Six Crores and Twenty Lakhs"
_WORD_CRORE = re.compile(r"(\w[\w\s-]+?)\s+[Cc]rore", re.IGNORECASE)


def _parse_value_cell(cell: str) -> float | None:
    """Extract Rs. Crore amount from the consideration cell."""
    if not cell:
        return None

    # 1. Explicit Rs. X Crore
    m = _RS_CRORE.search(cell)
    if m:
        raw = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        try:
            val = float(raw)
            if "lakh" in unit or unit.startswith("lac"):
                return round(val / 100, 2)
            if "million" in unit or "mn" in unit:
                return round(val / 10, 2)
            return round(val, 2)   # crore
        except ValueError:
            pass

    # 2. Indian format number (e.g. ₹76,20,00,000)
    # Strip currency symbol, then treat as paise-free integer → divide by 10^7 for crore
    stripped = cell.replace("₹", "").replace("Rs.", "").replace("INR", "").strip()
    m2 = _INDIAN_NUM.search(stripped)
    if m2:
        try:
            val = float(m2.group(1).replace(",", ""))
            # Heuristic: if value > 10,00,000 (10 lakh), it's in rupees not crore
            if val >= 10_000_000:       # ≥1 crore in rupees
                return round(val / 10_000_000, 2)
            elif val >= 100_000:        # in lakhs
                return round(val / 100, 2)
            else:
                return round(val, 2)   # already crore
        except ValueError:
            pass

    return None


def _duration_from_cell(cell: str) -> str:
    """Normalise a time period cell to a short string."""
    if not cell:
        return "not disclosed"
    cell = cell.strip()
    # Remove newlines / extra spaces
    cell = " ".join(cell.split())
    # Truncate long cells
    return cell[:80] if len(cell) <= 80 else cell[:77] + "..."


# ── Table finder ─────────────────────────────────────────────────────────────

def _is_annexure_table(table: list[list[str | None]]) -> bool:
    """Return True if this table looks like a SEBI Reg 30 Annexure-A table."""
    if not table or len(table) < 4:
        return False
    full_text = " ".join(
        str(cell).lower()
        for row in table[:4]
        for cell in (row or [])
        if cell
    )
    return "particular" in full_text and (
        "entity" in full_text
        or "order" in full_text
        or "contract" in full_text
    )


def _match_row(row: list[str | None], keywords: list[str]) -> bool:
    """Return True if the Particulars cell (col 1) matches any keyword."""
    if not row or len(row) < 2:
        return False
    cell = str(row[1] or "").lower().strip()
    return any(kw in cell for kw in keywords)


def _value_cell(row: list[str | None]) -> str:
    """Return the Description column value (index 2+), never the Particulars text (index 1)."""
    for cell in reversed(row[2:] if len(row) > 2 else []):
        if cell and str(cell).strip():
            return str(cell).strip()
    return ""


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_annexure_table(
    pdf_bytes: bytes,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> dict | None:
    """
    Parse SEBI Reg 30 Annexure-A table from PDF bytes using pdfplumber.

    Returns a dict with keys:
        client, domestic, description, duration, value_cr, related_party
    Returns None if the Annexure-A table is not found in the PDF.
    """
    try:
        import pdfplumber
    except ImportError:
        return None

    annexure_table: list[list[str | None]] | None = None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in (tables or []):
                    if _is_annexure_table(table):
                        annexure_table = table
                        break
                if annexure_table:
                    break
    except Exception:
        return None

    # ── Extract each field by row label ──────────────────────────────────────
    # Works whether annexure_table was found or not — text fallback covers both paths
    fields: dict[str, str] = {}

    if annexure_table:
        for row in annexure_table:
            for field_name, keywords in _ROW_KEYS.items():
                if field_name not in fields and _match_row(row, keywords):
                    fields[field_name] = _value_cell(row)

    # ── Text-based value fallback (covers non-standard table layouts) ─────────
    # Re-extract full PDF text and search for "broad consideration" keyword when
    # table parsing missed the value row (e.g. CEIGALL letter-header tables).
    _full_text: str | None = None

    def _get_text() -> str:
        nonlocal _full_text
        if _full_text is None:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as _pdf:
                    _full_text = "\n".join(
                        p.extract_text() or "" for p in _pdf.pages
                    )
            except Exception:
                _full_text = ""
        return _full_text

    # If table parse missed the value, try plain-text regex
    if not fields.get("value"):
        _txt = _get_text()
        _value_kws = ["broad consideration", "consideration or size", "value of the order",
                      "value of the contract", "broad commercial consideration"]
        for _kw in _value_kws:
            _pos = _txt.lower().find(_kw)
            if _pos >= 0:
                _ctx = _txt[_pos: _pos + 300]
                fields["value"] = _ctx
                break

    # Similarly fill client/duration from text if table missed them
    if not fields.get("client") and annexure_table is None:
        _txt = _get_text()
        _m = re.search(r'(?:name of the entity|entity awarding)[^\n]*\n([^\n]+)', _txt, re.IGNORECASE)
        if _m:
            fields["client"] = _m.group(1).strip()
        _dm = re.search(r'(\d+)\s*months?', _txt, re.IGNORECASE)
        if _dm and not fields.get("duration"):
            fields["duration"] = _dm.group(0)

    if not fields and annexure_table is None:
        return None

    value_cr = _parse_value_cell(fields.get("value", ""))
    duration = _duration_from_cell(fields.get("duration", ""))
    # Normalize client name — collapse whitespace, strip newlines, cap at 80 chars
    client_raw = fields.get("client", "not disclosed")
    client = " ".join(client_raw.split())[:80] if client_raw else "not disclosed"
    # Strip SEBI template prefix "order(s)/contract(s);" that some companies copy-paste
    # verbatim into the Description column instead of filling in the actual client name.
    client = _SEBI_TEMPLATE_CLIENT.sub("", client).strip()
    # Also unwrap leading parenthesized acronym: "('RSLDC') acting as..." → "RSLDC acting as..."
    client = re.sub(r"^\(['\"]?(\w+)['\"]?\)\s+", r"\1 ", client).strip()
    if not client:
        client = "not disclosed"
    desc   = " ".join(fields.get("description", "").split())[:200]

    return {
        "client":       client,
        "domestic":     fields.get("domestic", ""),
        "description":  desc,
        "duration":     duration,
        "value_cr":     value_cr,
        "related_party":fields.get("related_party", "No"),
        "value_raw":    fields.get("value", ""),
    }
