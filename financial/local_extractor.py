"""
LocalExtractor — zero-cost PDF extraction pipeline.

1. pdfplumber  : PDF bytes → plain text                (local, free, no ML)
2. Ollama      : plain text → JSON via format="json"   (local, free, llama3)
3. Pydantic    : JSON dict → typed model + validation  (zero-cost, type-safe)

Handles all extraction types: financial_results, order_win, acquisition,
restructuring, fundraising, buyback, open_offer, credit_rating, cirp.

Falls back to None if PDF is scanned/unreadable (Docling returns no text).

NOTE: Structured output uses Ollama's JSON mode + Pydantic validation rather
than PydanticAI tool-calling, because llama3:latest does not support function
calling. When upgraded to llama3.1 or similar, switch to pydantic-ai agents
in _get_agent() for constrained-decoding structured output.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from financial.models import FinancialResult
from financial.rule_extractor import try_rule_extract, extract_order_win_from_table
from financial.financial_table_extractor import extract_financial_table


_OLLAMA_MODEL   = "llama3:latest"
_MAX_TEXT_CHARS = 10_000   # chars sent to Ollama — covers most filings


# ── Pydantic extraction models (validation layer on top of Ollama JSON) ───────
#
# These models validate and coerce the raw JSON dict returned by Ollama, giving
# us type safety and consistent nulls without needing tool-calling support.

class FinancialExtraction(BaseModel):
    period:             str                                            = Field("", description="Quarter/year label e.g. 'Q1 FY2027'")
    period_type:        Literal["quarterly", "annual", "half_yearly"] = "quarterly"
    revenue_cr:         Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    ebitda_cr:          Optional[float] = None
    ebitda_margin_pct:  Optional[float] = None
    pat_cr:             Optional[float] = None
    pat_growth_pct:     Optional[float] = None
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

_FINANCIAL_PROMPT = """You are a senior equity analyst. Read the financial results document below and extract metrics.

RULES:
- All monetary values must be in Rs. Crore. Convert if in Lakhs (÷100) or Millions (÷10).
- Growth % must be Year-on-Year. Use null if not stated.
- Never guess — return null for missing fields.
- Return ONLY valid JSON, no extra text.

Return this exact JSON structure:
{{
  "period": "Q4 FY2026",
  "period_type": "quarterly",
  "revenue_cr": 1234.5,
  "revenue_growth_pct": 12.5,
  "ebitda_cr": 234.5,
  "ebitda_margin_pct": 19.0,
  "pat_cr": 123.4,
  "pat_growth_pct": 25.0,
  "eps": 5.6,
  "dividend_per_share": null,
  "key_highlights": ["Revenue grew 12% YoY", "EBITDA margin expanded 200bps"],
  "guidance": "Management expects 15% growth next year",
  "raw_summary": "2-sentence plain English summary of results."
}}

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
                print(f"    [extract] skipped — too little text", flush=True)
                return None
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
                print(f"    [extract] skipped — too little text", flush=True)
                return None
            text = text[:_MAX_TEXT_CHARS]
            print(f"    [ollama] calling {_OLLAMA_MODEL} for concall", flush=True)
            data = self._call_ollama_concall(text, symbol, company, broadcast_dt)
            if data is None:
                return None
            return self._build_result(data, symbol, company, broadcast_dt, source_url, "concall")

        text = self._pdf_to_text(pdf_bytes)
        if not text or len(text.strip()) < 80:
            print(f"    [extract] skipped — too little text ({len(text.strip())} chars)", flush=True)
            return None

        text = text[:_MAX_TEXT_CHARS]

        # ── Fast path: regex rules for predictable formats ─────────────────
        result = try_rule_extract(extraction_type, text, symbol, company, broadcast_dt, source_url)
        if result is not None:
            print(f"    [rules] extracted via regex (no LLM needed)", flush=True)
            return result

        # ── Slow path: Ollama for complex / irregular formats ──────────────
        print(f"    [ollama] no rules matched — calling {_OLLAMA_MODEL}", flush=True)
        if extraction_type == "financial_results":
            data = self._call_ollama_financial(text, symbol, company, broadcast_dt)
        else:
            data = self._call_ollama_general(text, symbol, company, broadcast_dt, extraction_type)

        if data is None:
            return None

        return self._build_result(data, symbol, company, broadcast_dt, source_url, extraction_type)

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
            if result is not None:
                print(f"    [table] P&L table parsed directly", flush=True)
                return result, None

        text = self._pdf_to_text(pdf_bytes)
        if not text or len(text.strip()) < 80:
            return None, None
        text = text[:_MAX_TEXT_CHARS]

        result = try_rule_extract(extraction_type, text, symbol, company, broadcast_dt, source_url)
        if result is not None:
            print(f"    [rules] extracted via regex", flush=True)
            return result, None

        return None, text

    def ollama_extract(
        self,
        text:            str,
        symbol:          str,
        company:         str,
        broadcast_dt:    str,
        source_url:      str,
        extraction_type: str,
    ) -> "FinancialResult | None":
        """
        Phase 2 — Ollama only.  Called with pre-extracted text from fast_extract().
        Always runs sequentially (single GPU can only process one request at a time).
        """
        print(f"    [ollama] calling {_OLLAMA_MODEL} for {extraction_type}", flush=True)
        if extraction_type == "financial_results":
            data = self._call_ollama_financial(text, symbol, company, broadcast_dt)
        elif extraction_type == "concall":
            data = self._call_ollama_concall(text, symbol, company, broadcast_dt)
        else:
            data = self._call_ollama_general(text, symbol, company, broadcast_dt, extraction_type)

        if data is None:
            return None

        etype = extraction_type
        if extraction_type == "press_release":
            etype = "order_win" if data.get("order_value_cr") else "press_release"
        return self._build_result(data, symbol, company, broadcast_dt, source_url, etype)

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

    def _call_ollama_financial(self, text: str, symbol: str, company: str, broadcast_dt: str) -> dict | None:
        prompt = _FINANCIAL_PROMPT.format(
            company=company, symbol=symbol, broadcast_dt=broadcast_dt, text=text,
        )
        return self._ollama_json_validated(prompt, FinancialExtraction)

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

        return FinancialResult(
            symbol             = symbol,
            company            = company,
            period             = str(data.get("period") or broadcast_dt[:10]),
            period_type        = extraction_type,
            source_url         = source_url,
            broadcast_dt       = broadcast_dt,
            revenue_cr         = _float("revenue_cr"),
            revenue_growth_pct = _float("revenue_growth_pct"),
            ebitda_cr          = _float("ebitda_cr"),
            ebitda_margin_pct  = _float("ebitda_margin_pct"),
            pat_cr             = _float("pat_cr"),
            pat_growth_pct     = _float("pat_growth_pct"),
            eps                = _float("eps"),
            dividend_per_share = _float("dividend_per_share"),
            key_highlights     = data.get("key_highlights") or [],
            guidance           = data.get("guidance") or None,
            raw_summary        = str(data.get("raw_summary") or ""),
        )
