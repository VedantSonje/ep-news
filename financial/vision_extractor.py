"""
VisionExtractor — multimodal fallback for image-based / scanned PDFs.

Uses Google Gemini Flash 2.0 (vision) when pdfplumber returns < 80 chars.
Converts PDF pages to PNG via pymupdf (no external poppler binary needed),
then sends them to Gemini with a structured JSON prompt.

Set GEMINI_API_KEY in .env to enable. If key is absent, returns None silently.
"""
from __future__ import annotations

import io
import json
import os
import re

from pydantic import ValidationError

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
_MAX_PAGES      = int(os.getenv("VISION_MAX_PAGES", "5"))   # pages to send per PDF
_IMG_ZOOM       = float(os.getenv("VISION_IMG_ZOOM", "1.5")) # lower = smaller images


# ── PDF → images ─────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_bytes: bytes) -> list:
    """Convert PDF pages to PIL Images using pymupdf (bundled, no poppler)."""
    try:
        import fitz                 # pymupdf
        import PIL.Image
    except ImportError:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page in enumerate(doc):
        if i >= _MAX_PAGES:
            break
        mat = fitz.Matrix(_IMG_ZOOM, _IMG_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(PIL.Image.open(io.BytesIO(pix.tobytes("png"))))
    return images


# ── Gemini prompts per extraction type ───────────────────────────────────────

_FINANCIAL_PROMPT = """You are a senior equity analyst. Extract financial results from this Indian BSE/NSE quarterly results PDF.

UNIT CONVERSION (read table header first — apply to every cell except EPS):
- "Rs. in Lakhs"   → divide ALL values by 100 to get Crores
- "Rs. in Millions" → divide ALL values by 10 to get Crores
- "Rs. in Crores"  → use values as-is
- Absolute rupees (₹X,XX,XX,XXX — 8+ digits) → divide by 1,00,00,000 to get Crores
EPS: NEVER apply the unit divisor — use the exact EPS value as printed.
Negative: (48.20) with parentheses = -48.20

COLUMN SELECTION: Indian results have 4 columns. Extract ONLY the FIRST (leftmost) = Current Quarter.
For prior_revenue_cr use the THIRD column = Same Quarter Last Year.

PERIOD: Output as "Qn FYyyyy" — never "Quarter ended …".
Indian FY starts April 1:  Apr-Jun = Q1,  Jul-Sep = Q2,  Oct-Dec = Q3,  Jan-Mar = Q4.
"Quarter ended June 2026" → "Q1 FY2027". If Consolidated: "Q1 FY2027 (Consolidated)".

ROW LABELS (Current Quarter column):
- revenue_cr      : "Revenue from Operations" / "Net Sales" / "Total Income"
- pat_cr          : "Profit After Tax" / "PAT" / "Net Profit"
- pbt_cr          : "Profit Before Tax" / "PBT" / "Profit Before Exceptional Items and Tax"
- depreciation_cr : "Depreciation" / "Depreciation and Amortisation" (standalone line)
- finance_costs_cr: "Finance Costs" / "Interest Expense" / "Borrowing Costs"
- ebitda_cr       : "EBITDA" row — extract directly if labelled; otherwise leave null
- eps             : "Basic EPS" / "Earnings Per Share (Basic)"
- prior_revenue_cr: revenue from Same Quarter Last Year (3rd column) — same unit conversion

Return ONLY valid JSON — no markdown, no extra text:
{
  "period": "<Qn FYyyyy format>",
  "period_type": "quarterly",
  "revenue_cr": <number in Crores or null>,
  "revenue_growth_pct": <float or null>,
  "prior_revenue_cr": <same quarter last year revenue or null>,
  "ebitda_cr": <number in Crores or null>,
  "ebitda_margin_pct": <float or null>,
  "pat_cr": <number in Crores or null>,
  "pat_growth_pct": <float or null>,
  "pbt_cr": <number in Crores or null>,
  "depreciation_cr": <number in Crores or null>,
  "finance_costs_cr": <number in Crores or null>,
  "eps": <float or null>,
  "dividend_per_share": <float or null>,
  "key_highlights": ["<fact 1>", "<fact 2>"],
  "guidance": "<management guidance or null>",
  "raw_summary": "<1-2 sentences using only numbers found in the document>",
  "confidence": "high"
}

confidence must be: "high" (3+ numeric fields), "medium" (1-2), "low" (0 or values uncertain).
"""

_ORDER_PROMPT = """You are an equity analyst. Extract order win details from this corporate announcement.

Return ONLY valid JSON:
{
  "period": "YYYY-MM-DD",
  "order_value_cr": <float — NEW order value ONLY, NOT cumulative order book/backlog>,
  "client_name": "client name or not disclosed",
  "client_type": "Government / PSU" | "Private sector",
  "order_sector": "defence / railways / power / roads / telecom / general",
  "execution_months": <int or null>,
  "description": "what goods/services are being supplied",
  "key_highlights": ["highlight 1", "highlight 2"],
  "raw_summary": "2-sentence summary"
}
"""

_GENERAL_PROMPT = """You are an equity analyst. Extract key facts from this corporate announcement.

Return ONLY valid JSON:
{
  "period": "YYYY-MM-DD",
  "period_type": "order_win" | "press_release",
  "order_value_cr": <float — only if a specific NEW order/contract is mentioned, else null>,
  "client_name": "",
  "client_type": "",
  "order_sector": "",
  "execution_months": null,
  "key_highlights": ["fact 1", "fact 2", "fact 3", "fact 4"],
  "raw_summary": "2-sentence plain English summary"
}

Set period_type to "order_win" only if a clear new contract/order value is stated.
"""

_CONCALL_PROMPT = """Extract management guidance from this conference call transcript or presentation.

Return ONLY valid JSON:
{
  "period": "Q1 FY2027 or similar",
  "revenue_guidance": "...",
  "margin_guidance": "...",
  "key_highlights": ["key management comment 1", "key management comment 2", "..."],
  "raw_summary": "2-sentence summary of management outlook"
}
"""

_PROMPTS: dict[str, str] = {
    "financial_results": _FINANCIAL_PROMPT,
    "order_win":         _ORDER_PROMPT,
    "press_release":     _GENERAL_PROMPT,
    "concall":           _CONCALL_PROMPT,
}


# ── Gemini call ───────────────────────────────────────────────────────────────

def _call_gemini(images: list, prompt: str) -> dict | None:
    """Send images + prompt to Gemini Flash, return parsed JSON dict."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("    [vision] google-genai not installed", flush=True)
        return None

    client = genai.Client(api_key=_GEMINI_API_KEY)

    # Build content parts: prompt text + PIL images
    parts: list = [prompt] + images

    import time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                ),
            )
            raw = response.text.strip()
            break
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                wait = 5 * (attempt + 1)
                print(f"    [vision] Gemini 503 — retrying in {wait}s (attempt {attempt+1}/3)", flush=True)
                time.sleep(wait)
                if attempt == 2:
                    print(f"    [vision] Gemini unavailable after 3 attempts", flush=True)
                    return None
            else:
                print(f"    [vision] Gemini API error: {e}", flush=True)
                return None
    else:
        return None

    # Strip markdown code fences (Gemini wraps JSON in ```json ... ``` blocks)
    raw = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()

    # Extract outermost JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        print(f"    [vision] no JSON found in Gemini response", flush=True)
        return None

    try:
        return json.loads(m.group())
    except json.JSONDecodeError as e:
        print(f"    [vision] JSON parse error: {e}", flush=True)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_with_vision(
    pdf_bytes: bytes,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
    extraction_type: str,
) -> dict | None:
    """
    Extract structured data from an image-based PDF via Gemini Flash vision.

    Returns a raw dict (same shape as Ollama extraction output) so the caller
    can pass it straight to LocalExtractor._build_result() and Pydantic validation.
    Returns None if GEMINI_API_KEY is unset or extraction fails.
    """
    if not _GEMINI_API_KEY:
        return None

    images = _pdf_to_images(pdf_bytes)
    if not images:
        print(f"    [vision] could not render PDF pages", flush=True)
        return None

    prompt = _PROMPTS.get(extraction_type, _GENERAL_PROMPT)
    print(f"    [vision] sending {len(images)} page(s) to Gemini ({_GEMINI_MODEL})", flush=True)

    data = _call_gemini(images, prompt)
    if data is None:
        return None

    # Pydantic validation — import here to avoid circular at module load
    from financial.local_extractor import (
        FinancialExtraction, OrderWinExtraction, GeneralExtraction, ConcallExtraction,
    )
    _MODEL_CLS = {
        "financial_results": FinancialExtraction,
        "order_win":         OrderWinExtraction,
        "press_release":     GeneralExtraction,
        "concall":           ConcallExtraction,
    }
    model_cls = _MODEL_CLS.get(extraction_type, GeneralExtraction)
    try:
        validated = model_cls.model_validate(data)
        data = validated.model_dump()
        print(f"    [vision] Pydantic validation passed", flush=True)
    except ValidationError as e:
        print(f"    [vision] Pydantic warning ({e.error_count()} errors) — using raw", flush=True)

    return data
