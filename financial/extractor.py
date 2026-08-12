"""
FinancialExtractor — sends a results PDF to Claude and gets structured
financial metrics back via JSON structured output.

Model: claude-opus-4-7 (best accuracy for financial document parsing).
The system prompt is cached via cache_control so repeated extractions
in the same process share the cached prefix.

Structured output uses output_config.format with a strict JSON schema,
guaranteeing parseable JSON even for complex multi-page PDFs.
"""
from __future__ import annotations

import base64
import json

import anthropic

from financial.models import FinancialResult


# ── JSON schema for structured extraction ────────────────────────────────
# anyOf allows null values; additionalProperties:false is required by Claude.

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "period": {
            "type": "string",
            "description": "Reporting period e.g. 'Q4 FY2026', 'FY2026', 'H1 FY2026'"
        },
        "period_type": {
            "type": "string",
            "description": "One of: quarterly, annual, half-yearly"
        },
        "revenue_cr": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Total revenue / net sales in Rs. Crore"
        },
        "revenue_growth_pct": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "YoY revenue growth percentage"
        },
        "ebitda_cr": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "EBITDA in Rs. Crore (use Operating Profit if EBITDA not stated)"
        },
        "ebitda_margin_pct": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "EBITDA margin as percentage of revenue"
        },
        "pat_cr": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Profit After Tax (Net Profit) in Rs. Crore"
        },
        "pat_growth_pct": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "YoY PAT growth percentage"
        },
        "eps": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Earnings Per Share in Rs."
        },
        "dividend_per_share": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Dividend declared per share in Rs. (null if not declared)"
        },
        "key_highlights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3 to 5 concise bullet points capturing the most important facts"
        },
        "guidance": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Management outlook or guidance for next quarter/year. Null if not mentioned."
        },
        "raw_summary": {
            "type": "string",
            "description": "2-3 sentence plain-English summary of the results"
        },
    },
    "required": [
        "period", "period_type",
        "revenue_cr", "revenue_growth_pct",
        "ebitda_cr", "ebitda_margin_pct",
        "pat_cr", "pat_growth_pct",
        "eps", "dividend_per_share",
        "key_highlights", "guidance", "raw_summary",
    ],
    "additionalProperties": False,
}

# ── Cached system prompt ─────────────────────────────────────────────────

_SYSTEM: list[dict] = [
    {
        "type": "text",
        "text": (
            "You are a senior equity research analyst specialising in Indian listed companies. "
            "Your job is to read a company's financial results document (quarterly or annual) "
            "and extract key financial metrics with precision.\n\n"
            "Rules:\n"
            "- All monetary values must be in Rs. Crore. Convert if stated in Rs. Lakhs (divide by 100) "
            "or Rs. Million (divide by 10).\n"
            "- Growth percentages must be Year-on-Year (Q4 FY26 vs Q4 FY25, or FY26 vs FY25).\n"
            "- If a figure is not present in the document, return null — never guess or hallucinate.\n"
            "- For EBITDA: use the stated EBITDA figure. If absent, use Operating Profit or EBIT.\n"
            "- EPS should be the diluted EPS if both basic and diluted are given.\n"
            "- Key highlights should be factual bullet points about results, margins, order books, "
            "or significant one-time items — not generic statements.\n"
            "- Return the period exactly as stated in the document (e.g. 'Q4 FY2026', 'FY2026')."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]

_USER_PROMPT = (
    "Extract all financial metrics from this results document for {company} ({symbol}). "
    "The document was filed on {broadcast_dt}. "
    "Return structured JSON following the schema exactly."
)


class FinancialExtractor:
    """
    Uses claude-opus-4-7 with PDF base64 input and JSON structured output
    to extract financial metrics from a results announcement PDF.
    """

    # Use claude-opus-4-7 for financial extraction — accuracy matters here
    _EXTRACT_MODEL = "claude-opus-4-7"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── public ────────────────────────────────────────────────────────────

    def extract(
        self,
        pdf_bytes: bytes,
        symbol: str,
        company: str,
        broadcast_dt: str,
        source_url: str,
    ) -> FinancialResult | None:
        """
        Send pdf_bytes to Claude and return a FinancialResult.
        Returns None if extraction or parsing fails.
        """
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        try:
            response = self._client.messages.create(
                model=self._EXTRACT_MODEL,
                max_tokens=2048,
                system=_SYSTEM,
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _SCHEMA,
                    }
                },
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": _USER_PROMPT.format(
                                    company=company,
                                    symbol=symbol,
                                    broadcast_dt=broadcast_dt,
                                ),
                            },
                        ],
                    }
                ],
            )
        except anthropic.BadRequestError:
            # PDF may be malformed or an image-only scan Claude can't parse
            return None
        except Exception:
            return None

        # ── parse structured output ───────────────────────────────────────
        try:
            text = next(
                (b.text for b in response.content if b.type == "text"), None
            )
            if not text:
                return None
            data: dict = json.loads(text)
        except (json.JSONDecodeError, StopIteration):
            return None

        return FinancialResult(
            symbol             = symbol,
            company            = company,
            period             = data.get("period") or "Unknown",
            period_type        = data.get("period_type") or "quarterly",
            source_url         = source_url,
            broadcast_dt       = broadcast_dt,
            revenue_cr         = data.get("revenue_cr"),
            revenue_growth_pct = data.get("revenue_growth_pct"),
            ebitda_cr          = data.get("ebitda_cr"),
            ebitda_margin_pct  = data.get("ebitda_margin_pct"),
            pat_cr             = data.get("pat_cr"),
            pat_growth_pct     = data.get("pat_growth_pct"),
            eps                = data.get("eps"),
            dividend_per_share = data.get("dividend_per_share"),
            key_highlights     = data.get("key_highlights") or [],
            guidance           = data.get("guidance"),
            raw_summary        = data.get("raw_summary") or "",
        )
