"""
GeneralExtractor — extracts key facts from non-financial PDFs using Claude.

Handles: order wins, acquisitions, restructuring, fundraising, buybacks,
open offers, credit ratings, and CIRP documents.

Returns a FinancialResult with P&L fields null and key_highlights + raw_summary
populated — reuses the same model and storage so no schema changes are needed.
"""
from __future__ import annotations

import base64
import json

import anthropic

from financial.models import FinancialResult


# ── Subject-specific extraction prompts ──────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "order_win": (
        "Extract the following from this order/contract announcement PDF:\n"
        "- Order value in Rs. Crore (exact figure)\n"
        "- Client or buyer name and type (government / private / PSU)\n"
        "- What goods or services are being supplied\n"
        "- Contract duration or delivery timeline\n"
        "- Sector / end-market (defence, railways, power, roads, etc.)\n"
        "- Any payment terms or advance mentioned\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "acquisition": (
        "Extract the following from this acquisition / investment announcement PDF:\n"
        "- Target company name and its core business\n"
        "- Deal value in Rs. Crore (convert USD/EUR if needed using ~84 rate)\n"
        "- Stake percentage being acquired\n"
        "- Valuation multiple or EV/EBITDA if disclosed\n"
        "- Strategic rationale stated by management\n"
        "- Regulatory approvals needed and expected completion\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "restructuring": (
        "Extract the following from this merger / demerger / scheme PDF:\n"
        "- Companies involved and their roles\n"
        "- Swap ratio or cash consideration (if merger)\n"
        "- Nature of restructuring (merger, demerger, hive-off, spin-off, etc.)\n"
        "- Strategic rationale — what entity/business gets created\n"
        "- NCLT filing status and expected effective date\n"
        "- Impact on shareholders of each entity\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "fundraising": (
        "Extract the following from this fundraising document PDF:\n"
        "- Instrument type (QIP / rights / preferential allotment / NCD, etc.)\n"
        "- Amount raised or proposed in Rs. Crore\n"
        "- Issue / floor price per share (for equity instruments)\n"
        "- Named allottees or lead investors if disclosed\n"
        "- Dilution % or number of shares to be issued\n"
        "- Stated use of proceeds\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "buyback": (
        "Extract the following from this buyback offer document PDF:\n"
        "- Total buyback size in Rs. Crore\n"
        "- Maximum buyback price per share\n"
        "- Method: open market purchase or tender offer\n"
        "- Number of shares proposed to be bought back\n"
        "- Buyback as % of total paid-up capital\n"
        "- Buyback period / record date\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "open_offer": (
        "Extract the following from this open offer / public announcement PDF:\n"
        "- Acquirer name and background\n"
        "- Offer price per share (and premium to last close if disclosed)\n"
        "- % stake being acquired in the open offer\n"
        "- Total offer size in Rs. Crore\n"
        "- Trigger for open offer (promoter stake purchase, agreement, etc.)\n"
        "- Regulatory / SEBI timeline\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "credit_rating": (
        "Extract the following from this credit rating report PDF:\n"
        "- New rating and outlook (e.g. CRISIL AA+/Stable)\n"
        "- Previous rating and outlook (if this is a revision)\n"
        "- Instrument rated and total debt amount\n"
        "- Key strengths cited by the rating agency\n"
        "- Key risk factors or concerns noted\n"
        "- Watch/review status if any\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
    "cirp": (
        "Extract the following from this CIRP / insolvency resolution PDF:\n"
        "- Total admitted debt amount\n"
        "- Resolution applicant (acquirer) name and their offered amount\n"
        "- Implied haircut percentage for lenders\n"
        "- NCLT bench and key hearing dates\n"
        "- Status: admitted / approved / challenged / implemented\n"
        "- Impact on equity shareholders (likely zero / partial / unknown)\n"
        "Give 4-6 concise bullet highlights and a 2-sentence plain-English summary."
    ),
}

# ── JSON schema ───────────────────────────────────────────────────────────────

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "key_highlights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-6 concise bullet points with the most important facts",
        },
        "raw_summary": {
            "type": "string",
            "description": "2-sentence plain-English summary of the document",
        },
        "period": {
            "type": "string",
            "description": "Date or period this document covers, e.g. '2026-06-20' or 'Q4 FY2026'",
        },
    },
    "required": ["key_highlights", "raw_summary", "period"],
    "additionalProperties": False,
}

# ── Cached system prompt ──────────────────────────────────────────────────────

_SYSTEM: list[dict] = [
    {
        "type": "text",
        "text": (
            "You are a senior equity analyst covering Indian listed companies. "
            "Read the attached corporate announcement PDF and extract the requested facts precisely. "
            "Rules:\n"
            "- All monetary amounts must be stated in Rs. Crore. "
            "Convert USD/EUR using ~84 Rs/USD rate; convert Rs. Lakh by dividing by 100.\n"
            "- Do not guess or hallucinate facts not present in the document — "
            "omit them or state 'not disclosed'.\n"
            "- Bullet points must be factual and specific, not generic marketing language.\n"
            "- Keep the summary to exactly 2 sentences."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


class GeneralExtractor:
    """
    Extracts key facts from non-financial announcement PDFs.
    Uses claude-haiku-4-5 (fast, cheap) since no arithmetic precision is needed.
    Returns a FinancialResult with P&L fields null and highlights/summary filled.
    """

    _MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def extract(
        self,
        pdf_bytes:    bytes,
        symbol:       str,
        company:      str,
        broadcast_dt: str,
        source_url:   str,
        extraction_type: str,
    ) -> FinancialResult | None:
        """
        Send pdf_bytes to Claude with a subject-specific prompt.
        Returns FinancialResult with P&L fields null; key_highlights and raw_summary filled.
        Returns None if extraction fails.
        """
        prompt_body = _PROMPTS.get(extraction_type, _PROMPTS["order_win"])
        user_text = (
            f"Company: {company} ({symbol})\n"
            f"Filing date: {broadcast_dt}\n"
            f"Document type: {extraction_type.replace('_', ' ').title()}\n\n"
            f"{prompt_body}"
        )

        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        try:
            response = self._client.messages.create(
                model=self._MODEL,
                max_tokens=1024,
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
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
        except anthropic.BadRequestError:
            return None
        except Exception:
            return None

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
            symbol          = symbol,
            company         = company,
            period          = data.get("period") or broadcast_dt[:10],
            period_type     = extraction_type,       # e.g. "order_win", "acquisition"
            source_url      = source_url,
            broadcast_dt    = broadcast_dt,
            key_highlights  = data.get("key_highlights") or [],
            raw_summary     = data.get("raw_summary") or "",
            # P&L fields intentionally null for non-financial documents
        )
