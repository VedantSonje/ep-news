"""
Domain models for the financial extraction layer.

FinancialResult  — all metrics extracted from one results PDF.
ExtractionStatus — tracks the outcome of a single PDF extraction attempt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class ExtractionStatus(Enum):
    SUCCESS    = "success"
    SKIPPED    = "skipped"       # already extracted previously
    DL_FAILED  = "dl_failed"     # PDF download failed
    PDF_SKIP   = "pdf_skip"      # URL is not a PDF (e.g. .zip)
    LLM_FAILED = "llm_failed"    # Claude returned unusable output
    TOO_LARGE  = "too_large"     # PDF > size cap


@dataclass
class FinancialResult:
    """
    Structured financial metrics extracted from one company's results PDF.
    All monetary amounts are in Indian Rupees Crore (Rs. Cr) unless noted.
    """
    symbol:             str
    company:            str
    period:             str             # "Q4 FY2026", "FY2026", etc.
    period_type:        str             # "quarterly" | "annual" | "half-yearly"
    source_url:         str
    broadcast_dt:       str

    # ── P&L metrics ────────────────────────────────────────────────────────
    revenue_cr:         float | None = None
    revenue_growth_pct: float | None = None   # YoY %
    ebitda_cr:          float | None = None
    ebitda_margin_pct:  float | None = None   # %
    pat_cr:             float | None = None   # Profit After Tax
    pat_growth_pct:     float | None = None   # YoY %
    eps:                float | None = None   # Rs per share
    dividend_per_share: float | None = None   # Rs

    # ── Qualitative ────────────────────────────────────────────────────────
    key_highlights:     list[str] = field(default_factory=list)
    guidance:           str | None = None
    raw_summary:        str = ""

    # ── Structural / classification ────────────────────────────────────────
    sector:             str | None = None   # sector tag (e.g. "defence", "railways")
    period_label:       str | None = None   # normalized "Q1 FY2027", "FY2026", etc.

    # ── Order win specifics ────────────────────────────────────────────────
    client_name:        str | None = None   # who placed the order
    client_type:        str | None = None   # "Government / PSU" | "Private sector"
    order_sector:       str | None = None   # "defence", "power", "roads", ...
    execution_months:   int | None = None   # estimated execution duration

    # ── Helpers ────────────────────────────────────────────────────────────

    def highlights_str(self) -> str:
        return " | ".join(self.key_highlights)

    def to_embed_doc(self) -> str:
        """Text that gets embedded in ChromaDB — rich narrative for semantic search."""
        rev  = f"Revenue Rs.{self.revenue_cr} cr" if self.revenue_cr else ""
        grow = f"({self.revenue_growth_pct:+.1f}% YoY)" if self.revenue_growth_pct is not None else ""
        pat  = f"PAT Rs.{self.pat_cr} cr" if self.pat_cr else ""
        pat_g= f"({self.pat_growth_pct:+.1f}% YoY)" if self.pat_growth_pct is not None else ""
        ebi  = f"EBITDA margin {self.ebitda_margin_pct:.1f}%" if self.ebitda_margin_pct else ""
        eps  = f"EPS Rs.{self.eps}" if self.eps else ""
        hl   = self.highlights_str()
        parts = filter(None, [
            f"{self.company} ({self.symbol}) {self.period}:",
            self.raw_summary,
            " ".join(filter(None, [rev, grow, "|", pat, pat_g, "|", ebi, eps])),
            hl,
            self.guidance or "",
        ])
        return " ".join(parts)

    def highlights_json(self) -> str:
        return json.dumps(self.key_highlights)
