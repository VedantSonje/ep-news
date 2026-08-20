"""
FilterConfig — subject-based filter configuration.

An announcement passes if its subject is in keep_subjects and NOT in drop_subjects.
No scoring or regex logic here — this is a pure subject allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilterConfig:
    """Immutable subject allowlist / denylist for the screener."""

    # ── Explicit deny list (wins over keep_subjects) ───────────────────────
    drop_subjects: frozenset[str] = field(default_factory=lambda: frozenset({
        "Copy of Newspaper Publication",
        "Trading Window",
        "ESOP/ESOS/ESPS",
        "Monitoring Agency Report",
        "Statement of deviation(s) or variation(s) under Reg. 32",
        "Clarification - Financial Results",
        "Shareholders meeting",
        "Registrar & Share Transfer Agent Update",
        "Forfeiture",
        "Disclosure of material issue",
        "Change in Company Secretary/Compliance Officer",
    }))

    # ── Subjects that trigger PDF download + extraction ────────────────────
    # Maps subject → extraction_type used by pdf_agent to route to the right extractor.
    # Only subjects that reliably attach meaningful PDFs are listed here.
    pdf_subjects: dict = field(default_factory=lambda: {
        # Full P&L extraction (FinancialExtractor)
        "Outcome of Board Meeting":                         "financial_results",

        # Order/contract details
        "Bagging/Receiving of orders/contracts":            "order_win",
        "Awarding of order(s)/contract(s)":                 "order_win",

        # Press releases & general updates — many contain order wins or key business facts;
        # extractor tries order_win regex first, falls back to general Ollama summary.
        "Press Release":                                    "press_release",
        "Press Release (Revised)":                          "press_release",
        "General Updates":                                  "press_release",
        "Updates":                                          "press_release",

        # Concall / investor meet — extract management guidance and forward-looking statements.
        "Analysts/Institutional Investor Meet/Con. Call Updates": "concall",

        # M&A / restructuring
        "Acquisition":                                      "acquisition",
        "Amalgamation/Merger":                              "restructuring",
        "Demerger":                                         "restructuring",
        "Scheme of Arrangement":                            "restructuring",
        "Scheme of Amalgamation - NCLT Order":              "restructuring",

        # Fundraising / capital
        "Qualified Institutional Placement":                "fundraising",
        "Rights Issue":                                     "fundraising",
        "Fund Raising - Preferential Issue":                "fundraising",
        "Buyback":                                          "buyback",

        # Takeover
        "Public Announcement-Open Offer":                   "open_offer",

        # Credit ratings
        "Credit Rating- Revision":                          "credit_rating",
        "Credit Rating- New":                               "credit_rating",

        # Distress / CIRP
        "Corporate Insolvency Resolution Process":          "cirp",
        "Resolution Plan (for CIRP companies)":             "cirp",

        # Summary-only — extract a plain-text summary; no financial table expected
        # Investor presentations, monthly updates, capacity news, MoUs all carry
        # useful business context but never contain P&L tables.
        "Investor Presentation":                            "summary_only",
        "Monthly Business Updates":                         "summary_only",
        "Capacity addition":                                "summary_only",
        "Commencement of commercial production/operations": "summary_only",
        "Agreements":                                       "summary_only",
        "Memorandum of Understanding/Agreements":           "summary_only",
        "Diversification/Disinvestment":                    "summary_only",
        "Sale or disposal":                                 "summary_only",
        "Adoption of new line(s) of business":              "summary_only",
    })

    # ── Comprehensive subject allowlist — full NSE/BSE taxonomy ───────────
    keep_subjects: frozenset[str] = field(default_factory=lambda: frozenset({
        # Board / orders
        "Outcome of Board Meeting",
        "Board Meeting Intimation",
        "Bagging/Receiving of orders/contracts",
        "Awarding of order(s)/contract(s)",
        # M&A / restructuring
        "Acquisition",
        "Amalgamation/Merger",
        "Demerger",
        "Scheme of Arrangement",
        "Scheme of Amalgamation - NCLT Order",
        "Other Restructuring",
        "Diversification/Disinvestment",
        "Sale or disposal",
        "Adoption of new line(s) of business",
        # Fundraising / capital
        "Qualified Institutional Placement",
        "Rights Issue",
        "Buyback",
        "Public Announcement - Buyback of Shares",
        "Fund Raising - Preferential Issue",
        "Issue of Securities",
        "Allotment of Securities",
        "Options to purchase securities",
        "Conversion",
        "Warrant Conversion",
        "Bonus",
        "Stock split",
        "Utilisation of Funds",
        # Returns to shareholders
        "Dividend",
        "Record Date",
        "Revised Record date",
        # Takeovers / open offers
        "Public Announcement-Open Offer",
        "Disclosure under SEBI Takeover Regulations",
        # Credit / debt
        "Credit Rating- Revision",
        "Credit Rating- New",
        "Defaults on Payment of Interest/Principal",
        "Giving guarantees/indemnity/ becoming a surety for third party",
        # Operational
        "Commencement of commercial production/operations",
        "Capacity addition",
        "Closure of operations",
        "Disruption of Operations",
        "Strikes/Lockouts/Disturbances",
        "Suspension of Trading",
        # Agreements / partnerships
        "Agreements",
        "Memorandum of Understanding/Agreements",
        # Distress / CIRP
        "Corporate Insolvency Resolution Process",
        "Resolution Plan (for CIRP companies)",
        "Delisting",
        # Legal / regulatory
        "Action(s) taken or orders passed",
        "Action(s) initiated or orders passed",
        "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company",
        "Granting/withdrawal/surrender/cancellation/suspension",
        # Management changes
        "Change in Management",
        "Change in Director(s)",
        "Resignation of Director/KMP/SMP",
        "Resignation of Statutory Auditor",
        "Change in Auditors",
        "Appointment",
        "Cessation",
        "Retirement",
        "Demise",
        "Resignation",
        # General / press / verification
        "Press Release",
        "Press Release (Revised)",
        "General Updates",
        "Updates",
        "Monthly Business Updates",
        "News Verification",
        "Rumour Verification - Regulation 30(11)",
        "Spurt in Volume",
        "Price movement",
        "Investor Presentation",
        "Analysts/Institutional Investor Meet/Con. Call Updates",
        # Admin / name changes
        "Name Change",
        "Name and Symbol Change",
        "Reply to Clarification- Financial results",
        "Corrigendum",
    }))
