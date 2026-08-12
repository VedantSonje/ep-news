"""
MaterialityExtractor — extracts structured financial context from the
announcement details to answer: "how big is this event, really?"

Outputs
-------
  order_value_cr      raw order value extracted from text
  relative_size       small / medium / large / mega
  catalyst_type       order_win / acquisition / commissioning /
                      results / dividend / management / regulatory
  yoy_surprise        positive / negative / neutral (from text signals)
  government_linked   True if PSU/ministry involved
  export_component    True if export mentioned
  strategic_note      one-line context note for the agent

Production upgrade path
-----------------------
  Replace relative_size with (order_value / TTM_revenue) once you
  plug in a market-data feed.  Currently we use industry medians:
    SME:   median quarterly rev ~50 cr  → large = >25% = >12 cr
    Mid:   median quarterly rev ~500 cr → large = >25% = >125 cr
    Large: median quarterly rev ~5000cr → large = >25% = >1250 cr
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class MaterialityInfo:
    order_value_cr:    float | None = None
    relative_size:     str = "unknown"        # tiny/small/medium/large/mega
    catalyst_type:     str = "general"
    yoy_surprise:      str = "neutral"        # positive/negative/neutral
    government_linked: bool = False
    export_component:  bool = False
    strategic_note:    str = ""


# ── Regex patterns ────────────────────────────────────────────────────────

_CRORE = re.compile(
    r"Rs\.?\s*([\d,\.]+)\s*(?:crore|cr\.?)", re.IGNORECASE
)
_LAKH  = re.compile(
    r"Rs\.?\s*([\d,\.]+)\s*(?:lakh|lac)", re.IGNORECASE
)
_MILLION_FX = re.compile(
    r"(?:USD|EUR|GBP)\s*([\d,\.]+)\s*(?:million|mn|m)\b", re.IGNORECASE
)
_INR_MILLION = re.compile(
    r"Rs\.?\s*([\d,\.]+)\s*(?:million|mn)\b", re.IGNORECASE
)
_POSITIVE_SURPRISE = re.compile(
    r"highest.ever|record|beat|outperform|surge|jump|soar|"
    r"strong\s+growth|exceed|above\s+estimate|"
    r"YoY\s+growth|up\s+\d+%|grew\s+\d+%",
    re.IGNORECASE,
)
_NEGATIVE_SURPRISE = re.compile(
    r"miss|below\s+estimate|decline|fall|drop|loss|"
    r"YoY\s+de.?growth|down\s+\d+%|shrink",
    re.IGNORECASE,
)
_GOVT_LINKED = re.compile(
    r"government|ministry|PSU|public sector|CPSE|"
    r"Jal Jeevan|PMGSY|NHAI|RVNL|NTPC|NHPC|DRDO|ISRO|"
    r"defence ministry|state government|L1",
    re.IGNORECASE,
)
_EXPORT = re.compile(
    r"export|overseas|international order|Israel|"
    r"Middle East|Africa|ASEAN|Europe|US order|USD order",
    re.IGNORECASE,
)
_CATALYST_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"order|contract|LOA|letter of award|tender", re.I), "order_win"),
    (re.compile(r"acquisition|acquire|takeover|stake purchase|merger", re.I), "acquisition"),
    (re.compile(r"commissioning|commercial production|plant.*commission|"
                r"capacity.*added|new capacity", re.I), "commissioning"),
    (re.compile(r"financial results|quarterly results|audited results|"
                r"annual results", re.I), "results"),
    (re.compile(r"dividend|interim dividend|final dividend", re.I), "dividend"),
    (re.compile(r"CEO|CFO|MD|Managing Director|Chairman|resignation|"
                r"appointment", re.I), "management"),
    (re.compile(r"ANDA|FDA|USFDA|WHO PQ|regulatory approval|"
                r"drug approval|patent", re.I), "regulatory"),
]


def _parse_value(text: str) -> float | None:
    """Extract first monetary value and normalise to Rs. crore."""
    m = _CRORE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    m = _LAKH.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "")) / 100
        except ValueError:
            pass
    m = _INR_MILLION.search(text)
    if m:
        try:
            return round(float(m.group(1).replace(",", "")) / 100, 1)  # INR Mn → Cr
        except ValueError:
            pass
    m = _MILLION_FX.search(text)
    if m:
        try:
            fx_m = float(m.group(1).replace(",", ""))
            return round(fx_m * 8.4, 1)    # ~USD/INR 84
        except ValueError:
            pass
    return None


def _relative_size(value_cr: float | None) -> str:
    if value_cr is None:
        return "unknown"
    if value_cr >= 1000:
        return "mega"
    if value_cr >= 500:
        return "large"
    if value_cr >= 100:
        return "medium"
    if value_cr >= 20:
        return "small"
    return "tiny"


def _catalyst_type(subject: str, details: str) -> str:
    text = f"{subject} {details}"
    for pattern, ctype in _CATALYST_MAP:
        if pattern.search(text):
            return ctype
    return "general"


class MaterialityExtractor:
    """
    Extracts materiality context from an announcement's subject + details.
    All operations are pure Python regex — no network or LLM calls.
    """

    def extract(
        self, subject: str, details: str, company: str = ""
    ) -> MaterialityInfo:
        text = f"{subject} {details} {company}"
        val  = _parse_value(details)

        # YoY surprise signal
        if _POSITIVE_SURPRISE.search(text):
            surprise = "positive"
        elif _NEGATIVE_SURPRISE.search(text):
            surprise = "negative"
        else:
            surprise = "neutral"

        # Strategic note
        notes = []
        if val and val >= 1000:
            notes.append(f"Rs.{val:.0f} cr order — mega size")
        elif val and val >= 100:
            notes.append(f"Rs.{val:.0f} cr order")

        govt = bool(_GOVT_LINKED.search(text))
        exp  = bool(_EXPORT.search(text))
        if govt:
            notes.append("government/PSU contract")
        if exp:
            notes.append("export component")
        if surprise == "positive":
            notes.append("positive surprise")

        return MaterialityInfo(
            order_value_cr    = val,
            relative_size     = _relative_size(val),
            catalyst_type     = _catalyst_type(subject, details),
            yoy_surprise      = surprise,
            government_linked = govt,
            export_component  = exp,
            strategic_note    = " · ".join(notes) if notes else "",
        )
