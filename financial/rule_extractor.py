"""
RuleExtractor — regex-based fast extraction for predictable PDF formats.

Order win announcements, buyback disclosures, and credit rating reports all
follow near-identical templates filed with BSE/NSE. Regex extracts the key
fields in milliseconds with no LLM needed.

Returns None when the patterns don't match — caller falls back to Ollama.
"""
from __future__ import annotations

import re
from financial.models import FinancialResult


# ── Amount normalisation ──────────────────────────────────────────────────────

def _to_crore(value: float, unit: str) -> float:
    u = unit.lower()
    if "lakh" in u or u.startswith("lac"):
        return round(value / 100, 2)
    if "million" in u or " mn" in u:
        return round(value / 10, 2)
    if "billion" in u or " bn" in u:
        return round(value * 100, 2)
    return round(value, 2)   # already crore


def _parse_amount(val_str: str, unit: str) -> float | None:
    try:
        return _to_crore(float(val_str.replace(",", "")), unit)
    except ValueError:
        return None


# ── Shared regex building blocks ──────────────────────────────────────────────

_RS   = r"(?:Rs\.?|INR|₹)\s*"
_UNIT = r"(crore|cr\.?|lakh|lakhs|lacs?|million|mn|billion|bn)"
_AMT  = rf"{_RS}([\d,]+(?:\.\d+)?)\s*{_UNIT}"   # captures (value, unit)

_GOVT = re.compile(
    r"\b(NHAI|NHPC|ONGC|NTPC|SAIL|BHEL|HAL|BEL|DRDO|ISRO|AAI|GAIL|"
    r"Coal\s+India|IREDA|RVNL|IRCON|KRCL|NBCC|CPWD|PWD|NHIDCL|"
    r"Ministry|Government|Railways|Metro|Municipal|Defence|Army|Navy|"
    r"Air\s+Force|Border\s+Roads|State\s+Govt|Central\s+Govt|"
    r"Public\s+Sector|PSU|PSE|"
    # Full names of PSUs often written out instead of abbreviation
    r"Hindustan\s+Aeronautics|Bharat\s+Electronics|Bharat\s+Heavy|"
    r"Oil\s+and\s+Natural\s+Gas|National\s+Thermal\s+Power|"
    r"Steel\s+Authority|Gas\s+Authority|Airports\s+Authority|"
    r"Indian\s+Railways|Indian\s+Navy|Indian\s+Army|Indian\s+Air\s+Force|"
    r"Nuclear\s+Power|NPCIL|MECL|BEML|MIDHANI|GRSE|Mazagon\s+Dock|"
    r"Cochin\s+Shipyard|Garden\s+Reach|Mishra\s+Dhatu)\b",
    re.IGNORECASE,
)

_DURATION = re.compile(
    r"(?:within|over|period\s+of|duration\s+of|to\s+be\s+(?:completed|executed)\s+in)\s*"
    r"([\d]+)\s*(month|year|week|day)",
    re.IGNORECASE,
)


# ── Order win ─────────────────────────────────────────────────────────────────

# Marks cumulative "order book / backlog / pipeline" — NOT a new order win value.
_ORDER_BOOK_KEYWORD = re.compile(
    r"\border[\s-]*(?:book|backlog|pipeline|position|status|inflow)\b",
    re.IGNORECASE,
)

# Sentence boundary: period/!/? followed by whitespace + capital letter.
# "Rs. 960" is NOT a boundary (digit follows), so abbreviations are handled correctly.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _strip_order_book_sentences(text: str) -> str:
    """Remove sentences about order book/backlog so they don't pollute value extraction."""
    sentences = _SENT_SPLIT.split(text)
    return " ".join(s for s in sentences if not _ORDER_BOOK_KEYWORD.search(s))


# Patterns to find the main order value
_ORDER_VALUE_PATTERNS = [
    # "order valued at / worth Rs. 500 crore"
    re.compile(
        rf"(?:order|contract|LOA|letter\s+of\s+award|work\s+order|supply\s+order)"
        rf"[^.]*?(?:valued?\s+at|worth|amounting\s+to|of)\s*{_AMT}",
        re.IGNORECASE,
    ),
    # "total order value / contract value of Rs. 500 crore"
    re.compile(
        rf"(?:total\s+(?:order|contract|project|work)\s+value|"
        rf"order\s+(?:value|size)|contract\s+value)[^.]*?{_AMT}",
        re.IGNORECASE,
    ),
    # General: any "Rs. X crore" near the word 'order'
    re.compile(
        rf"(?:order|contract|project)[^.{{}}]{{0,120}}{_AMT}",
        re.IGNORECASE,
    ),
    # Fallback: largest Rs. X crore in the document
    re.compile(_AMT, re.IGNORECASE),
]

_SECTOR_TAGS = {
    "defence":    re.compile(r"\b(defence|defense|military|army|navy|air force|DRDO|HAL|BEL|ordnance)\b", re.I),
    "railways":   re.compile(r"\b(railways?|RVNL|IRCON|metro|railway ministry)\b", re.I),
    "power":      re.compile(r"\b(power|electricity|solar|wind|energy|NTPC|NHPC)\b", re.I),
    "roads":      re.compile(r"\b(highway|road|bridge|NHAI|expressway|NHIDCL)\b", re.I),
    "water":      re.compile(r"\b(water|irrigation|sewage|sewerage|pipeline)\b", re.I),
    "oil_gas":    re.compile(r"\b(oil|gas|petrochemical|refinery|ONGC|GAIL)\b", re.I),
    "real_estate":re.compile(r"\b(real estate|construction|building|township|residential|commercial)\b", re.I),
    "telecom":    re.compile(r"\b(telecom|5G|fibre|optic|broadband|BSNL)\b", re.I),
}


def extract_order_win_from_table(
    pdf_bytes: bytes,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialResult | None:
    """
    Fast path: parse SEBI Reg 30 Annexure-A table directly from PDF bytes.
    This is the primary extractor for order_win — no LLM, no text regex.
    Falls back to None if table not found (caller tries text regex next).
    """
    from financial.sebi_table_extractor import extract_from_annexure_table

    parsed = extract_from_annexure_table(pdf_bytes, symbol, company, broadcast_dt, source_url)
    if not parsed:
        return None

    client    = parsed["client"]
    value_cr  = parsed["value_cr"]
    duration  = parsed["duration"]
    desc      = parsed["description"]
    value_raw = parsed["value_raw"]

    # Detect client type
    is_govt   = bool(_GOVT.search(client) or _GOVT.search(desc))
    client_type = "Government / PSU" if is_govt else "Private sector"

    # Detect sector from description + client
    combined = f"{client} {desc}"
    sectors  = [tag for tag, rx in _SECTOR_TAGS.items() if rx.search(combined)]
    sector_str = ", ".join(sectors) if sectors else "general"

    highlights = [
        f"Client: {client}",
        f"Order value: {'Rs.' + str(value_cr) + ' Cr' if value_cr else value_raw or 'not disclosed'}",
        f"Client type: {client_type}",
        f"Sector: {sector_str}",
        f"Execution period: {duration}",
    ]
    if desc:
        highlights.append(f"Description: {desc[:120]}")

    value_str = f"Rs.{value_cr:,.0f} Cr" if value_cr else (value_raw or "undisclosed value")
    summary = (
        f"{company} received an order worth {value_str} from {client} "
        f"({client_type.lower()}, {sector_str} sector). "
        f"Execution period: {duration}."
    )

    return FinancialResult(
        symbol         = symbol,
        company        = company,
        period         = broadcast_dt[:10],
        period_type    = "order_win",
        source_url     = source_url,
        broadcast_dt   = broadcast_dt,
        key_highlights = highlights,
        raw_summary    = summary,
    )


def extract_order_win(
    text: str,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialResult | None:
    """
    Extract order win details from announcement text via regex.
    Returns None if no order value is found (caller should try Ollama).
    """
    # 1. Find order value — strip order-book/backlog sentences first so large
    #    cumulative figures (e.g. "Order Book crosses Rs.25,750 Cr") don't
    #    override the actual new order value (e.g. Rs.960 Cr).
    clean_text   = _strip_order_book_sentences(text)
    order_cr     = None
    value_display = ""
    for i, pattern in enumerate(_ORDER_VALUE_PATTERNS):
        matches = pattern.findall(clean_text)
        if matches:
            candidates = []
            for m in matches:
                # m is tuple of capture groups from _AMT: (value_str, unit)
                val = _parse_amount(m[-2], m[-1])   # last 2 groups are always (value, unit)
                if val and val > 0:
                    candidates.append(val)
            if candidates:
                # Specific patterns (0,1) → take smallest specific mention.
                # General/fallback patterns (2,3) → also prefer smaller to avoid
                # picking up total-order-book figures that survived stripping.
                order_cr = min(candidates)
                value_display = f"Rs.{order_cr:,.0f} Cr"
                break

    if not order_cr:
        return None   # can't extract without a value — fall back to Ollama

    # 2. Client type (government / PSU / private)
    is_govt     = bool(_GOVT.search(text))
    client_type = "Government / PSU" if is_govt else "Private sector"

    # 3. Sector detection
    sectors = [tag for tag, rx in _SECTOR_TAGS.items() if rx.search(text)]
    sector_str = ", ".join(sectors) if sectors else "general"

    # 4. Execution duration
    dur = _DURATION.search(text)
    duration = f"{dur.group(1)} {dur.group(2)}s" if dur else "not disclosed"

    # 5. Build highlights + summary
    highlights = [
        f"Order value: {value_display}",
        f"Client type: {client_type}",
        f"Sector: {sector_str}",
        f"Execution period: {duration}",
    ]

    summary = (
        f"{company} ({symbol}) received an order worth {value_display} "
        f"from a {client_type.lower()} client in the {sector_str} sector. "
        f"Execution period: {duration}."
    )

    return FinancialResult(
        symbol         = symbol,
        company        = company,
        period         = broadcast_dt[:10],
        period_type    = "order_win",
        source_url     = source_url,
        broadcast_dt   = broadcast_dt,
        key_highlights = highlights,
        raw_summary    = summary,
    )


# ── Buyback ───────────────────────────────────────────────────────────────────

_BUYBACK_SIZE    = re.compile(rf"buyback\s+size[^.]*?{_AMT}|{_AMT}[^.]*?buyback", re.I)
_BUYBACK_PRICE   = re.compile(r"maximum\s+(?:buyback\s+)?price[^.]*?(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+(?:equity\s+)?share)?", re.I)
_BUYBACK_METHOD  = re.compile(r"(open\s+market|tender\s+offer|tender offer)", re.I)


def extract_buyback(
    text: str,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialResult | None:
    size_cr = None
    for m in _BUYBACK_SIZE.findall(text):
        v = _parse_amount(m[-2], m[-1])
        if v:
            size_cr = v
            break

    if not size_cr:
        return None

    price_m  = _BUYBACK_PRICE.search(text)
    method_m = _BUYBACK_METHOD.search(text)

    max_price = f"Rs.{price_m.group(1)}/share" if price_m else "not disclosed"
    method    = method_m.group(1).title()      if method_m else "not disclosed"

    highlights = [
        f"Buyback size: Rs.{size_cr:,.0f} Cr",
        f"Max price: {max_price}",
        f"Method: {method}",
    ]
    summary = (
        f"{company} announced a buyback of Rs.{size_cr:,.0f} Cr via {method} "
        f"at a maximum price of {max_price}."
    )

    return FinancialResult(
        symbol=symbol, company=company,
        period=broadcast_dt[:10], period_type="buyback",
        source_url=source_url, broadcast_dt=broadcast_dt,
        key_highlights=highlights, raw_summary=summary,
    )


# ── Credit rating ─────────────────────────────────────────────────────────────

_RATING_PATTERN = re.compile(
    r"(?:rated?|rating|assigned?|reaffirmed?|upgraded?|downgraded?|revised?)"
    r"[^.]{0,80}?"
    r"((?:CRISIL|ICRA|CARE|India\s+Ratings|Brickwork|Acuite)\s+)"
    r"([A-D][A-D+\-]*(?:\s*/\s*[A-D][A-D+\-]*)?)"
    r"[^.]{0,40}?"
    r"(Stable|Positive|Negative|Watch|Developing)?",
    re.IGNORECASE,
)
_PREV_RATING     = re.compile(r"(?:from|previous(?:ly)?|earlier)\s+([A-D][A-D+\-]+)", re.I)


def extract_credit_rating(
    text: str,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialResult | None:
    m = _RATING_PATTERN.search(text)
    if not m:
        return None

    agency   = m.group(1).strip()
    rating   = m.group(2).strip()
    outlook  = m.group(3).strip() if m.group(3) else "not stated"
    prev_m   = _PREV_RATING.search(text)
    prev     = prev_m.group(1) if prev_m else None

    highlights = [f"Rating: {agency} {rating} / Outlook: {outlook}"]
    if prev:
        direction = "upgrade" if rating > prev else "downgrade"
        highlights.append(f"Previous rating: {prev} ({direction})")

    summary = (
        f"{company} assigned {agency} {rating} with {outlook} outlook"
        + (f", revised from {prev}." if prev else ".")
    )

    return FinancialResult(
        symbol=symbol, company=company,
        period=broadcast_dt[:10], period_type="credit_rating",
        source_url=source_url, broadcast_dt=broadcast_dt,
        key_highlights=highlights, raw_summary=summary,
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────

_HANDLERS = {
    "order_win":     extract_order_win,
    # Press releases often contain order wins — try the same regex; returns None if no value found.
    "press_release": extract_order_win,
    "buyback":       extract_buyback,
    "credit_rating": extract_credit_rating,
}


def try_rule_extract(
    extraction_type: str,
    text: str,
    symbol: str,
    company: str,
    broadcast_dt: str,
    source_url: str,
) -> FinancialResult | None:
    """
    Try to extract using rules. Returns None if no handler exists or patterns
    don't match — caller should fall back to Ollama.
    """
    handler = _HANDLERS.get(extraction_type)
    if handler is None:
        return None
    return handler(text, symbol, company, broadcast_dt, source_url)
