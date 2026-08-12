#!/usr/bin/env python3
"""
EP Screener — BSE / NSE corporate announcements filter
Stockbee-style Episodic Pivot pre-screen for Indian equities.

Usage:
    python screener.py announcements.csv
    python screener.py announcements.csv --verbose
    python screener.py announcements.csv --min-score 6 --csv-out ep_today.csv
"""

import csv
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ── Subject-level filter lists ────────────────────────────────────────────────

KEEP_SUBJECTS = {
    "Outcome of Board Meeting",
    "Bagging/Receiving of orders/contracts",
    "Acquisition",
    "Commencement of commercial production/operations",
    "Capacity addition",
    "Press Release",
    "Press Release (Revised)",
    "General Updates",
    "Updates",
    "Scheme of Arrangement",
    "Sale or disposal",
    "Dividend",
    "Change in Management",
    "Appointment",
    "Cessation",
    "Resignation",
}

# These are always noise regardless of details
DROP_SUBJECTS = {
    "Copy of Newspaper Publication",
    "Trading Window",
    "Disclosure under SEBI Takeover Regulations",
    "Analysts/Institutional Investor Meet/Con. Call Updates",
    "ESOP/ESOS/ESPS",
    "Record Date",
    "Monitoring Agency Report",
    "Statement of deviation(s) or variation(s) under Reg. 32",
    "Clarification - Financial Results",
    "Reply to Clarification- Financial results",
    "Shareholders meeting",
    "Registrar & Share Transfer Agent Update",
    "Investor Presentation",
    "Allotment of Securities",
    "Options to purchase securities",
    "Credit Rating",
    "Action(s) initiated or orders passed",
    "Price movement",
    "Spurt in Volume",
    "Forfeiture",
    "Corporate Insolvency Resolution Process",
    "Disclosure of material issue",
    "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company",
    "Resignation of Director/KMP/SMP",
    "Change in Director(s)",
    "Change in Company Secretary/Compliance Officer",
    "Allotment",
    "Giving guarantees/indemnity/ becoming a surety for third party",
}

# ── Keyword patterns for content scoring ─────────────────────────────────────

_HIGH_VALUE = re.compile(
    r"order[s]?\s*(received|won|bagged|secured|awarded|bag)"
    r"|contract[s]?\s*(received|won|bagged|secured|awarded)"
    r"|commission(ing|ed|s)?"
    r"|commercial\s+production"
    r"|government|ministry|defence|defense"
    r"|tender|bid\s+win"
    r"|export\s+order"
    r"|joint\s+venture|collaboration"
    r"|\bMOU\b|memorandum\s+of\s+understanding"
    r"|plant\s+(commission|capacity|expansion|launch)"
    r"|new\s+(plant|facility|unit|product|line)"
    r"|USFDA|US\s*FDA|\bFDA\b|\bWHO\b|\bANDA\b|\bNDA\b"
    r"|drug\s+approval|regulatory\s+approval|patent"
    r"|acquisition|merger|amalgamation|demerger|takeover|spin.?off"
    r"|buyback|QIP|fund.?rais|stake\s+(acqui|purchas|increas)"
    r"|letter\s+of\s+(award|intent)|LOA\b"
    r"|strategic\s+(partner|acqui|tie.?up)"
    r"|subsidiary.*incorporat|new.*market"
    r"|Rs\.?\s*\d[\d,\.]*\s*(?:crore|cr\.?|billion|lakh)"
    r"|₹\s*\d",
    re.IGNORECASE,
)

_LOW_VALUE = re.compile(
    r"trading\s+window|newspaper|IEPF|demat\s+report"
    r"|KYC|scrutinizer|postal\s+ballot|annual\s+report"
    r"|100.?days|saksham|reminder\s+letter"
    r"|non.?applicab|large\s+entity|compliance\s+report"
    r"|board\s+comments\s+on\s+fine|security\s+cover"
    r"|demat\s+report|physical\s+shares|IEPF",
    re.IGNORECASE,
)

_RESULTS = re.compile(
    r"financial\s+results|audited\s+results|quarterly\s+results"
    r"|annual\s+results|submitted.*financial|financial.*period\s+ended",
    re.IGNORECASE,
)

_CSUITE = re.compile(
    r"Managing Director|Chief Executive|\bMD\b|\bCEO\b|\bCFO\b"
    r"|Chief Financial|\bCTO\b|Chief Technology|Chairman",
    re.IGNORECASE,
)

_CRORE_VAL = re.compile(
    r"Rs\.?\s*([\d,\.]+)\s*(?:crore|cr\.?)", re.IGNORECASE
)

_DIV_VAL = re.compile(
    r"(?:Rs\.?|₹)\s*([\d\.]+)\s*per\s+(?:equity\s+)?share", re.IGNORECASE
)

# ── Catalyst scoring ──────────────────────────────────────────────────────────

_BASE_SCORE = {
    "Bagging/Receiving of orders/contracts": 10,
    "Acquisition":                            9,
    "Commencement of commercial production/operations": 9,
    "Scheme of Arrangement":                  8,
    "Capacity addition":                      8,
    "Outcome of Board Meeting":               6,
    "Sale or disposal":                       6,
    "Press Release":                          5,
    "Press Release (Revised)":                5,
    "General Updates":                        3,
    "Updates":                                3,
    "Dividend":                               4,
    "Change in Management":                   2,
    "Appointment":                            2,
    "Cessation":                              1,
    "Resignation":                            1,
}


def score(subject: str, details: str) -> tuple[int, list[str]]:
    """Return (catalyst_score, [reason_tags]) for a single announcement."""
    s = subject.strip()
    reasons: list[str] = []
    pts = _BASE_SCORE.get(s, 2)

    # ── Outcome of Board Meeting ──────────────────────────────────────────
    if s == "Outcome of Board Meeting":
        if _RESULTS.search(details):
            pts = 8
            reasons.append("financial results")
        if _HIGH_VALUE.search(details):
            pts = max(pts, 9)
            reasons.append("high-value content")

    # ── General Updates / Updates / Press Release ─────────────────────────
    elif s in ("General Updates", "Updates", "Press Release", "Press Release (Revised)"):
        if _LOW_VALUE.search(details):
            return 0, ["noise"]
        if _HIGH_VALUE.search(details):
            pts = 8
            reasons.append("high-value content")
        else:
            pts = 3

    # ── Orders / contracts ────────────────────────────────────────────────
    elif s == "Bagging/Receiving of orders/contracts":
        m = _CRORE_VAL.search(details)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                reasons.append(f"Rs.{m.group(1)} cr")
                if val >= 1000:
                    pts = 13
                elif val >= 500:
                    pts = 12
                elif val >= 100:
                    pts = 11
            except ValueError:
                pass

    # ── Dividend ──────────────────────────────────────────────────────────
    elif s == "Dividend":
        m = _DIV_VAL.search(details)
        if m:
            try:
                amt = float(m.group(1))
                reasons.append(f"Rs.{amt}/share")
                pts = 5 if amt >= 5 else (4 if amt >= 1 else 2)
            except ValueError:
                pass

    # ── Management changes ────────────────────────────────────────────────
    elif s in ("Appointment", "Change in Management", "Cessation", "Resignation"):
        if _CSUITE.search(details):
            pts = 5
            reasons.append("C-suite")
        else:
            pts = 1  # sub-threshold — will be filtered by min_score

    return pts, reasons


# ── Column detection ──────────────────────────────────────────────────────────

_COL_CANDIDATES: dict[str, list[str]] = {
    "symbol":     ["symbol", "scrip code", "scripcode", "security code", "nse symbol"],
    "company":    ["company name", "company", "security name", "scrip name", "issuer name"],
    "subject":    ["subject", "headline", "category", "announcement type", "purpose"],
    "details":    ["details", "description", "body", "announcement", "content"],
    "datetime":   [
        "broadcast date/time", "broadcast date", "date time", "date",
        "datetime", "submitted date/time", "submission date", "announcement date",
    ],
    "attachment": ["attachment", "pdf", "link", "url", "file"],
}


def detect_columns(headers: list[str]) -> dict[str, str]:
    lower = [h.lower().strip() for h in headers]
    out: dict[str, str] = {}
    for field, opts in _COL_CANDIDATES.items():
        for opt in opts:
            if opt in lower:
                out[field] = headers[lower.index(opt)]
                break
    return out


# ── Date parsing ──────────────────────────────────────────────────────────────

_DT_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y %H:%M",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%Y-%m-%d",
)


def parse_dt(s: str) -> datetime | None:
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


# ── Main screener ─────────────────────────────────────────────────────────────

def run(csv_path: str, min_score: int = 4) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows: list[dict] = []

    with open(path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        col = detect_columns(headers)

        if "subject" not in col:
            raise ValueError(
                f"Cannot locate a 'subject' column.\n"
                f"Detected headers: {headers}"
            )

        for row in reader:
            subject  = row.get(col.get("subject",  ""), "").strip()
            details  = row.get(col.get("details",  ""), "").strip()
            symbol   = row.get(col.get("symbol",   ""), "").strip()
            company  = row.get(col.get("company",  ""), "").strip()
            dt_raw   = row.get(col.get("datetime", ""), "").strip()
            attach   = row.get(col.get("attachment",""), "").strip()

            # subject-level gate
            if subject in DROP_SUBJECTS:
                continue
            if subject not in KEEP_SUBJECTS:
                continue

            pts, reasons = score(subject, details)
            if pts < min_score:
                continue

            dt = parse_dt(dt_raw)
            rows.append({
                "score":      pts,
                "symbol":     symbol,
                "company":    company,
                "subject":    subject,
                "tags":       ", ".join(reasons) if reasons else subject,
                "details":    details,
                "datetime":   dt.strftime("%Y-%m-%d %H:%M") if dt else dt_raw,
                "attachment": attach,
            })

    rows.sort(key=lambda r: (-r["score"], r["datetime"]))
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def _p(text: str) -> None:
    """Print with safe fallback for narrow-codec terminals (Windows cp1252 etc.)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def print_report(rows: list[dict], verbose: bool = False) -> None:
    if not rows:
        _p("\nNo EP candidates found after filtering.\n")
        return

    sep = "=" * 88
    _p(f"\n{sep}")
    _p(f"  EP SCREENER  --  {len(rows)} candidate(s)  |  ranked by catalyst strength")
    _p(f"{sep}\n")

    for r in rows:
        bar       = "#" * min(r["score"], 13)
        score_tag = f"[{r['score']:>2}]"
        sym       = r["symbol"][:12].ljust(12)
        co        = r["company"][:34].ljust(34)
        _p(f"{score_tag} {bar:<13}  {sym} {co}")
        _p(f"       {r['datetime']}  |  {r['tags']}")
        if verbose:
            detail_line = r["details"][:300] + ("..." if len(r["details"]) > 300 else "")
            _p(f"       {detail_line}")
            if r["attachment"] and r["attachment"] not in ("-", ""):
                _p(f"       >> {r['attachment']}")
        _p("")


def save_csv(rows: list[dict], out_path: str) -> None:
    p = Path(out_path)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {len(rows)} rows → {p}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EP Screener — filters BSE/NSE announcements for episodic pivot candidates",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("csv_file", help="Path to BSE/NSE corporate announcements CSV")
    parser.add_argument(
        "--min-score", type=int, default=4, metavar="N",
        help="Minimum catalyst score to include (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full announcement details and PDF links",
    )
    parser.add_argument(
        "--csv-out", metavar="FILE",
        help="Also write filtered results to this CSV path",
    )
    args = parser.parse_args()

    try:
        results = run(args.csv_file, min_score=args.min_score)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(results, verbose=args.verbose)

    if args.csv_out:
        save_csv(results, args.csv_out)


if __name__ == "__main__":
    main()
