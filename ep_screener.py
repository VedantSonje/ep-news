#!/usr/bin/env python3
"""
EP Screener  —  Stockbee-style Episodic Pivot filter for Indian equity announcements.
Supports both BSE and NSE corporate announcement CSV exports.

Classes
-------
FilterConfig      : Immutable configuration — subject lists, keyword regexes, base scores.
Announcement      : Data-only representation of one CSV row.
ScoreResult       : Output of the scoring engine — numeric score + human-readable tags.
ColumnMapper      : Maps BSE/NSE header variants to canonical field names.
CsvParser         : Reads a CSV file and yields Announcement objects.
AnnouncementFilter: Subject-level keep/drop gate (uses FilterConfig).
ScoringEngine     : Scores one Announcement (uses FilterConfig).
EPScreener        : Pipeline orchestrator — filter → score → sort.
ReportPrinter     : Renders results to stdout.
CsvExporter       : Saves results to a CSV file.
ScreenerApp       : CLI entry point — wires all components together.
"""

from __future__ import annotations

import csv
import re
import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


# ──────────────────────────────────────────────────────────────────────────────
# FilterConfig  —  single source of truth for every rule and score
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FilterConfig:
    """
    Immutable configuration container.
    All sets/dicts/patterns are computed once at construction and shared
    across the Filter and Scoring engines.
    """

    # Subjects that are always dropped before any scoring
    drop_subjects: frozenset[str] = field(default_factory=lambda: frozenset({
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
        "Monthly Business Updates",
    }))

    # Subjects that pass the first gate and may be scored
    keep_subjects: frozenset[str] = field(default_factory=lambda: frozenset({
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
    }))

    # Base catalyst score by subject (before detail-level adjustments)
    base_scores: dict[str, int] = field(default_factory=lambda: {
        "Bagging/Receiving of orders/contracts":              10,
        "Acquisition":                                         9,
        "Commencement of commercial production/operations":    9,
        "Scheme of Arrangement":                               8,
        "Capacity addition":                                   8,
        "Outcome of Board Meeting":                            6,
        "Sale or disposal":                                    6,
        "Press Release":                                       5,
        "Press Release (Revised)":                             5,
        "General Updates":                                     3,
        "Updates":                                             3,
        "Dividend":                                            4,
        "Change in Management":                                2,
        "Appointment":                                         2,
        "Cessation":                                           1,
        "Resignation":                                         1,
    })

    # Compiled regexes (built once)
    re_high_value: re.Pattern = field(init=False)
    re_low_value: re.Pattern = field(init=False)
    re_results: re.Pattern = field(init=False)
    re_csuite: re.Pattern = field(init=False)
    re_crore: re.Pattern = field(init=False)
    re_dividend: re.Pattern = field(init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "re_high_value", re.compile(
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
            r"|letter\s+of\s+(award|intent)|\bLOA\b"
            r"|strategic\s+(partner|acqui|tie.?up)"
            r"|subsidiary.*incorporat|new.*market"
            r"|Rs\.?\s*\d[\d,\.]*\s*(?:crore|cr\.?|billion|lakh)"
            r"|[Rr]s\.\s*\d",
            re.IGNORECASE,
        ))
        object.__setattr__(self, "re_low_value", re.compile(
            r"trading\s+window|newspaper|IEPF|demat\s+report"
            r"|KYC|scrutinizer|postal\s+ballot|annual\s+report"
            r"|100.?days|saksham|reminder\s+letter"
            r"|non.?applicab|large\s+entity|compliance\s+report"
            r"|board\s+comments\s+on\s+fine|security\s+cover"
            r"|physical\s+shares",
            re.IGNORECASE,
        ))
        object.__setattr__(self, "re_results", re.compile(
            r"financial\s+results|audited\s+results|quarterly\s+results"
            r"|annual\s+results|submitted.*financial|financial.*period\s+ended",
            re.IGNORECASE,
        ))
        object.__setattr__(self, "re_csuite", re.compile(
            r"Managing Director|Chief Executive|\bMD\b|\bCEO\b|\bCFO\b"
            r"|Chief Financial|\bCTO\b|Chief Technology|Chairman",
            re.IGNORECASE,
        ))
        object.__setattr__(self, "re_crore", re.compile(
            r"Rs\.?\s*([\d,\.]+)\s*(?:crore|cr\.?)", re.IGNORECASE
        ))
        object.__setattr__(self, "re_dividend", re.compile(
            r"(?:Rs\.?|[Rr]s\.)\s*([\d\.]+)\s*per\s+(?:equity\s+)?share",
            re.IGNORECASE,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# Announcement  —  data-only, one CSV row
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Announcement:
    """Represents a single corporate announcement row from the CSV."""
    symbol: str
    company: str
    subject: str
    details: str
    broadcast_dt: datetime | None
    broadcast_raw: str
    attachment: str


# ──────────────────────────────────────────────────────────────────────────────
# ScoreResult  —  output of ScoringEngine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Holds the numeric catalyst score and human-readable reason tags."""
    score: int
    tags: list[str]

    @property
    def tag_str(self) -> str:
        return ", ".join(self.tags) if self.tags else "-"


# ──────────────────────────────────────────────────────────────────────────────
# ColumnMapper  —  handles BSE / NSE header naming differences
# ──────────────────────────────────────────────────────────────────────────────

class ColumnMapper:
    """
    Maps actual CSV header strings to canonical field names.
    Handles both BSE and NSE column naming conventions.
    """

    _CANDIDATES: dict[str, list[str]] = {
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

    def __init__(self, headers: list[str]) -> None:
        lower = [h.lower().strip() for h in headers]
        self._map: dict[str, str] = {}
        for field_name, candidates in self._CANDIDATES.items():
            for candidate in candidates:
                if candidate in lower:
                    self._map[field_name] = headers[lower.index(candidate)]
                    break

    def get(self, field_name: str) -> str | None:
        """Return the actual header string for a canonical field name."""
        return self._map.get(field_name)

    def validate(self) -> None:
        """Raise ValueError if the mandatory 'subject' column is missing."""
        if "subject" not in self._map:
            raise ValueError(
                "Cannot locate a 'subject' column in the CSV.\n"
                f"Mapped fields found: {list(self._map.keys())}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# CsvParser  —  reads file, yields Announcement objects
# ──────────────────────────────────────────────────────────────────────────────

class CsvParser:
    """
    Reads a BSE/NSE corporate announcements CSV and yields Announcement objects.
    Auto-detects column layout via ColumnMapper.
    """

    _DT_FORMATS = (
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%Y-%m-%d",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found: {self.path}")

    # ── public ────────────────────────────────────────────────────────────────

    def parse(self) -> Iterator[Announcement]:
        """Yield Announcement objects for every data row in the CSV."""
        with open(self.path, encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            col = ColumnMapper(list(reader.fieldnames or []))
            col.validate()
            for row in reader:
                yield self._build(row, col)

    # ── private ───────────────────────────────────────────────────────────────

    def _build(self, row: dict[str, str], col: ColumnMapper) -> Announcement:
        dt_raw = row.get(col.get("datetime") or "", "").strip()
        return Announcement(
            symbol      = row.get(col.get("symbol")     or "", "").strip(),
            company     = row.get(col.get("company")    or "", "").strip(),
            subject     = row.get(col.get("subject")    or "", "").strip(),
            details     = row.get(col.get("details")    or "", "").strip(),
            attachment  = row.get(col.get("attachment") or "", "").strip(),
            broadcast_raw = dt_raw,
            broadcast_dt  = self._parse_dt(dt_raw),
        )

    @staticmethod
    def _parse_dt(s: str) -> datetime | None:
        for fmt in CsvParser._DT_FORMATS:
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        return None


# ──────────────────────────────────────────────────────────────────────────────
# AnnouncementFilter  —  subject-level keep / drop gate
# ──────────────────────────────────────────────────────────────────────────────

class AnnouncementFilter:
    """
    First-pass filter based purely on the announcement subject.
    Drops noise subjects before the (more expensive) scoring step.
    """

    def __init__(self, config: FilterConfig) -> None:
        self._config = config

    def should_keep(self, ann: Announcement) -> bool:
        """Return True if the announcement should proceed to scoring."""
        subject = ann.subject
        if subject in self._config.drop_subjects:
            return False
        if subject not in self._config.keep_subjects:
            return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# ScoringEngine  —  assigns catalyst score to one Announcement
# ──────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Produces a ScoreResult for a single Announcement.
    Scoring rules are driven by FilterConfig patterns and base scores.
    Each announcement subject has its own dedicated scoring method.
    """

    def __init__(self, config: FilterConfig) -> None:
        self._cfg = config

    # ── public ────────────────────────────────────────────────────────────────

    def score(self, ann: Announcement) -> ScoreResult:
        """Dispatch to the correct subject handler and return a ScoreResult."""
        subject = ann.subject
        handler = self._HANDLERS.get(subject, self._generic)
        return handler(self, ann)

    # ── subject handlers ──────────────────────────────────────────────────────

    def _board_meeting(self, ann: Announcement) -> ScoreResult:
        pts = self._cfg.base_scores.get(ann.subject, 4)
        tags: list[str] = []
        if self._cfg.re_results.search(ann.details):
            pts = max(pts, 8)
            tags.append("financial results")
        if self._cfg.re_high_value.search(ann.details):
            pts = max(pts, 9)
            tags.append("high-value content")
        return ScoreResult(pts, tags or [ann.subject])

    def _orders(self, ann: Announcement) -> ScoreResult:
        pts = self._cfg.base_scores.get(ann.subject, 10)
        tags: list[str] = ["order/contract"]
        match = self._cfg.re_crore.search(ann.details)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                tags.append(f"Rs.{match.group(1)} cr")
                if val >= 1000:
                    pts = 13
                elif val >= 500:
                    pts = 12
                elif val >= 100:
                    pts = 11
            except ValueError:
                pass
        return ScoreResult(pts, tags)

    def _general_or_press(self, ann: Announcement) -> ScoreResult:
        if self._cfg.re_low_value.search(ann.details):
            return ScoreResult(0, ["noise"])
        if self._cfg.re_high_value.search(ann.details):
            return ScoreResult(8, ["high-value content"])
        return ScoreResult(3, [ann.subject])

    def _dividend(self, ann: Announcement) -> ScoreResult:
        tags: list[str] = []
        pts = self._cfg.base_scores.get(ann.subject, 4)
        match = self._cfg.re_dividend.search(ann.details)
        if match:
            try:
                amt = float(match.group(1))
                tags.append(f"Rs.{amt}/share")
                pts = 5 if amt >= 5 else (4 if amt >= 1 else 2)
            except ValueError:
                pass
        return ScoreResult(pts, tags or ["dividend"])

    def _management(self, ann: Announcement) -> ScoreResult:
        if self._cfg.re_csuite.search(ann.details):
            return ScoreResult(5, ["C-suite change"])
        return ScoreResult(1, ["non-senior"])

    def _commissioning(self, ann: Announcement) -> ScoreResult:
        return ScoreResult(
            self._cfg.base_scores.get(ann.subject, 9),
            ["plant/capacity commissioning"],
        )

    def _acquisition(self, ann: Announcement) -> ScoreResult:
        pts = self._cfg.base_scores.get(ann.subject, 9)
        tags = ["acquisition"]
        match = self._cfg.re_crore.search(ann.details)
        if match:
            tags.append(f"Rs.{match.group(1)} cr")
            try:
                val = float(match.group(1).replace(",", ""))
                if val >= 500:
                    pts = 11
                elif val >= 100:
                    pts = 10
            except ValueError:
                pass
        return ScoreResult(pts, tags)

    def _scheme(self, ann: Announcement) -> ScoreResult:
        return ScoreResult(
            self._cfg.base_scores.get(ann.subject, 8),
            ["scheme of arrangement / merger"],
        )

    def _capacity(self, ann: Announcement) -> ScoreResult:
        return ScoreResult(
            self._cfg.base_scores.get(ann.subject, 8),
            ["capacity expansion"],
        )

    def _generic(self, ann: Announcement) -> ScoreResult:
        pts = self._cfg.base_scores.get(ann.subject, 3)
        if self._cfg.re_high_value.search(ann.details):
            pts = max(pts, 7)
            return ScoreResult(pts, ["high-value content"])
        return ScoreResult(pts, [ann.subject])

    # ── dispatch table ────────────────────────────────────────────────────────
    # Maps subject strings to bound handler methods.

    _HANDLERS: dict[str, object] = {
        "Outcome of Board Meeting":                            _board_meeting,
        "Bagging/Receiving of orders/contracts":               _orders,
        "Press Release":                                       _general_or_press,
        "Press Release (Revised)":                             _general_or_press,
        "General Updates":                                     _general_or_press,
        "Updates":                                             _general_or_press,
        "Dividend":                                            _dividend,
        "Change in Management":                                _management,
        "Appointment":                                         _management,
        "Cessation":                                           _management,
        "Resignation":                                         _management,
        "Commencement of commercial production/operations":    _commissioning,
        "Acquisition":                                         _acquisition,
        "Scheme of Arrangement":                               _scheme,
        "Capacity addition":                                   _capacity,
        "Sale or disposal":                                    _generic,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ScoredAnnouncement  —  pairs an Announcement with its ScoreResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredAnnouncement:
    """An Announcement paired with its catalyst ScoreResult."""
    announcement: Announcement
    result: ScoreResult

    @property
    def score(self) -> int:
        return self.result.score

    @property
    def dt_str(self) -> str:
        dt = self.announcement.broadcast_dt
        return dt.strftime("%Y-%m-%d %H:%M") if dt else self.announcement.broadcast_raw


# ──────────────────────────────────────────────────────────────────────────────
# EPScreener  —  pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class EPScreener:
    """
    Orchestrates the full pipeline:
      CsvParser  ->  AnnouncementFilter  ->  ScoringEngine  ->  sort
    Returns a ranked list of ScoredAnnouncement objects.
    """

    def __init__(
        self,
        parser: CsvParser,
        ann_filter: AnnouncementFilter,
        scorer: ScoringEngine,
    ) -> None:
        self._parser = parser
        self._filter = ann_filter
        self._scorer = scorer

    def run(self, min_score: int = 4) -> list[ScoredAnnouncement]:
        """
        Execute the pipeline and return results sorted by score (desc),
        then by broadcast time (desc).
        """
        results: list[ScoredAnnouncement] = []

        for ann in self._parser.parse():
            if not self._filter.should_keep(ann):
                continue
            result = self._scorer.score(ann)
            if result.score < min_score:
                continue
            results.append(ScoredAnnouncement(ann, result))

        results.sort(
            key=lambda sa: (-sa.score, sa.announcement.broadcast_raw),
        )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# ReportPrinter  —  terminal output
# ──────────────────────────────────────────────────────────────────────────────

class ReportPrinter:
    """Renders a list of ScoredAnnouncement objects to stdout."""

    _WIDTH = 88

    def print(self, items: list[ScoredAnnouncement], verbose: bool = False) -> None:
        if not items:
            self._out("\nNo EP candidates found after filtering.\n")
            return

        self._out(f"\n{'=' * self._WIDTH}")
        self._out(
            f"  EP SCREENER  --  {len(items)} candidate(s)"
            f"  |  ranked by catalyst strength"
        )
        self._out(f"{'=' * self._WIDTH}\n")

        for sa in items:
            ann  = sa.announcement
            bar  = "#" * min(sa.score, 13)
            sym  = ann.symbol[:12].ljust(12)
            co   = ann.company[:34].ljust(34)
            self._out(f"[{sa.score:>2}] {bar:<13}  {sym} {co}")
            self._out(f"       {sa.dt_str}  |  {sa.result.tag_str}")
            if verbose:
                detail = ann.details[:300] + ("..." if len(ann.details) > 300 else "")
                self._out(f"       {detail}")
                if ann.attachment and ann.attachment not in ("-", ""):
                    self._out(f"       >> {ann.attachment}")
            self._out("")

    @staticmethod
    def _out(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", errors="replace").decode("ascii"))


# ──────────────────────────────────────────────────────────────────────────────
# CsvExporter  —  saves results to disk
# ──────────────────────────────────────────────────────────────────────────────

class CsvExporter:
    """Serialises ScoredAnnouncement results to a CSV file."""

    _FIELDS = ["score", "tags", "symbol", "company", "subject",
               "datetime", "details", "attachment"]

    def export(self, items: list[ScoredAnnouncement], out_path: str | Path) -> None:
        out = Path(out_path)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._FIELDS)
            writer.writeheader()
            for sa in items:
                ann = sa.announcement
                writer.writerow({
                    "score":      sa.score,
                    "tags":       sa.result.tag_str,
                    "symbol":     ann.symbol,
                    "company":    ann.company,
                    "subject":    ann.subject,
                    "datetime":   sa.dt_str,
                    "details":    ann.details,
                    "attachment": ann.attachment,
                })
        print(f"Saved {len(items)} rows -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
# ScreenerApp  —  CLI entry point, wires all components
# ──────────────────────────────────────────────────────────────────────────────

class ScreenerApp:
    """
    Parses CLI arguments and wires the component graph:

        CsvParser
            |
        AnnouncementFilter  (uses FilterConfig)
            |
        ScoringEngine       (uses FilterConfig)
            |
        EPScreener
            |
        ReportPrinter  /  CsvExporter
    """

    def run(self, argv: list[str] | None = None) -> None:
        args = self._parse_args(argv)

        config  = FilterConfig()
        parser  = CsvParser(args.csv_file)
        fltr    = AnnouncementFilter(config)
        scorer  = ScoringEngine(config)
        screener = EPScreener(parser, fltr, scorer)

        try:
            results = screener.run(min_score=args.min_score)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        ReportPrinter().print(results, verbose=args.verbose)

        if args.csv_out:
            CsvExporter().export(results, args.csv_out)

    @staticmethod
    def _parse_args(argv: list[str] | None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog="ep_screener",
            description=(
                "EP Screener — filters BSE/NSE announcements for "
                "Stockbee-style episodic pivot candidates."
            ),
        )
        parser.add_argument("csv_file", help="Path to BSE/NSE announcements CSV")
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
        return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ScreenerApp().run()
