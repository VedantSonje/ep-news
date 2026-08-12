"""
CSV parsing layer.

ColumnMapper  — maps actual BSE/NSE header names to canonical field names.
CsvParser     — reads a CSV file and yields Announcement objects.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator

from models import Announcement


class ColumnMapper:
    """
    Detects and maps exchange-specific CSV column headers to canonical
    field names so the rest of the pipeline doesn't care about BSE vs NSE.
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
        if "subject" not in self._map:
            raise ValueError(
                "Cannot locate a 'subject' column in the CSV.\n"
                f"Mapped fields found: {list(self._map.keys())}"
            )


class CsvParser:
    """
    Reads a BSE/NSE announcements CSV and yields Announcement objects.
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

    def parse(self) -> Iterator[Announcement]:
        """Yield one Announcement per data row."""
        with open(self.path, encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            col = ColumnMapper(list(reader.fieldnames or []))
            col.validate()
            for row in reader:
                yield self._build(row, col)

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
