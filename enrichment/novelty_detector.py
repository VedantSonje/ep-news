"""
NoveltyDetector — identifies whether an announcement is genuinely new
or a repeat / update of something already disclosed.

Heuristics (all deterministic, no LLM):
  1. Subject-level repeats: same symbol + same subject within N days
  2. Content similarity: >60% word overlap with a recent announcement
  3. Keyword signals: subject/details contain update/revision/corrigendum
  4. Filing-type signals: "Revised", "Outcome" re-filed same day

A low novelty score does NOT mean the stock is uninteresting — it means
the catalyst is not new information.  The agent uses this to avoid
double-counting the same event.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class NoveltyResult:
    is_novel: bool
    confidence: float      # 0.0 (definitely repeat) – 1.0 (definitely new)
    reason: str            # human-readable explanation


# Signals that suggest this is a follow-up / revision
_REPEAT_SIGNALS = re.compile(
    r"\b(update|revised|revision|corrigendum|re-filed|"
    r"reminder|addendum|amendment|erratum|clarification|"
    r"reclassif|re-submission|follow.?up)\b",
    re.IGNORECASE,
)

# Subjects that are inherently recurring (not novel events)
_RECURRING_SUBJECTS = frozenset({
    "Trading Window",
    "Record Date",
    "Shareholders meeting",
    "Copy of Newspaper Publication",
    "Monthly Business Updates",
    "Analysts/Institutional Investor Meet/Con. Call Updates",
})


def _word_set(text: str) -> set[str]:
    """Tokenise text into lowercase words for overlap calculation."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NoveltyDetector:
    """
    Checks the announcements table for recent similar filings.
    Requires access to the SQLite connection.
    """

    def __init__(self, lookback_days: int = 30) -> None:
        self._lookback = lookback_days

    def detect(
        self,
        symbol: str,
        subject: str,
        details: str,
        broadcast_dt: str,
        conn: sqlite3.Connection,
    ) -> NoveltyResult:
        """
        Returns NoveltyResult for the given announcement.
        Queries the announcements table for recent filings from the same symbol.
        """

        # ── 1. Recurring subject → never novel ────────────────────────────
        if subject in _RECURRING_SUBJECTS:
            return NoveltyResult(False, 0.1, "recurring subject type")

        # ── 2. Keyword signals in the text ────────────────────────────────
        if _REPEAT_SIGNALS.search(f"{subject} {details}"):
            return NoveltyResult(False, 0.2, "revision/update keyword detected")

        # ── 3. Check DB for recent same-subject filings ───────────────────
        try:
            dt = datetime.fromisoformat(broadcast_dt[:16])
        except ValueError:
            dt = datetime.now()

        cutoff = (dt - timedelta(days=self._lookback)).strftime("%Y-%m-%d")

        rows = conn.execute(
            "SELECT subject, details FROM announcements "
            "WHERE symbol=? AND broadcast_dt >= ? AND subject=?",
            (symbol, cutoff, subject),
        ).fetchall()

        # Same subject in lookback window?
        if rows:
            new_words = _word_set(details)
            for row in rows:
                old_words = _word_set(row[1] or "")
                sim = _jaccard(new_words, old_words)
                if sim > 0.60:
                    return NoveltyResult(
                        False, round(1 - sim, 2),
                        f"content overlap {sim:.0%} with recent {subject}"
                    )
            # Same subject but different content — moderately novel
            return NoveltyResult(True, 0.65,
                                 f"same subject seen {len(rows)}x but different content")

        # ── 4. Completely new subject for this symbol ─────────────────────
        return NoveltyResult(True, 0.95, "first occurrence in lookback window")
