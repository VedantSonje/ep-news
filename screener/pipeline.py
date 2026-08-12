"""
EPScreener — pipeline orchestrator.

Wires CsvParser -> AnnouncementFilter -> Enrichment -> store.
No scoring gate: every announcement whose subject is in the allowlist is stored.
Enrichment (sector_tags, novelty, materiality) runs inline when conn is provided.
"""
from __future__ import annotations

import sqlite3

from models import ScoredAnnouncement, ScoreResult
from screener.csv_parser import CsvParser
from screener.scoring_engine import AnnouncementFilter
from enrichment.sector_tagger import SectorTagger
from enrichment.novelty_detector import NoveltyDetector
from enrichment.materiality import MaterialityExtractor


class EPScreener:
    """
    Runs the full screening pipeline for a single CSV file.
    Returns results sorted by broadcast time descending.
    """

    def __init__(
        self,
        parser:     CsvParser,
        ann_filter: AnnouncementFilter,
    ) -> None:
        self._parser   = parser
        self._filter   = ann_filter
        self._tagger   = SectorTagger()
        self._novelty  = NoveltyDetector(lookback_days=30)
        self._material = MaterialityExtractor()

    def run(
        self,
        conn: sqlite3.Connection | None = None,
    ) -> list[ScoredAnnouncement]:
        results: list[ScoredAnnouncement] = []

        for ann in self._parser.parse():
            if not self._filter.should_keep(ann):
                continue

            # Score is 0 — subject filter is the only gate
            sa = ScoredAnnouncement(ann, ScoreResult(0, [ann.subject]))

            # ── Enrichment ──────────────────────────────────────────────
            sector_tags = self._tagger.tag_str(ann.subject, ann.details, ann.company)
            mat         = self._material.extract(ann.subject, ann.details, ann.company)

            sa.sector_tags       = sector_tags
            sa.relative_size     = mat.relative_size
            sa.catalyst_type     = mat.catalyst_type
            sa.government_linked = mat.government_linked
            sa.export_component  = mat.export_component
            sa.order_value_cr    = mat.order_value_cr

            if conn is not None:
                nov = self._novelty.detect(
                    ann.symbol, ann.subject, ann.details, ann.broadcast_raw, conn
                )
                sa.is_novel = nov.is_novel
            else:
                sa.is_novel = True

            results.append(sa)

        results.sort(key=lambda sa: sa.announcement.broadcast_raw, reverse=True)
        return results
