"""
DatabaseManager — coordinates SQLiteStorage and ChromaDBStorage.

Single point of contact for the agent and ingest pipeline.
Both databases are always written together to stay in sync.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from models import ScoredAnnouncement
from storage.sql_storage import SQLiteStorage
from storage.vector_storage import ChromaDBStorage


class DatabaseManager:
    """
    Facade over both storage backends.
    Callers never interact with SQLiteStorage or ChromaDBStorage directly.
    """

    def __init__(self, db_path: Path | str, chroma_path: Path | str) -> None:
        self.sql = SQLiteStorage(db_path)
        self.vector = ChromaDBStorage(chroma_path)

    # ── Write ──────────────────────────────────────────────────────────────

    def save_all(
        self,
        items: list[ScoredAnnouncement],
        source_file: str = "",
        refresh_vectors: bool = True,
    ) -> dict[str, int]:
        """
        Persist items in both backends.

        refresh_vectors=True (default): before upserting to ChromaDB, purge
        any existing entries for each (symbol, date) pair in the batch.
        This ensures updated announcements replace stale embeddings rather
        than accumulating alongside them.
        """
        sql_saved = self.sql.save(items, source_file)

        if refresh_vectors:
            # Collect (symbol, date_str) pairs present in this batch
            symbol_dates = {
                (sa.announcement.symbol, sa.dt_str[:10])
                for sa in items
                if sa.dt_str
            }
            purged = 0
            for symbol, date_str in symbol_dates:
                purged += self.vector.purge_stale_for_date(symbol, date_str)

        vec_saved = self.vector.save(items, source_file)
        return {"sql": sql_saved, "vector": vec_saved}

    # ── Read (agent tools delegate here) ──────────────────────────────────

    def sql_query(self, sql: str) -> list[dict[str, Any]]:
        return self.sql.query(sql)

    def semantic_search(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        return self.vector.search(query, n_results)

    def get_top_ep_candidates(
        self, min_score: int = 8, limit: int = 10
    ) -> list[dict[str, Any]]:
        return self.sql.get_top_by_score(min_score, limit)

    def get_company_announcements(self, symbol: str) -> list[dict[str, Any]]:
        return self.sql.get_by_symbol(symbol)

    def stats(self) -> dict[str, Any]:
        return {
            "sql_total": self.sql.count(),
            "vector_total": self.vector.count(),
            **self.sql.stats(),
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        self.sql.close()
        self.vector.close()

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, *_) -> None:
        self.close()
