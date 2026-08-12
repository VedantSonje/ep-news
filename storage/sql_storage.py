"""
SQLiteStorage — structured storage for scored announcements.

Recommended production upgrade: PostgreSQL via psycopg2.
The SQL here is ANSI-compatible; only the connection string changes.

Schema
------
announcements(id, symbol, company, subject, details, score, tags,
              broadcast_dt, attachment, source_file, ingested_at,
              sector_tags, is_novel, relative_size, catalyst_type,
              government_linked, export_component, order_value_cr)

Unique key: (symbol, broadcast_dt, subject, SUBSTR(details,1,80))
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from models import ScoredAnnouncement
from storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """
    SQLite-backed structured storage.

    All public query methods return plain list[dict] so callers
    (especially the AI agent tools) don't need ORM knowledge.
    """

    # Core DDL — only columns that have existed since v1 (no enrichment cols here,
    # those are added via _migrate_enrichment_columns to support older DBs).
    _DDL = """
    CREATE TABLE IF NOT EXISTS announcements (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol           TEXT    NOT NULL,
        company          TEXT    NOT NULL,
        subject          TEXT    NOT NULL,
        details          TEXT,
        score            INTEGER NOT NULL,
        tags             TEXT,
        broadcast_dt     TEXT,
        attachment       TEXT,
        source_file      TEXT,
        ingested_at      TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_uniq
        ON announcements(symbol, broadcast_dt, subject, SUBSTR(details,1,80));
    CREATE INDEX IF NOT EXISTS idx_score
        ON announcements(score DESC);
    CREATE INDEX IF NOT EXISTS idx_symbol
        ON announcements(symbol);
    CREATE INDEX IF NOT EXISTS idx_subject
        ON announcements(subject);
    CREATE INDEX IF NOT EXISTS idx_broadcast
        ON announcements(broadcast_dt);
    """

    # Columns added after initial schema — applied via ALTER TABLE if missing
    _ENRICHMENT_COLUMNS: list[tuple[str, str]] = [
        ("sector_tags",       "TEXT"),
        ("is_novel",          "INTEGER"),
        ("relative_size",     "TEXT"),
        ("catalyst_type",     "TEXT"),
        ("government_linked", "INTEGER"),
        ("export_component",  "INTEGER"),
        ("order_value_cr",    "REAL"),
    ]

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._DDL)
        self._migrate_enrichment_columns()
        self._conn.commit()

    def _migrate_enrichment_columns(self) -> None:
        """Add enrichment columns to an existing DB that pre-dates them."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(announcements)").fetchall()
        }
        added_sector = False
        for col_name, col_type in self._ENRICHMENT_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE announcements ADD COLUMN {col_name} {col_type}"
                )
                if col_name == "sector_tags":
                    added_sector = True

        # Create the sector index only after the column is guaranteed to exist
        if "sector_tags" in existing or added_sector:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sector ON announcements(sector_tags)"
            )

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose raw connection for FTS5 setup and novelty detector."""
        return self._conn

    # ── BaseStorage interface ──────────────────────────────────────────────

    def save(self, items: list[ScoredAnnouncement], source_file: str = "") -> int:
        """Insert scored announcements; silently skip duplicates."""
        sql = """
        INSERT OR IGNORE INTO announcements
            (symbol, company, subject, details, score, tags,
             broadcast_dt, attachment, source_file,
             sector_tags, is_novel, relative_size, catalyst_type,
             government_linked, export_component, order_value_cr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                sa.announcement.symbol,
                sa.announcement.company,
                sa.announcement.subject,
                sa.announcement.details,
                sa.score,
                sa.result.tag_str,
                sa.dt_str,
                sa.announcement.attachment,
                source_file,
                getattr(sa, "sector_tags", None),
                int(getattr(sa, "is_novel", True)),
                getattr(sa, "relative_size", None),
                getattr(sa, "catalyst_type", None),
                int(getattr(sa, "government_linked", False)),
                int(getattr(sa, "export_component", False)),
                getattr(sa, "order_value_cr", None),
            )
            for sa in items
        ]
        cur = self._conn.executemany(sql, rows)
        self._conn.commit()
        return cur.rowcount

    def update_enrichment(self, row_id: int, **fields) -> None:
        """Update enrichment columns for a single row (used by enrich command)."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [row_id]
        self._conn.execute(
            f"UPDATE announcements SET {set_clause} WHERE id = ?", params
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM announcements").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    # ── Query helpers (used by agent tools) ───────────────────────────────

    def query(self, sql: str) -> list[dict[str, Any]]:
        """
        Execute any SELECT query and return rows as plain dicts.
        Only SELECT is permitted — DML is blocked for safety.
        """
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")
        cur = self._conn.execute(sql)
        return [dict(row) for row in cur.fetchall()]

    def get_top_by_score(
        self, min_score: int = 8, limit: int = 10
    ) -> list[dict[str, Any]]:
        return self.query(
            f"SELECT symbol, company, subject, score, tags, broadcast_dt, attachment "
            f"FROM announcements WHERE score >= {min_score} "
            f"ORDER BY score DESC, broadcast_dt DESC LIMIT {limit}"
        )

    def get_by_symbol(self, symbol: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT symbol, company, subject, score, tags, broadcast_dt, attachment "
            "FROM announcements WHERE symbol = ? ORDER BY broadcast_dt DESC",
            (symbol.upper(),),
        )
        return [dict(row) for row in cur.fetchall()]

    def stats(self) -> dict[str, Any]:
        total = self.count()
        top = self._conn.execute(
            "SELECT symbol, MAX(score) as score FROM announcements GROUP BY symbol "
            "ORDER BY score DESC LIMIT 5"
        ).fetchall()
        return {
            "total_announcements": total,
            "top_symbols": [dict(r) for r in top],
        }
