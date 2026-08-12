"""
BM25Search — wraps SQLite FTS5 virtual table for keyword retrieval.

FTS5 uses BM25 by default (bm25() ranking function built-in).
setup_fts() creates the virtual table + a trigger to keep it in sync.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS announcements_fts
USING fts5(
    symbol, company, subject, details,
    content='announcements',
    content_rowid='id',
    tokenize='porter ascii'
)
"""

_FTS_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS announcements_ai
       AFTER INSERT ON announcements BEGIN
           INSERT INTO announcements_fts(rowid, symbol, company, subject, details)
           VALUES (new.id, new.symbol, new.company, new.subject, new.details);
       END""",
    """CREATE TRIGGER IF NOT EXISTS announcements_ad
       AFTER DELETE ON announcements BEGIN
           INSERT INTO announcements_fts(announcements_fts, rowid, symbol, company, subject, details)
           VALUES ('delete', old.id, old.symbol, old.company, old.subject, old.details);
       END""",
    """CREATE TRIGGER IF NOT EXISTS announcements_au
       AFTER UPDATE ON announcements BEGIN
           INSERT INTO announcements_fts(announcements_fts, rowid, symbol, company, subject, details)
           VALUES ('delete', old.id, old.symbol, old.company, old.subject, old.details);
           INSERT INTO announcements_fts(rowid, symbol, company, subject, details)
           VALUES (new.id, new.symbol, new.company, new.subject, new.details);
       END""",
]


@dataclass
class BM25Hit:
    id:           int
    symbol:       str
    company:      str
    subject:      str
    details:      str
    score:        int
    broadcast_dt: str
    bm25_rank:    float   # negative — more negative = better match


def setup_fts(conn: sqlite3.Connection) -> None:
    """
    Create the FTS5 virtual table and sync triggers if they don't exist.
    Safe to call on every startup — all DDL uses IF NOT EXISTS.
    Uses individual execute() calls (not executescript) to preserve
    the caller's transaction state.
    """
    try:
        conn.execute(_FTS_TABLE)
        for trigger_sql in _FTS_TRIGGERS:
            conn.execute(trigger_sql)
        conn.commit()
    except sqlite3.OperationalError:
        # FTS5 not available in this SQLite build — BM25 degrades gracefully
        return

    # Rebuild FTS index for rows ingested before triggers existed
    try:
        count = conn.execute("SELECT COUNT(*) FROM announcements_fts").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
        if count < total:
            conn.execute("INSERT INTO announcements_fts(announcements_fts) VALUES ('rebuild')")
            conn.commit()
    except sqlite3.Error:
        pass


class BM25Search:
    """
    Full-text search over announcements using SQLite FTS5 / BM25.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def search(
        self,
        query: str,
        limit: int = 20,
        min_score: int | None = None,
        date_filter: str | None = None,
        symbols: list[str] | None = None,
        themes: list[str] | None = None,
    ) -> list[BM25Hit]:
        """
        Run FTS5 BM25 search. Returns hits sorted best-first.
        Applies optional metadata filters via a JOIN on announcements.
        When query is empty but themes/filters exist, falls back to pure SQL.
        """
        # Build WHERE clause for the JOIN
        where_parts: list[str] = []
        join_params: list = []

        if min_score is not None:
            where_parts.append("a.score >= ?")
            join_params.append(min_score)
        if date_filter:
            where_parts.append("DATE(a.broadcast_dt) = ?")
            join_params.append(date_filter)
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            where_parts.append(f"a.symbol IN ({placeholders})")
            join_params.extend(symbols)
        if themes:
            for theme in themes:
                where_parts.append("a.sector_tags LIKE ?")
                join_params.append(f"%{theme}%")

        where_clause = ("AND " + " AND ".join(where_parts)) if where_parts else ""

        query_stripped = query.strip()
        if not query_stripped:
            # No text query — fall back to pure SQL
            w = ("WHERE " + " AND ".join(where_parts)) if where_parts else "WHERE 1=1"
            sql = f"""
                SELECT id, symbol, company, subject, details, score, broadcast_dt,
                       CAST(score AS REAL) / 13.0 AS rank
                FROM announcements {w}
                ORDER BY score DESC, broadcast_dt DESC LIMIT {limit}
            """
            try:
                rows = self._conn.execute(sql, join_params).fetchall()
            except sqlite3.OperationalError:
                return []
            return [
                BM25Hit(id=r[0], symbol=r[1], company=r[2], subject=r[3],
                        details=r[4], score=r[5], broadcast_dt=r[6], bm25_rank=-r[7])
                for r in rows
            ]

        params: list = [query_stripped] + join_params + [limit]
        sql = f"""
            SELECT
                a.id, a.symbol, a.company, a.subject, a.details,
                a.score, a.broadcast_dt,
                bm25(announcements_fts) AS rank
            FROM announcements_fts
            JOIN announcements a ON announcements_fts.rowid = a.id
            WHERE announcements_fts MATCH ?
            {where_clause}
            ORDER BY rank
            LIMIT ?
        """

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS table not set up yet
            return []

        return [
            BM25Hit(
                id=r[0], symbol=r[1], company=r[2],
                subject=r[3], details=r[4], score=r[5],
                broadcast_dt=r[6], bm25_rank=r[7],
            )
            for r in rows
        ]
