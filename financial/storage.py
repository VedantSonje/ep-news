"""
FinancialStorage — persists extracted financial results in two backends:

  SQL  : financial_results table (added to the existing ep_news.db)
  Chroma: ep_financials collection (for semantic search over financial narratives)

The SQL schema is deliberately flat / analytical (no normalisation) so the
AI agent can query it with simple SQL without any joins.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from financial.models import FinancialResult
from financial.vector_store import VectorStore


# ── SQL DDL ───────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS financial_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    company             TEXT    NOT NULL,
    period              TEXT,
    period_label        TEXT,
    period_type         TEXT,
    revenue_cr          REAL,
    revenue_growth_pct  REAL,
    ebitda_cr           REAL,
    ebitda_margin_pct   REAL,
    pat_cr              REAL,
    pat_growth_pct      REAL,
    eps                 REAL,
    dividend_per_share  REAL,
    key_highlights      TEXT,
    guidance            TEXT,
    raw_summary         TEXT,
    source_url          TEXT,
    broadcast_dt        TEXT,
    extracted_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    sector              TEXT,
    client_name         TEXT,
    client_type         TEXT,
    order_sector        TEXT,
    execution_months    INTEGER,
    confidence          TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fin_sym_period
    ON financial_results(symbol, period);
CREATE INDEX IF NOT EXISTS idx_fin_symbol      ON financial_results(symbol);
CREATE INDEX IF NOT EXISTS idx_fin_pat         ON financial_results(pat_cr DESC);
CREATE INDEX IF NOT EXISTS idx_fin_revenue     ON financial_results(revenue_cr DESC);
CREATE INDEX IF NOT EXISTS idx_fin_sector      ON financial_results(sector);
CREATE INDEX IF NOT EXISTS idx_fin_client_type ON financial_results(client_type);
CREATE INDEX IF NOT EXISTS idx_fin_order_sector ON financial_results(order_sector);
"""

_ORDER_WINS_DDL = """
CREATE TABLE IF NOT EXISTS order_wins (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    company             TEXT NOT NULL,
    broadcast_dt        TEXT NOT NULL,
    order_value_cr      REAL,
    client_name         TEXT,
    client_type         TEXT,
    order_sector        TEXT,
    execution_months    INTEGER,
    description         TEXT,
    raw_summary         TEXT,
    source_url          TEXT UNIQUE,
    from_genuine_order  INTEGER DEFAULT 0,
    extracted_at        TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ow_symbol     ON order_wins(symbol);
CREATE INDEX IF NOT EXISTS idx_ow_date       ON order_wins(broadcast_dt);
CREATE INDEX IF NOT EXISTS idx_ow_value      ON order_wins(order_value_cr DESC);
CREATE INDEX IF NOT EXISTS idx_ow_client_type ON order_wins(client_type);
CREATE INDEX IF NOT EXISTS idx_ow_sector     ON order_wins(order_sector);
"""

_ORDER_SUBJECTS = frozenset({
    "Bagging/Receiving of orders/contracts",
    "Awarding of order(s)/contract(s)",
})

# ALTER TABLE statements to add new columns to existing DBs (idempotent via try/except)
_MIGRATIONS = [
    "ALTER TABLE financial_results ADD COLUMN period_label     TEXT",
    "ALTER TABLE financial_results ADD COLUMN sector           TEXT",
    "ALTER TABLE financial_results ADD COLUMN client_name      TEXT",
    "ALTER TABLE financial_results ADD COLUMN client_type      TEXT",
    "ALTER TABLE financial_results ADD COLUMN order_sector     TEXT",
    "ALTER TABLE financial_results ADD COLUMN execution_months INTEGER",
    "ALTER TABLE financial_results ADD COLUMN confidence       TEXT",
]


class FinancialStorage:
    """
    Dual-backend storage for FinancialResult objects.
    Shares the same SQLite file as SQLiteStorage but manages its own
    connection and table (financial_results).
    """

    def __init__(self, db_path: Path | str, chroma_path: Path | str) -> None:
        # ── SQL ──────────────────────────────────────────────────────────
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.executescript(_ORDER_WINS_DDL)
        # Apply migrations for existing DBs (columns added after initial schema)
        for stmt in _MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

        # ── ChromaDB ─────────────────────────────────────────────────────
        try:
            self._vector = VectorStore(chroma_path)
            self._chroma_ok = True
        except Exception as e:
            print(f"  [chroma] init failed: {e}")
            self._chroma_ok = False
            self._vector = None

    # ── Write ──────────────────────────────────────────────────────────────

    def save(self, result: FinancialResult) -> bool:
        """
        Insert FinancialResult into SQL and ChromaDB.
        Order wins are routed to the order_wins table; all others go to financial_results.
        Returns True if newly inserted, False if already existed (unique key clash).
        """
        if result.period_type == "order_win":
            sql_new = self._sql_save_order_win(result)
        else:
            sql_new = self._sql_save(result)
        if self._chroma_ok:
            self._chroma_save(result)
        return sql_new

    def _sql_save_order_win(self, r: FinancialResult) -> bool:
        """Insert into order_wins table. Resolves from_genuine_order via announcements join."""
        from_genuine = self._lookup_genuine(r.source_url)
        # Parse description from key_highlights if not set directly
        description = None
        for hl in (r.key_highlights or []):
            if isinstance(hl, str) and hl.lower().startswith("description:"):
                description = hl.split(":", 1)[1].strip()
                break
        cur = self._conn.execute("""
            INSERT OR IGNORE INTO order_wins
                (symbol, company, broadcast_dt, order_value_cr,
                 client_name, client_type, order_sector, execution_months,
                 description, raw_summary, source_url, from_genuine_order)
            VALUES (?,?,?,?,  ?,?,?,?,  ?,?,?,?)
        """, (
            r.symbol, r.company, r.broadcast_dt,
            r.order_value_cr,
            r.client_name, r.client_type, r.order_sector, r.execution_months,
            description, r.raw_summary, r.source_url, from_genuine,
        ))
        self._conn.commit()
        return cur.rowcount > 0

    def _lookup_genuine(self, source_url: str | None) -> int:
        """Return 1 if the source announcement subject is a genuine order subject."""
        if not source_url:
            return 0
        row = self._conn.execute(
            "SELECT subject FROM announcements WHERE attachment = ? LIMIT 1",
            (source_url,),
        ).fetchone()
        if row and row[0] in _ORDER_SUBJECTS:
            return 1
        return 0

    def already_extracted(self, symbol: str, period: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM financial_results WHERE symbol=? AND period=?",
            (symbol, period),
        ).fetchone()
        return row is not None

    # ── SQL helpers ────────────────────────────────────────────────────────

    def _sql_save(self, r: FinancialResult) -> bool:
        sql = """
        INSERT OR IGNORE INTO financial_results
            (symbol, company, period, period_label, period_type,
             revenue_cr, revenue_growth_pct,
             ebitda_cr, ebitda_margin_pct,
             pat_cr, pat_growth_pct,
             eps, dividend_per_share,
             key_highlights, guidance, raw_summary,
             source_url, broadcast_dt,
             sector, client_name, client_type, order_sector, execution_months,
             confidence)
        VALUES (?,?,?,?,?,  ?,?,  ?,?,  ?,?,  ?,?,  ?,?,?,  ?,?,  ?,?,?,?,?,  ?)
        """
        cur = self._conn.execute(sql, (
            r.symbol, r.company, r.period, r.period_label, r.period_type,
            r.revenue_cr, r.revenue_growth_pct,
            r.ebitda_cr, r.ebitda_margin_pct,
            r.pat_cr, r.pat_growth_pct,
            r.eps, r.dividend_per_share,
            r.highlights_json(), r.guidance, r.raw_summary,
            r.source_url, r.broadcast_dt,
            r.sector, r.client_name, r.client_type, r.order_sector, r.execution_months,
            r.confidence,
        ))
        self._conn.commit()
        return cur.rowcount > 0

    def _sql_update_by_url(self, r: FinancialResult) -> bool:
        """Update existing record matched by source_url, or INSERT if not found."""
        existing = self._conn.execute(
            "SELECT id FROM financial_results WHERE source_url=?",
            (r.source_url,),
        ).fetchone()
        if existing:
            try:
                self._conn.execute("""
                    UPDATE financial_results SET
                        period=?, period_label=?, period_type=?,
                        revenue_cr=?, revenue_growth_pct=?,
                        ebitda_cr=?, ebitda_margin_pct=?,
                        pat_cr=?, pat_growth_pct=?,
                        eps=?, dividend_per_share=?,
                        key_highlights=?, guidance=?, raw_summary=?,
                        sector=?, client_name=?, client_type=?, order_sector=?, execution_months=?,
                        confidence=?,
                        extracted_at=CURRENT_TIMESTAMP
                    WHERE source_url=?
                """, (
                    r.period, r.period_label, r.period_type,
                    r.revenue_cr, r.revenue_growth_pct,
                    r.ebitda_cr, r.ebitda_margin_pct,
                    r.pat_cr, r.pat_growth_pct,
                    r.eps, r.dividend_per_share,
                    r.highlights_json(), r.guidance, r.raw_summary,
                    r.sector, r.client_name, r.client_type, r.order_sector, r.execution_months,
                    r.confidence,
                    r.source_url,
                ))
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Another row already has (symbol, period) — delete old and re-insert
                self._conn.execute(
                    "DELETE FROM financial_results WHERE source_url=?", (r.source_url,)
                )
                self._conn.commit()
                return self._sql_save(r)
            return True
        return self._sql_save(r)

    def save_or_update(self, result: FinancialResult) -> bool:
        """Upsert by source_url — used during re-extraction to overwrite stale data."""
        if result.period_type == "order_win":
            sql_updated = self._sql_upsert_order_win(result)
        else:
            sql_updated = self._sql_update_by_url(result)
        if self._chroma_ok:
            self._chroma_save(result)
        return sql_updated

    def _sql_upsert_order_win(self, r: FinancialResult) -> bool:
        """Update order_wins by source_url, or insert if not found."""
        existing = self._conn.execute(
            "SELECT id FROM order_wins WHERE source_url=?", (r.source_url,)
        ).fetchone()
        if existing:
            from_genuine = self._lookup_genuine(r.source_url)
            description = None
            for hl in (r.key_highlights or []):
                if isinstance(hl, str) and hl.lower().startswith("description:"):
                    description = hl.split(":", 1)[1].strip()
                    break
            self._conn.execute("""
                UPDATE order_wins SET
                    order_value_cr=?, client_name=?, client_type=?,
                    order_sector=?, execution_months=?, description=?,
                    raw_summary=?, from_genuine_order=?,
                    extracted_at=CURRENT_TIMESTAMP
                WHERE source_url=?
            """, (
                r.order_value_cr, r.client_name, r.client_type,
                r.order_sector, r.execution_months, description,
                r.raw_summary, from_genuine, r.source_url,
            ))
            self._conn.commit()
            return True
        return self._sql_save_order_win(r)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SELECT against financial_results and return plain dicts."""
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")
        cur = self._conn.execute(sql)
        return [dict(row) for row in cur.fetchall()]

    def get_by_symbol(self, symbol: str) -> list[dict[str, Any]]:
        return self.query(
            f"SELECT * FROM financial_results WHERE symbol='{symbol.upper()}' "
            f"ORDER BY extracted_at DESC"
        )

    def get_top_by_pat(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.query(
            f"SELECT symbol, company, period, pat_cr, pat_growth_pct, "
            f"revenue_cr, ebitda_margin_pct, eps "
            f"FROM financial_results WHERE pat_cr IS NOT NULL "
            f"ORDER BY pat_cr DESC LIMIT {limit}"
        )

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM financial_results"
        ).fetchone()
        return row[0] if row else 0

    # ── ChromaDB helpers ───────────────────────────────────────────────────

    def _chroma_save(self, r: FinancialResult) -> None:
        self._vector.save_financial(r)

    def search_financials(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Semantic search over financial result narratives."""
        if not self._chroma_ok:
            return []
        return self._vector.search_financials(query, n=n_results)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FinancialStorage:
        return self

    def __exit__(self, *_) -> None:
        self.close()
