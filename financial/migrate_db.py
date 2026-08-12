"""
One-time migration + backfill for the ep_news.db improvements.

Runs safely on an existing DB:
  1. ALTER TABLE to add new columns (idempotent)
  2. CREATE companies master table
  3. Backfill financial_results:
       - period_label  : normalized "Q1 FY2027" from raw period/broadcast_dt
       - sector        : copied from announcements.sector_tags for quarterly/annual
       - client_name   : parsed from key_highlights for order_win rows
       - client_type   : parsed from key_highlights for order_win rows
       - order_sector  : parsed from key_highlights for order_win rows
       - execution_months : parsed from key_highlights for order_win rows
       - revenue_growth_pct / pat_growth_pct : computed from financial_statements
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ep_news.db"


# ── Migrations ────────────────────────────────────────────────────────────────

_ALTER_STMTS = [
    "ALTER TABLE financial_results ADD COLUMN period_label     TEXT",
    "ALTER TABLE financial_results ADD COLUMN sector           TEXT",
    "ALTER TABLE financial_results ADD COLUMN client_name      TEXT",
    "ALTER TABLE financial_results ADD COLUMN client_type      TEXT",
    "ALTER TABLE financial_results ADD COLUMN order_sector     TEXT",
    "ALTER TABLE financial_results ADD COLUMN execution_months INTEGER",
]

_COMPANIES_DDL = """
CREATE TABLE IF NOT EXISTS companies (
    symbol        TEXT PRIMARY KEY,
    company       TEXT NOT NULL,
    sector        TEXT,
    market_cap_cr REAL,
    index_member  TEXT,
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);
"""


# ── Period normalisation ──────────────────────────────────────────────────────

_Q_LABEL_RE = re.compile(
    r'\b(Q[1-4]\s*FY[\s\']*\d{2,4})\b'          # "Q1 FY2027" / "Q1 FY'27"
    r'|(FY\s*\d{4})\b'                            # "FY2026"
    r'|(H[12]\s*FY[\s\']*\d{2,4})\b',             # "H1 FY2027"
    re.IGNORECASE,
)

_CONCALL_RE = re.compile(r'\s*concall\s*$', re.I)


def _date_to_period_label(dt_str: str) -> str:
    """Convert broadcast_dt like '2026-08-04 10:00' to 'Q2 FY2027'."""
    try:
        date_part = dt_str[:10]
        year, month, _ = date_part.split("-")
        year, month = int(year), int(month)
    except Exception:
        return ""
    # Indian fiscal year: Apr=Q1, Jul=Q2, Oct=Q3, Jan=Q4
    if month >= 4:
        fy = year + 1
        q = (month - 4) // 3 + 1
    else:
        fy = year
        q = (month + 8) // 3
    return f"Q{q} FY{fy}"


def _normalize_period(period: str, broadcast_dt: str) -> str:
    """Return a clean label like 'Q1 FY2027' or 'FY2026' from the raw period field."""
    if not period:
        return _date_to_period_label(broadcast_dt)
    # Strip trailing " concall"
    p = _CONCALL_RE.sub("", period).strip()
    m = _Q_LABEL_RE.search(p)
    if m:
        label = next(g for g in m.groups() if g)
        # Normalise spacing: "Q1FY2027" → "Q1 FY2027"
        label = re.sub(r'(Q[1-4])\s*(FY)', r'\1 \2', label, flags=re.I)
        label = re.sub(r"FY\s*'?(\d{2})$", lambda x: f"FY20{x.group(1)}", label)
        return label.strip()
    # Raw date string (e.g. "2026-08-04") → derive from broadcast_dt
    if re.match(r'^\d{4}-\d{2}-\d{2}$', p):
        return _date_to_period_label(broadcast_dt)
    return p  # leave as-is if unrecognised


# ── Order win field parsers ───────────────────────────────────────────────────

_EXEC_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(month|year|week|day)', re.IGNORECASE
)


def _parse_hl(highlights_json: str) -> dict[str, str]:
    """Return a flat dict from key_highlights list: {'Client': '...', 'Sector': '...', ...}"""
    try:
        items = json.loads(highlights_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return {}
    result: dict[str, str] = {}
    for item in items:
        if isinstance(item, str) and ":" in item:
            key, _, val = item.partition(":")
            result[key.strip()] = val.strip()
    return result


def _execution_months(text: str) -> int | None:
    m = _EXEC_RE.search(text)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    if "year" in unit:
        return round(val * 12)
    if "week" in unit:
        return max(1, round(val / 4))
    if "day" in unit:
        return max(1, round(val / 30))
    return round(val)  # months


# ── YoY growth from financial_statements ─────────────────────────────────────

def _compute_growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 1)


def _get_statement_values(conn: sqlite3.Connection, symbol: str, broadcast_dt: str) -> dict:
    """
    Return {revenue_col1, revenue_col3, pat_col1, pat_col3} from financial_statements
    where col1 = current period, col3 = same period last year.
    """
    row = conn.execute("""
        SELECT line_items FROM financial_statements
        WHERE symbol = ? AND DATE(broadcast_dt) = DATE(?)
        LIMIT 1
    """, (symbol, broadcast_dt)).fetchone()
    if not row:
        return {}
    try:
        items: dict = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return {}

    # Keywords for revenue and PAT
    rev_kws  = ["revenue from operations", "total revenue from operations",
                "net sales", "net revenue", "total income from operations"]
    pat_kws  = ["profit for the period", "profit for the year",
                "profit/(loss) for the period", "profit/(loss) for the year",
                "net profit", "profit after tax"]

    def _find_vals(kws: list[str]) -> list | None:
        for name, vals in items.items():
            name_l = name.lower()
            if any(kw in name_l for kw in kws):
                if isinstance(vals, list) and len(vals) >= 3:
                    return vals
        return None

    rev_vals = _find_vals(rev_kws)
    pat_vals = _find_vals(pat_kws)

    return {
        "revenue_col1": rev_vals[0] if rev_vals else None,
        "revenue_col3": rev_vals[2] if rev_vals and len(rev_vals) > 2 else None,
        "pat_col1":     pat_vals[0] if pat_vals else None,
        "pat_col3":     pat_vals[2] if pat_vals and len(pat_vals) > 2 else None,
    }


# ── Main backfill ─────────────────────────────────────────────────────────────

def run(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. Migrations
    print("Running ALTER TABLE migrations...")
    for stmt in _ALTER_STMTS:
        try:
            conn.execute(stmt)
            print(f"  + {stmt.split('ADD COLUMN')[1].strip()}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()

    # 2. Companies table
    print("\nCreating companies table...")
    conn.executescript(_COMPANIES_DDL)
    conn.execute("""
        INSERT OR IGNORE INTO companies (symbol, company, sector)
        SELECT symbol, company,
               (SELECT sector_tags FROM announcements a
                WHERE a.symbol = f.symbol AND a.sector_tags IS NOT NULL
                  AND a.sector_tags != ''
                ORDER BY broadcast_dt DESC LIMIT 1)
        FROM (SELECT DISTINCT symbol, company FROM financial_results) f
    """)
    r = conn.execute("SELECT COUNT(*) FROM companies").fetchone()
    print(f"  {r[0]} companies inserted")
    conn.commit()

    # 3. Build sector map from announcements (symbol → most common sector_tags)
    print("\nBuilding sector map from announcements...")
    sector_map: dict[str, str] = {}
    rows = conn.execute("""
        SELECT symbol, sector_tags, COUNT(*) c
        FROM announcements
        WHERE sector_tags IS NOT NULL AND sector_tags != ''
        GROUP BY symbol, sector_tags
        ORDER BY c DESC
    """).fetchall()
    for row in rows:
        sym = row["symbol"]
        if sym not in sector_map:  # first = most common
            sector_map[sym] = row["sector_tags"]
    print(f"  {len(sector_map)} symbols with sector data")

    # 4. Backfill financial_results
    print("\nBackfilling financial_results...")
    rows = conn.execute("""
        SELECT id, symbol, company, period, period_type,
               broadcast_dt, key_highlights,
               revenue_cr, pat_cr
        FROM financial_results
    """).fetchall()

    updated = 0
    for row in rows:
        row = dict(row)
        rid          = row["id"]
        symbol       = row["symbol"]
        period_type  = row["period_type"] or ""
        broadcast_dt = row["broadcast_dt"] or ""
        raw_period   = row["period"] or ""

        # period_label
        period_label = _normalize_period(raw_period, broadcast_dt)

        # sector (for quarterly/annual, from announcements; for order_win, from highlights)
        hl = _parse_hl(row["key_highlights"] or "[]")
        order_sector = None
        client_name  = None
        client_type  = None
        exec_months  = None
        sector       = None

        if period_type == "order_win":
            client_name  = hl.get("Client") or hl.get("client")
            client_type  = hl.get("Client type") or hl.get("client_type")
            order_sector = hl.get("Sector") or hl.get("sector")
            exec_months  = _execution_months(hl.get("Execution period") or "")
        else:
            sector = sector_map.get(symbol)

        # YoY growth (only for quarterly/annual with matching financial_statements)
        rev_growth = None
        pat_growth = None
        if period_type in ("quarterly", "annual") and row["revenue_cr"]:
            sv = _get_statement_values(conn, symbol, broadcast_dt)
            rev_growth = _compute_growth(sv.get("revenue_col1"), sv.get("revenue_col3"))
            pat_growth = _compute_growth(sv.get("pat_col1"),     sv.get("pat_col3"))

        conn.execute("""
            UPDATE financial_results SET
                period_label     = ?,
                sector           = ?,
                client_name      = ?,
                client_type      = ?,
                order_sector     = ?,
                execution_months = ?,
                revenue_growth_pct = COALESCE(revenue_growth_pct, ?),
                pat_growth_pct     = COALESCE(pat_growth_pct, ?)
            WHERE id = ?
        """, (period_label, sector, client_name, client_type,
              order_sector, exec_months, rev_growth, pat_growth, rid))
        updated += 1

    conn.commit()
    print(f"  {updated} rows backfilled")

    # 5. Create order_wins table and migrate from financial_results
    print("\nCreating order_wins table...")
    from financial.storage import _ORDER_WINS_DDL, _ORDER_SUBJECTS
    conn.executescript(_ORDER_WINS_DDL)

    print("Migrating order_win rows from financial_results...")
    ow_rows = conn.execute("""
        SELECT f.id, f.symbol, f.company, f.broadcast_dt,
               f.key_highlights, f.raw_summary, f.source_url,
               a.order_value_cr AS ann_order_val,
               src_ann.subject AS source_subject
        FROM financial_results f
        LEFT JOIN announcements a
          ON f.symbol = a.symbol AND DATE(f.broadcast_dt) = DATE(a.broadcast_dt)
         AND a.order_value_cr IS NOT NULL
        LEFT JOIN announcements src_ann ON src_ann.attachment = f.source_url
        WHERE f.period_type = 'order_win'
    """).fetchall()

    _ov_re = re.compile(r'[Oo]rder\s+value\s*[:\-]\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*Cr', re.I)
    migrated = 0
    for ow in ow_rows:
        ow = dict(ow)
        hl          = _parse_hl(ow["key_highlights"] or "[]")
        client_name  = hl.get("Client")
        client_type  = hl.get("Client type")
        order_sector = hl.get("Sector")
        exec_months  = _execution_months(hl.get("Execution period") or "")
        description  = hl.get("Description")

        # Resolve order value: announcements JOIN → key_highlights text
        order_val = ow["ann_order_val"]
        if order_val is None:
            ov_text = (ow["key_highlights"] or "") + " " + (ow["raw_summary"] or "")
            m = _ov_re.search(ov_text)
            if m:
                try:
                    order_val = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass

        from_genuine = 1 if (ow["source_subject"] or "") in _ORDER_SUBJECTS else 0
        # Discard large values from non-genuine sources
        if order_val and not from_genuine and order_val > 500:
            order_val = None

        conn.execute("""
            INSERT OR IGNORE INTO order_wins
                (symbol, company, broadcast_dt, order_value_cr,
                 client_name, client_type, order_sector, execution_months,
                 description, raw_summary, source_url, from_genuine_order)
            VALUES (?,?,?,?,  ?,?,?,?,  ?,?,?,?)
        """, (
            ow["symbol"], ow["company"], ow["broadcast_dt"], order_val,
            client_name, client_type, order_sector, exec_months,
            description, ow["raw_summary"], ow["source_url"], from_genuine,
        ))
        migrated += 1

    conn.commit()
    print(f"  {migrated} order win rows migrated to order_wins table")

    # Quick stats
    print("\nFinal column coverage:")
    for col in ["period_label", "sector", "revenue_growth_pct", "pat_growth_pct"]:
        r = conn.execute(
            f"SELECT COUNT(*) FROM financial_results WHERE {col} IS NOT NULL AND {col} != ''"
        ).fetchone()
        print(f"  financial_results.{col:20} {r[0]:>5} populated")

    for col in ["order_value_cr", "client_name", "client_type", "order_sector"]:
        r = conn.execute(
            f"SELECT COUNT(*) FROM order_wins WHERE {col} IS NOT NULL AND {col} != ''"
        ).fetchone()
        print(f"  order_wins.{col:25} {r[0]:>5} populated")

    total_ow = conn.execute("SELECT COUNT(*) FROM order_wins").fetchone()[0]
    print(f"\n  order_wins total rows: {total_ow}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
