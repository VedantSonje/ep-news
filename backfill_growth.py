"""
backfill_growth.py — Fill missing revenue_growth_pct / pat_growth_pct in financial_results.

For each row that is missing a growth rate but has a revenue/pat value, look up the
same symbol's prior-period row and compute the YoY growth rate.

Period matching:
  - Quarterly (Q1/Q2/Q3/Q4): compare to same quarter of prior year
  - Annual (FY): compare to prior FY row

Usage:
    python backfill_growth.py          # dry-run: print what would change
    python backfill_growth.py --apply  # write updates to DB
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import AppConfig


def _prior_period(period: str) -> str | None:
    """Return the expected prior-year period string for YoY comparison."""
    # Quarterly: Q1FY26 → Q1FY25, or 2025-Q1 → 2024-Q1
    m = re.match(r"(Q[1-4])FY(\d{2,4})", period, re.IGNORECASE)
    if m:
        q, yr = m.group(1).upper(), int(m.group(2))
        return f"{q}FY{yr - 1:02d}" if yr >= 10 else f"{q}FY{yr - 1}"

    m = re.match(r"(\d{4})-(Q[1-4])", period)
    if m:
        yr, q = int(m.group(1)), m.group(2)
        return f"{yr - 1}-{q}"

    # Annual: FY2026 → FY2025
    m = re.match(r"FY(\d{4})", period, re.IGNORECASE)
    if m:
        return f"FY{int(m.group(1)) - 1}"

    m = re.match(r"(\d{4})-(\d{2,4})", period)
    if m:
        yr = int(m.group(1))
        return f"{yr - 1}-{m.group(2)}"

    return None


def backfill(db_path: Path, apply: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))

    rows = conn.execute("""
        SELECT id, symbol, period, revenue_cr, pat_cr, revenue_growth_pct, pat_growth_pct
        FROM financial_results
        WHERE period_type = 'financial_results'
          AND (revenue_cr IS NOT NULL OR pat_cr IS NOT NULL)
          AND (revenue_growth_pct IS NULL OR pat_growth_pct IS NULL)
        ORDER BY symbol, period
    """).fetchall()

    updates = []
    for row_id, symbol, period, rev, pat, rev_g, pat_g in rows:
        prior = _prior_period(period)
        if prior is None:
            continue

        prior_row = conn.execute("""
            SELECT revenue_cr, pat_cr FROM financial_results
            WHERE symbol = ? AND period = ? AND period_type = 'financial_results'
            LIMIT 1
        """, (symbol, prior)).fetchone()
        if prior_row is None:
            continue

        prior_rev, prior_pat = prior_row
        new_rev_g = rev_g
        new_pat_g = pat_g

        if rev_g is None and rev is not None and prior_rev and prior_rev != 0:
            new_rev_g = round((rev - prior_rev) / abs(prior_rev) * 100, 1)

        if pat_g is None and pat is not None and prior_pat and prior_pat != 0:
            new_pat_g = round((pat - prior_pat) / abs(prior_pat) * 100, 1)

        if new_rev_g != rev_g or new_pat_g != pat_g:
            updates.append((new_rev_g, new_pat_g, row_id, symbol, period))
            print(f"  {symbol} [{period}]: "
                  f"rev_g {rev_g} → {new_rev_g}, pat_g {pat_g} → {new_pat_g}")

    print(f"\nTotal rows to update: {len(updates)}")
    if apply and updates:
        conn.executemany("""
            UPDATE financial_results
            SET revenue_growth_pct = ?, pat_growth_pct = ?
            WHERE id = ?
        """, [(r[0], r[1], r[2]) for r in updates])
        conn.commit()
        print("Applied.")
    elif not apply:
        print("Dry-run — pass --apply to write changes.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write updates to DB")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    backfill(cfg.db_path, apply=args.apply)
