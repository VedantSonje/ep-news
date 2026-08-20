"""
backfill_sectors.py — Fill missing sector_tags in announcements + financial_results.

Runs the regex-based SectorTagger on every row that has a null/empty sector_tags
and writes the result back. Zero LLM cost — pure regex, runs in seconds.

Usage:
    python backfill_sectors.py          # dry-run: print counts only
    python backfill_sectors.py --apply  # write updates to DB
"""
from __future__ import annotations

import argparse
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
from enrichment.sector_tagger import SectorTagger


def backfill(db_path: Path, apply: bool = False) -> None:
    tagger = SectorTagger()
    conn   = sqlite3.connect(str(db_path))

    # ── 1. announcements table ────────────────────────────────────────────────
    rows = conn.execute("""
        SELECT id, symbol, company, subject, details
        FROM announcements
        WHERE sector_tags IS NULL OR sector_tags = '' OR sector_tags = 'unknown'
    """).fetchall()

    ann_updates = []
    for row_id, symbol, company, subject, details in rows:
        tags = tagger.tag(subject or "", details or "", company or "")
        if tags:
            ann_updates.append((", ".join(tags), row_id))

    print(f"announcements: {len(rows)} missing sector_tags → {len(ann_updates)} can be filled")

    # ── 2. financial_results table ────────────────────────────────────────────
    # financial_results has a 'sector' column (not sector_tags)
    try:
        fin_rows = conn.execute("""
            SELECT id, symbol, company, raw_summary
            FROM financial_results
            WHERE sector IS NULL OR sector = ''
        """).fetchall()
    except sqlite3.OperationalError:
        fin_rows = []

    fin_updates = []
    for row_id, symbol, company, raw_summary in fin_rows:
        tags = tagger.tag("", raw_summary or "", company or "")
        if tags:
            fin_updates.append((tags[0], row_id))  # single primary sector for financials

    print(f"financial_results: {len(fin_rows)} missing sector → {len(fin_updates)} can be filled")

    total = len(ann_updates) + len(fin_updates)
    print(f"\nTotal rows to update: {total}")

    if not apply:
        print("Dry-run — pass --apply to write changes.")

        # Show sample
        if ann_updates:
            print("\nSample announcement updates:")
            for tag, row_id in ann_updates[:5]:
                r = conn.execute(
                    "SELECT symbol, subject FROM announcements WHERE id=?", (row_id,)
                ).fetchone()
                if r:
                    print(f"  [{row_id}] {r[0]} — {r[1][:60]} → {tag}")
        conn.close()
        return

    # Apply
    if ann_updates:
        conn.executemany(
            "UPDATE announcements SET sector_tags = ? WHERE id = ?", ann_updates
        )
        print(f"Updated {len(ann_updates)} announcements")

    if fin_updates:
        conn.executemany(
            "UPDATE financial_results SET sector = ? WHERE id = ?", fin_updates
        )
        print(f"Updated {len(fin_updates)} financial_results")

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write updates to DB")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    backfill(cfg.db_path, apply=args.apply)
