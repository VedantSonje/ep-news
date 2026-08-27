"""
_import_backtest.py — Import backtest CSVs into the backtest_signals table.

Each CSV represents stocks passing a screener on a given date.
Source tags:
  pct_chg_gt9        — "Backtest % change grt than 9.csv"
  intraday_range_gt9 — "Backtest intraday range greater than 9 pct.csv"
"""
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "data/ep_news.db"

FILES = [
    ("Backtest % change grt than 9.csv",                "pct_chg_gt9"),
    ("Backtest intraday range greater than 9 pct.csv",  "intraday_range_gt9"),
]

conn = sqlite3.connect(DB_PATH)

conn.execute("""
    CREATE TABLE IF NOT EXISTS backtest_signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_date TEXT NOT NULL,
        symbol      TEXT NOT NULL,
        marketcap   TEXT,
        sector      TEXT,
        source      TEXT DEFAULT 'pct_chg_gt9',
        ingested_at TEXT DEFAULT (datetime('now')),
        UNIQUE(signal_date, symbol, source)
    )
""")
conn.commit()

for csv_file, source in FILES:
    path = Path(csv_file)
    if not path.exists():
        print(f"SKIP (not found): {csv_file}")
        continue

    inserted = skipped = 0
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw_date = row["Date"].strip()
            try:
                signal_date = datetime.strptime(raw_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError:
                signal_date = raw_date

            cur = conn.execute(
                "INSERT OR IGNORE INTO backtest_signals "
                "(signal_date, symbol, marketcap, sector, source) VALUES (?,?,?,?,?)",
                (signal_date, row["Symbol"].strip(),
                 row["Marketcapname"].strip(), row["Sector"].strip(), source),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1

    conn.commit()
    total = conn.execute(
        "SELECT COUNT(*) FROM backtest_signals WHERE source=?", (source,)
    ).fetchone()[0]
    dates = conn.execute(
        "SELECT MIN(signal_date), MAX(signal_date) FROM backtest_signals WHERE source=?",
        (source,),
    ).fetchone()
    print(f"[{source}]")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped} (duplicates)")
    print(f"  Total    : {total}")
    print(f"  Dates    : {dates[0]}  to  {dates[1]}")

grand = conn.execute("SELECT COUNT(*) FROM backtest_signals").fetchone()[0]
print(f"\nGrand total in backtest_signals: {grand}")
conn.close()
