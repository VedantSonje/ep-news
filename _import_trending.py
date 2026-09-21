"""
_import_trending.py — Import Chartink "Backtest trending stocks.csv" into trending_stocks table.
"""
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH  = "data/ep_news.db"
CSV_FILE = "Backtest trending stocks.csv"

conn = sqlite3.connect(DB_PATH)

conn.execute("""
    CREATE TABLE IF NOT EXISTS trending_stocks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_date TEXT NOT NULL,
        symbol      TEXT NOT NULL,
        marketcap   TEXT,
        sector      TEXT,
        close       REAL,
        per_chg     REAL,
        volume      INTEGER,
        company     TEXT,
        ingested_at TEXT DEFAULT (datetime('now')),
        UNIQUE(signal_date, symbol)
    )
""")
conn.commit()

path = Path(CSV_FILE)
if not path.exists():
    raise SystemExit(f"File not found: {CSV_FILE}")

inserted = skipped = 0
with open(path, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        raw_date = row["Date"].strip()
        try:
            signal_date = datetime.strptime(raw_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        except ValueError:
            signal_date = raw_date

        cur = conn.execute(
            "INSERT OR IGNORE INTO trending_stocks "
            "(signal_date, symbol, marketcap, sector) VALUES (?,?,?,?)",
            (signal_date, row["Symbol"].strip(),
             row["Marketcapname"].strip(), row["Sector"].strip()),
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1

conn.commit()

total = conn.execute("SELECT COUNT(*) FROM trending_stocks").fetchone()[0]
dates = conn.execute("SELECT MIN(signal_date), MAX(signal_date) FROM trending_stocks").fetchone()
print(f"Inserted : {inserted}")
print(f"Skipped  : {skipped} (duplicates)")
print(f"Total    : {total}")
print(f"Dates    : {dates[0]}  to  {dates[1]}")
conn.close()
