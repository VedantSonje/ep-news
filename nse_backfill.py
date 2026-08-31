"""
nse_backfill.py — Backfill close, per_chg, volume from NSE Bhav Copy files.

One HTTP request per date gets ALL stocks — much faster than per-symbol calls.
150 dates → 150 requests instead of 2,789 individual API calls.

Usage:
    python nse_backfill.py              # backfill all NULL rows
    python nse_backfill.py --dry-run    # preview without writing
"""

import argparse
import io
import sqlite3
import time
import zipfile
from pathlib import Path

import requests

DB_PATH = Path("data/ep_news.db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept":          "*/*",
    "Referer":         "https://www.nseindia.com/",
}


def _bhav_url(date_str: str) -> str:
    """Return NSE bhav copy URL for YYYY-MM-DD."""
    d = date_str.replace("-", "")          # 20260813
    return (
        f"https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip"
    )


def _fetch_bhav(session: requests.Session, date_str: str) -> dict[str, dict] | None:
    """
    Download bhav copy for a date and return {SYMBOL: {close, per_chg, volume}}.
    Returns None on failure.
    """
    url = _bhav_url(date_str)
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = zf.namelist()[0]
            csv_bytes = zf.read(csv_name)
    except Exception as e:
        print(f"  [error] {date_str}: {e}")
        return None

    result: dict[str, dict] = {}
    lines = csv_bytes.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return result

    header = [h.strip().upper() for h in lines[0].split(",")]
    try:
        i_sym   = header.index("SYMBOL")
        i_ser   = header.index("SERIES")
        i_close = header.index("CLOSE")
        i_prev  = header.index("PREVCLOSE")
        i_vol   = header.index("TOTTRDQTY")
    except ValueError:
        return None

    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) <= max(i_sym, i_close, i_prev, i_vol):
            continue
        series = cols[i_ser].strip()
        if series != "EQ":
            continue
        sym = cols[i_sym].strip()
        try:
            close    = float(cols[i_close].strip())
            prev     = float(cols[i_prev].strip())
            volume   = int(float(cols[i_vol].strip()))
            per_chg  = round((close - prev) / prev * 100, 2) if prev else None
        except (ValueError, ZeroDivisionError):
            continue
        result[sym] = {"close": close, "per_chg": per_chg, "volume": volume}

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", default="hvy_breakout")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    # All (symbol, signal_date) pairs with NULL close
    rows = conn.execute(
        "SELECT symbol, signal_date FROM volume_breakouts "
        "WHERE source = ? AND close IS NULL ORDER BY signal_date, symbol",
        (args.source,),
    ).fetchall()

    if not rows:
        print("Nothing to backfill.")
        return

    # Group by date
    by_date: dict[str, list[str]] = {}
    for sym, dt in rows:
        by_date.setdefault(dt, []).append(sym)

    print(f"Rows to backfill : {len(rows)}")
    print(f"Unique dates     : {len(by_date)}")
    print()

    session  = requests.Session()
    updated  = 0
    no_data  = 0

    for i, (date_str, syms) in enumerate(sorted(by_date.items()), 1):
        print(f"[{i:3d}/{len(by_date)}] {date_str}  ({len(syms)} stocks) ... ", end="", flush=True)
        bhav = _fetch_bhav(session, date_str)

        if bhav is None:
            print("download failed")
            no_data += len(syms)
            time.sleep(1)
            continue

        day_updated = 0
        day_missing = 0
        for sym in syms:
            if sym in bhav:
                d = bhav[sym]
                if not args.dry_run:
                    conn.execute(
                        "UPDATE volume_breakouts SET close=?, per_chg=?, volume=? "
                        "WHERE symbol=? AND signal_date=? AND source=?",
                        (d["close"], d["per_chg"], d["volume"], sym, date_str, args.source),
                    )
                day_updated += 1
            else:
                day_missing += 1

        if not args.dry_run:
            conn.commit()

        updated  += day_updated
        no_data  += day_missing
        print(f"updated={day_updated}  missing={day_missing}  (bhav had {len(bhav)} stocks)")
        time.sleep(0.3)   # polite delay between dates

    conn.close()
    print(f"\nDone.  Updated: {updated}  No data: {no_data}")


if __name__ == "__main__":
    main()
