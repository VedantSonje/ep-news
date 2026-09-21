"""
fyers_backfill.py — Backfill close, per_chg, volume for volume_breakouts rows
               where those fields are NULL, using Fyers historical API.

Setup:
    pip install fyers-apiv3

Usage:
    python fyers_backfill.py                             # backfill NULL rows in volume_breakouts
    python fyers_backfill.py --dry-run                   # show what would be updated, no writes
    python fyers_backfill.py --date 2026-08-07           # only a specific date (volume_breakouts)
    python fyers_backfill.py --from-backtest             # import backtest_signals → volume_breakouts with Fyers prices
    python fyers_backfill.py --from-backtest --since 2026-09-15   # only dates >= this
    python fyers_backfill.py --table trending --since 2026-09-15  # backfill trending_stocks table

Credentials — add to .env:
    FYERS_CLIENT_ID=XXXX-100             (your App ID)
    FYERS_ACCESS_TOKEN=eyJ...            (paste today's token here)
"""

import argparse
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = Path("data/ep_news.db")

# ── Fyers client ──────────────────────────────────────────────────────────────

def _get_fyers():
    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        raise SystemExit("Run:  pip install fyers-apiv3")

    client_id    = os.getenv("FYERS_CLIENT_ID", "").strip()
    access_token = os.getenv("FYERS_ACCESS_TOKEN", "").strip()

    if not client_id or not access_token:
        raise SystemExit(
            "Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in your .env file."
        )

    return fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        log_path="",
    )


# ── Fetch one day's OHLCV for a symbol ───────────────────────────────────────

def fetch_day(fyers, symbol: str, day: str) -> dict | None:
    """
    Returns {"close": float, "per_chg": float, "volume": int} or None.
    day format: "YYYY-MM-DD"
    """
    fyers_sym = f"NSE:{symbol}-EQ"
    data = {
        "symbol":      fyers_sym,
        "resolution":  "1D",
        "date_format": "1",          # unix timestamps
        "range_from":  day,
        "range_to":    day,
        "cont_flag":   "1",
    }
    try:
        resp = fyers.history(data=data)
    except Exception as e:
        print(f"  [error] {symbol} {day}: {e}")
        return None

    if resp.get("s") != "ok":
        # Try BSE fallback
        data["symbol"] = f"BSE:{symbol}"
        try:
            resp = fyers.history(data=data)
        except Exception:
            return None

    candles = resp.get("candles", [])
    if not candles:
        return None

    # candle: [timestamp, open, high, low, close, volume]
    c = candles[0]
    close  = float(c[4])
    open_  = float(c[1])
    volume = int(c[5])
    per_chg = round((close - open_) / open_ * 100, 2) if open_ else None

    return {"close": close, "per_chg": per_chg, "volume": volume}


# ── Main ──────────────────────────────────────────────────────────────────────

# backtest source → volume_breakouts source mapping
_BACKTEST_SOURCE_MAP = {
    "hvy_breakout":        "hvy_breakout",
    "hvy_breakout_v2":     "hvy_breakout",
    "hvy_breakout_v3":     "hvy_breakout",
    "intraday_range_gt9":     "intraday_range_gt9",
    "intraday_range_gt9_v2":  "intraday_range_gt9",
    "intraday_range_gt9_v3":  "intraday_range_gt9",
    "pct_chg_gt9":         "pct_chg_gt9",
    "pct_chg_gt9_v2":      "pct_chg_gt9",
    "pct_chg_gt9_v3":      "pct_chg_gt9",
}


def backfill_table(conn, fyers, table: str, since: str | None, dry_run: bool):
    """Fetch Fyers prices for rows in `table` where close IS NULL."""
    where = "close IS NULL"
    params = []
    if since:
        where += " AND signal_date >= ?"
        params.append(since)

    rows = conn.execute(
        f"SELECT DISTINCT symbol, signal_date FROM {table} WHERE {where} ORDER BY signal_date, symbol",
        params,
    ).fetchall()

    print(f"Rows with missing close in {table}: {len(rows)}")
    if not rows:
        print("Nothing to do.")
        return

    lock    = Lock()
    total   = len(rows)
    counter = [0]
    updated = 0
    failed  = 0

    def _process(item):
        symbol, day = item
        result = fetch_day(fyers, symbol, day)
        with lock:
            counter[0] += 1
            idx = counter[0]
        if result is None:
            print(f"[{idx:4d}/{total}] {symbol:15s} {day}  no data", flush=True)
            return None
        print(f"[{idx:4d}/{total}] {symbol:15s} {day}  close={result['close']:.2f}  chg={result['per_chg']}%", flush=True)
        return (symbol, day, result)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_process, r): r for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            if res and not dry_run:
                symbol, day, d = res
                with lock:
                    conn.execute(
                        f"UPDATE {table} SET close=?, per_chg=?, volume=? WHERE symbol=? AND signal_date=?",
                        (d["close"], d["per_chg"], d["volume"], symbol, day),
                    )
                    conn.commit()
                    updated += 1
            elif res is None:
                failed += 1

    print(f"\nDone.  Updated: {updated}  Failed/no data: {failed}")


def backfill_from_backtest(conn, fyers, since: str | None, dry_run: bool):
    """Read backtest_signals, fetch Fyers prices, insert into volume_breakouts."""
    where = "1=1"
    params = []
    if since:
        where = "signal_date >= ?"
        params = [since]

    rows = conn.execute(
        f"SELECT signal_date, symbol, marketcap, sector, source FROM backtest_signals WHERE {where} ORDER BY signal_date, symbol",
        params,
    ).fetchall()

    # De-duplicate: keep only rows not already in volume_breakouts
    existing = set(conn.execute(
        "SELECT symbol, signal_date, source FROM volume_breakouts"
    ).fetchall())

    to_fetch = []
    for signal_date, symbol, marketcap, sector, bs_source in rows:
        vb_source = _BACKTEST_SOURCE_MAP.get(bs_source, bs_source)
        if (symbol, signal_date, vb_source) not in existing:
            to_fetch.append((signal_date, symbol, marketcap, sector, vb_source))

    print(f"backtest_signals rows in range : {len(rows)}")
    print(f"Already in volume_breakouts     : {len(rows) - len(to_fetch)}")
    print(f"To fetch from Fyers             : {len(to_fetch)}")
    if not to_fetch:
        print("Nothing to do.")
        return

    lock    = Lock()
    total   = len(to_fetch)
    counter = [0]
    inserted = 0
    failed   = 0

    def _process(item):
        signal_date, symbol, marketcap, sector, vb_source = item
        result = fetch_day(fyers, symbol, signal_date)
        with lock:
            counter[0] += 1
            idx = counter[0]
        if result is None:
            print(f"[{idx:4d}/{total}] {symbol:15s} {signal_date}  no data", flush=True)
            return None
        print(f"[{idx:4d}/{total}] {symbol:15s} {signal_date}  close={result['close']:.2f}  chg={result['per_chg']}%", flush=True)
        return (signal_date, symbol, marketcap, sector, vb_source, result)

    WORKERS = 10
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process, r): r for r in to_fetch}
        for fut in as_completed(futures):
            res = fut.result()
            if res and not dry_run:
                signal_date, symbol, marketcap, sector, vb_source, d = res
                with lock:
                    conn.execute(
                        "INSERT OR IGNORE INTO volume_breakouts "
                        "(symbol, signal_date, source, marketcap, sector, close, per_chg, volume, company) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (symbol, signal_date, vb_source, marketcap, sector,
                         d["close"], d["per_chg"], d["volume"], None),
                    )
                    conn.commit()
                    inserted += 1
            elif res is None:
                failed += 1

    print(f"\nDone.  Inserted: {inserted}  Failed/no data: {failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",       action="store_true", help="Print updates, don't write")
    ap.add_argument("--date",          help="Force-update ALL rows for this date (YYYY-MM-DD), ignoring NULL check")
    ap.add_argument("--source",        default="hvy_breakout", help="Source filter (default: hvy_breakout)")
    ap.add_argument("--from-backtest", action="store_true", help="Import backtest_signals → volume_breakouts using Fyers prices")
    ap.add_argument("--table",         default="", help="Table to backfill prices into: 'trending' = trending_stocks")
    ap.add_argument("--since",         help="Only process dates >= YYYY-MM-DD")
    args = ap.parse_args()

    fyers = _get_fyers()
    conn  = sqlite3.connect(str(DB_PATH))

    if args.from_backtest:
        backfill_from_backtest(conn, fyers, args.since, args.dry_run)
        conn.close()
        return

    if args.table == "trending":
        backfill_table(conn, fyers, "trending_stocks", args.since, args.dry_run)
        conn.close()
        return

    # Original mode: update NULL rows in volume_breakouts
    if args.date:
        where  = "source = ? AND signal_date = ?"
        params = [args.source, args.date]
    else:
        where  = "source = ? AND close IS NULL"
        params = [args.source]

    rows = conn.execute(
        f"SELECT DISTINCT symbol, signal_date FROM volume_breakouts WHERE {where} ORDER BY signal_date",
        params,
    ).fetchall()

    print(f"Rows to backfill: {len(rows)}")
    if not rows:
        print("Nothing to do.")
        conn.close()
        return

    updated  = 0
    failed   = 0
    lock     = Lock()
    total    = len(rows)
    counter  = [0]

    def _process(item):
        symbol, day = item
        result = fetch_day(fyers, symbol, day)
        with lock:
            counter[0] += 1
            idx = counter[0]
        if result is None:
            print(f"[{idx:4d}/{total}] {symbol:15s} {day}  no data", flush=True)
            return None
        print(f"[{idx:4d}/{total}] {symbol:15s} {day}  close={result['close']:.2f}  chg={result['per_chg']}%", flush=True)
        return (symbol, day, result)

    WORKERS = 20
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process, r): r for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            if res and not args.dry_run:
                symbol, day, d = res
                with lock:
                    conn.execute(
                        "UPDATE volume_breakouts SET close=?, per_chg=?, volume=? "
                        "WHERE symbol=? AND signal_date=? AND source=?",
                        (d["close"], d["per_chg"], d["volume"], symbol, day, args.source),
                    )
                    conn.commit()
                    updated += 1
            elif res is None:
                failed += 1

    conn.close()
    print(f"\nDone.  Updated: {updated}  Failed/no data: {failed}")


if __name__ == "__main__":
    main()
