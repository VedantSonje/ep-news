"""
fetch_today.py — Fetch latest NSE corporate announcements and ingest into DB.

Usage:
    python fetch_today.py              # fetches from last DB date to today
    python fetch_today.py --days 3     # fetch last N days
    python fetch_today.py --date 2026-07-27  # specific date

Flow:
  1. Hit NSE API for corporate announcements (equities + SME)
  2. Save raw JSON as CSV in data/fetched/
  3. Run the standard ingest pipeline (filter + enrich + store SQL + ChromaDB)
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

# ── Config ────────────────────────────────────────────────────────────────────

NSE_BASE   = "https://www.nseindia.com"
NSE_API    = "/api/corporate-announcements"
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "X-Requested-With": "XMLHttpRequest",
}
FETCH_DIR  = Path("data/fetched")
DB_PATH    = Path("data/ep_news.db")


# ── NSE session + fetch ───────────────────────────────────────────────────────

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Warm up cookies
    try:
        s.get(NSE_BASE, timeout=15)
        time.sleep(1)
        s.get(
            f"{NSE_BASE}/companies-listing/corporate-filings-announcements",
            timeout=15,
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"  [warn] NSE warm-up partial: {e}")
    return s


def _fetch_nse(session: requests.Session, from_dt: date, to_dt: date) -> list[dict]:
    """Fetch NSE announcements. Returns list of raw JSON records."""
    from_str = from_dt.strftime("%d-%m-%Y")
    to_str   = to_dt.strftime("%d-%m-%Y")
    params   = {"index": "equities", "from_date": from_str, "to_date": to_str}

    records: list[dict] = []
    for idx in ("equities", "sme"):
        params["index"] = idx
        try:
            r = session.get(
                NSE_BASE + NSE_API,
                params=params,
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict) and "data" in data:
                records.extend(data["data"])
            print(f"  NSE [{idx}] {from_str}→{to_str}: {len(data) if isinstance(data,list) else len(data.get('data',[]))} records")
        except Exception as e:
            print(f"  [warn] NSE {idx} fetch error: {e}")
        time.sleep(0.3)
    return records


def _nse_to_row(r: dict) -> dict:
    """Normalise a raw NSE JSON record to a flat dict for our CSV."""
    dt_raw = (
        r.get("bcastDt") or r.get("broadcast_date") or
        r.get("exchdisstime") or r.get("date") or ""
    )
    # NSE attachment URL
    attach = ""
    attid  = r.get("attchmntFile") or r.get("attachment") or ""
    if attid:
        attach = (
            f"https://nsearchives.nseindia.com/corporate/{attid}"
            if not attid.startswith("http") else attid
        )

    subject = r.get("desc") or r.get("subject") or r.get("an_type") or ""
    # attchmntText has the actual announcement description; fall back to subject
    details = r.get("attchmntText") or r.get("body") or r.get("details") or subject

    return {
        "Symbol":            r.get("symbol") or r.get("sm_isin") or "",
        "Company Name":      r.get("comp") or r.get("sm_name") or r.get("company") or "",
        "Subject":           subject,
        "Details":           details,
        "Broadcast Date/Time": dt_raw,
        "Attachment":        attach,
    }


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_FIELDS = ["Symbol", "Company Name", "Subject", "Details",
              "Broadcast Date/Time", "Attachment"]


def _save_csv(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        rows = [_nse_to_row(r) for r in records]
        # Deduplicate by symbol+subject+date
        seen: set[tuple] = set()
        unique = []
        for row in rows:
            key = (row["Symbol"], row["Subject"], row["Broadcast Date/Time"][:10])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        w.writerows(unique)
    return len(unique)


# ── Last date from DB ─────────────────────────────────────────────────────────

def _last_db_date() -> date:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row  = conn.execute("SELECT MAX(broadcast_dt) FROM announcements").fetchone()
        conn.close()
        if row and row[0]:
            return datetime.fromisoformat(row[0][:10]).date()
    except Exception:
        pass
    return date.today() - timedelta(days=1)


# ── Ingest via existing pipeline ─────────────────────────────────────────────

def _ingest(csv_path: Path) -> None:
    from config import AppConfig
    from screener.filter_config import FilterConfig
    from screener.csv_parser import CsvParser
    from screener.scoring_engine import AnnouncementFilter
    from screener.pipeline import EPScreener
    from storage.database_manager import DatabaseManager
    from storage.sql_storage import SQLiteStorage
    from retrieval.bm25_search import setup_fts

    cfg      = AppConfig.from_env()
    config   = FilterConfig()
    parser   = CsvParser(csv_path)
    fltr     = AnnouncementFilter(config)
    screener = EPScreener(parser, fltr)

    sql_store = SQLiteStorage(cfg.db_path)
    setup_fts(sql_store.connection)
    results   = screener.run(conn=sql_store.connection)
    sql_store.close()

    print(f"  Screener passed : {len(results)} announcements")

    if not results:
        print("  Nothing new to store.")
        return

    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        counts = db.save_all(results, source_file=str(csv_path))

    print(f"  SQL new rows    : {counts['sql']}")
    print(f"  ChromaDB upserted: {counts['vector']}")

    if counts["sql"] > 0:
        print("\n  Top new announcements:")
        for sa in results[:8]:
            ann = sa.announcement
            print(f"    [{getattr(sa,'score',0):>2}]  {ann.symbol:<12} {ann.company[:28]}")
            print(f"          {ann.broadcast_dt or ''!s:.16}  |  {ann.subject[:50]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NSE announcements and ingest into DB")
    ap.add_argument("--days",  type=int, default=None, metavar="N",
                    help="Fetch last N days (default: since last DB entry)")
    ap.add_argument("--date",  type=str, default=None, metavar="YYYY-MM-DD",
                    help="Fetch a specific date only")
    ap.add_argument("--from-date", type=str, default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to-date",   type=str, default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--no-ingest", action="store_true",
                    help="Only fetch + save CSV, skip DB ingest")
    args = ap.parse_args()

    today = date.today()

    if args.date:
        from_dt = to_dt = date.fromisoformat(args.date)
    elif args.from_date:
        from_dt = date.fromisoformat(args.from_date)
        to_dt   = date.fromisoformat(args.to_date) if args.to_date else today
    elif args.days:
        from_dt = today - timedelta(days=args.days - 1)
        to_dt   = today
    else:
        # Default: from day after last DB entry to today
        last = _last_db_date()
        from_dt = last + timedelta(days=1)
        to_dt   = today

    print(f"\n{'='*60}")
    print(f"  Fetching NSE announcements: {from_dt} → {to_dt}")
    print(f"{'='*60}\n")

    if from_dt > today:
        print("  DB is already up to date.")
        return

    session = _nse_session()
    records = _fetch_nse(session, from_dt, to_dt)

    if not records:
        print("\n  No announcements returned from NSE API.")
        print("  The exchange may be closed (weekend/holiday) or the API is unavailable.")
        return

    csv_name = f"nse_{from_dt}_{to_dt}.csv".replace("-", "")
    csv_path = FETCH_DIR / csv_name
    n_saved  = _save_csv(records, csv_path)
    print(f"\n  Saved {n_saved} unique announcements → {csv_path}")

    if args.no_ingest:
        print("  Skipping ingest (--no-ingest).")
        return

    print("\n  Running ingest pipeline...")
    _ingest(csv_path)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
