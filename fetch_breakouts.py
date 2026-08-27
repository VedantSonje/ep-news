"""
fetch_breakouts.py — Auto-download Chartink screener signals and store in DB.

Screeners fetched daily:
  1. HVY breakout    — https://chartink.com/screener/copy-hvy-atfinallynitin-537
  2. Intraday rng>9% — https://chartink.com/screener/intraday-range-greater-than-9-pct

Usage:
    python fetch_breakouts.py                  # fetch all screeners for today
    python fetch_breakouts.py --date 2026-08-14
    python fetch_breakouts.py --days 3         # fetch last N trading days

Requires in .env:
    CHARTINK_EMAIL=...
    CHARTINK_PASSWORD=...
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

ROOT        = Path(__file__).resolve().parent
PROCESS_URL = "https://chartink.com/screener/process"
LOGIN_URL   = "https://chartink.com/login"

# All screeners to run daily — add new ones here
SCREENERS = [
    {
        "name":    "hvy_breakout",
        "url":     "https://chartink.com/screener/copy-hvy-atfinallynitin-537",
        "scan_id": 19532327,   # fallback when scan_clause cannot be scraped
    },
    {
        "name":    "intraday_range_gt9",
        "url":     "https://chartink.com/screener/intraday-range-greater-than-9-pct",
        "scan_id": None,
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _login(session: requests.Session) -> bool:
    email    = os.getenv("CHARTINK_EMAIL", "")
    password = os.getenv("CHARTINK_PASSWORD", "")
    if not email or not password:
        print("ERROR: set CHARTINK_EMAIL and CHARTINK_PASSWORD in .env")
        return False

    session.get("https://chartink.com", headers=HEADERS, timeout=15)

    xsrf = urllib.parse.unquote(session.cookies.get("XSRF-TOKEN", ""))
    if not xsrf:
        print("ERROR: could not get XSRF-TOKEN cookie")
        return False

    resp = session.post(LOGIN_URL, headers={
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "X-XSRF-TOKEN": xsrf,
        "Referer":      LOGIN_URL,
        "Origin":       "https://chartink.com",
    }, data={"email": email, "password": password},
       timeout=15, allow_redirects=True)

    if "logout" in resp.text.lower():
        print("Chartink login OK")
        return True

    print("ERROR: Chartink login failed — check CHARTINK_EMAIL / CHARTINK_PASSWORD")
    return False


def _get_csrf(session: requests.Session, screener_url: str) -> str:
    session.get(screener_url, headers=HEADERS, timeout=15)
    xsrf = urllib.parse.unquote(session.cookies.get("XSRF-TOKEN", ""))
    if not xsrf:
        raise RuntimeError("Could not get XSRF-TOKEN after login")
    return xsrf


def _get_scan_clause(session: requests.Session, screener_url: str) -> str | None:
    r = session.get(screener_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    scanner = soup.find("scanner")
    if not scanner:
        return None

    try:
        scan_json = json.loads(scanner.get(":scan-json", "{}"))
        return scan_json.get("atlas_query") or scan_json.get("scan_clause") or None
    except Exception:
        return None


def _run_scan(
    session: requests.Session,
    csrf: str,
    scan_clause: str | None,
    screener_url: str,
    scan_id: int | None,
) -> list[dict]:
    payload: dict = {}
    if scan_clause:
        payload["scan_clause"] = scan_clause
    elif scan_id:
        payload["scan_id"] = str(scan_id)
    else:
        raise RuntimeError("No scan_clause and no scan_id — cannot run scan")

    resp = session.post(PROCESS_URL, headers={
        **HEADERS,
        "Accept":           "application/json, text/javascript, */*; q=0.01",
        "X-XSRF-TOKEN":     csrf,
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type":     "application/x-www-form-urlencoded",
        "Referer":          screener_url,
        "Origin":           "https://chartink.com",
    }, data=payload, timeout=30)

    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response: {resp.text[:300]}")

    return data.get("data", [])


def _classify_cap(row: dict) -> str:
    mc = str(row.get("mcap", "") or row.get("Market Cap", "") or "").lower()
    if "small" in mc:
        return "Smallcap"
    if "mid" in mc:
        return "Midcap"
    if "large" in mc:
        return "Largecap"
    return ""


def _store(db_path: Path, signal_date: str, rows: list[dict], source: str) -> int:
    if not rows:
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS volume_breakouts (
            symbol      TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'hvy_breakout',
            marketcap   TEXT,
            sector      TEXT,
            close       REAL,
            per_chg     REAL,
            volume      INTEGER,
            company     TEXT,
            PRIMARY KEY (symbol, signal_date, source)
        )
    """)
    # Safe migrations for older DBs missing columns
    for col, typ in [
        ("close",   "REAL"),
        ("per_chg", "REAL"),
        ("volume",  "INTEGER"),
        ("company", "TEXT"),
        ("source",  "TEXT NOT NULL DEFAULT 'hvy_breakout'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE volume_breakouts ADD COLUMN {col} {typ}")
        except Exception:
            pass

    # Sector lookup: stock_sectors first, then companies
    sector_map: dict[str, str] = {}
    try:
        for r in conn.execute("SELECT symbol, sector FROM stock_sectors").fetchall():
            if r[1]:
                sector_map[r[0]] = r[1]
    except Exception:
        pass
    for r in conn.execute(
        "SELECT symbol, sector FROM companies WHERE sector IS NOT NULL"
    ).fetchall():
        if r[0] not in sector_map:
            sector_map[r[0]] = r[1] or ""

    # Marketcap lookup from companies.market_cap_cr
    cap_map: dict[str, str] = {}
    for r in conn.execute(
        "SELECT symbol, market_cap_cr FROM companies WHERE market_cap_cr IS NOT NULL"
    ).fetchall():
        cr = r[1] or 0
        if cr >= 20000:
            cap_map[r[0]] = "Largecap"
        elif cr >= 5000:
            cap_map[r[0]] = "Midcap"
        elif cr > 0:
            cap_map[r[0]] = "Smallcap"

    inserted = 0
    for row in rows:
        sym = (
            row.get("nsecode") or row.get("symbol") or row.get("Symbol") or ""
        ).strip().upper()
        if not sym:
            continue
        mc      = _classify_cap(row) or cap_map.get(sym, "")
        sector  = sector_map.get(sym, "")
        close   = row.get("close")
        per_chg = row.get("per_chg")
        volume  = row.get("volume")
        company = row.get("name", "")
        try:
            conn.execute("""
                INSERT INTO volume_breakouts
                    (symbol, signal_date, source, marketcap, sector, close, per_chg, volume, company)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, signal_date, source) DO UPDATE SET
                    marketcap = excluded.marketcap,
                    sector    = excluded.sector,
                    close     = excluded.close,
                    per_chg   = excluded.per_chg,
                    volume    = excluded.volume,
                    company   = excluded.company
            """, (sym, signal_date, source, mc, sector, close, per_chg, volume, company))
            inserted += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return inserted


def _trading_days(n: int) -> list[date]:
    result = []
    d = date.today()
    while len(result) < n:
        if d.weekday() < 5:
            result.append(d)
        d -= timedelta(days=1)
    return list(reversed(result))


def fetch_screener(
    session: requests.Session,
    screener: dict,
    target_date: str,
    db_path: Path,
) -> int:
    name = screener["name"]
    url  = screener["url"]

    print(f"\n  [{name}]  {url}")
    csrf         = _get_csrf(session, url)
    scan_clause  = _get_scan_clause(session, url)

    if scan_clause:
        print(f"  Scan clause found ({len(scan_clause)} chars)")
    else:
        print("  Scan clause not found — using scan_id fallback")

    rows = _run_scan(session, csrf, scan_clause, url, screener.get("scan_id"))
    print(f"  Chartink returned {len(rows)} symbols")

    if not rows:
        print("  No data — market may be closed or login needed")
        return 0

    if rows:
        print("  Sample:", rows[0])

    count = _store(db_path, target_date, rows, source=name)
    print(f"  Stored {count} new signals for {target_date}")
    return count


def fetch(target_date: str, db_path: Path) -> int:
    session = requests.Session()
    if not _login(session):
        return 0

    time.sleep(1)

    total = 0
    for screener in SCREENERS:
        try:
            total += fetch_screener(session, screener, target_date, db_path)
            time.sleep(2)  # be polite between screeners
        except Exception as e:
            print(f"  ERROR fetching {screener['name']}: {e}")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Chartink screener signals into DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", default="", help="Specific date YYYY-MM-DD (default: today)")
    group.add_argument("--days", type=int,   help="Fetch last N trading days")
    args = parser.parse_args()

    from config import AppConfig
    cfg = AppConfig.from_env()

    if args.days:
        targets = [d.isoformat() for d in _trading_days(args.days)]
    else:
        targets = [args.date or date.today().isoformat()]

    grand_total = 0
    for t in targets:
        print(f"\n── {t} ──────────────────────────────")
        grand_total += fetch(t, cfg.db_path)

    print(f"\nDone. Total new signals stored: {grand_total}")


if __name__ == "__main__":
    main()
