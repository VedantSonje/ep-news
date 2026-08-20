"""
fetch_breakouts.py — Auto-download HVY breakout signals from Chartink and store in DB.

Usage:
    python fetch_breakouts.py                  # fetch for today
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

ROOT           = Path(__file__).resolve().parent
SCREENER_URL   = "https://chartink.com/screener/copy-hvy-atfinallynitin-537"
PROCESS_URL    = "https://chartink.com/screener/process"
LOGIN_URL      = "https://chartink.com/login"
SCAN_ID        = 19532327

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

    # Step 1: GET home page to receive XSRF-TOKEN cookie (Laravel standard)
    session.get("https://chartink.com", headers=HEADERS, timeout=15)

    xsrf = urllib.parse.unquote(session.cookies.get("XSRF-TOKEN", ""))
    if not xsrf:
        print("ERROR: could not get XSRF-TOKEN cookie")
        return False

    # Step 2: POST form-encoded credentials with X-XSRF-TOKEN header
    resp = session.post(LOGIN_URL, headers={
        **HEADERS,
        "Content-Type":  "application/x-www-form-urlencoded",
        "X-XSRF-TOKEN":  xsrf,
        "Referer":       LOGIN_URL,
        "Origin":        "https://chartink.com",
    }, data={
        "email":    email,
        "password": password,
    }, timeout=15, allow_redirects=True)

    if "logout" in resp.text.lower():
        print("Chartink login OK")
        return True

    print("ERROR: Chartink login failed — check CHARTINK_EMAIL / CHARTINK_PASSWORD")
    return False


def _get_csrf(session: requests.Session) -> str:
    # After login the XSRF-TOKEN cookie is refreshed — use it directly
    session.get(SCREENER_URL, headers=HEADERS, timeout=15)
    xsrf = urllib.parse.unquote(session.cookies.get("XSRF-TOKEN", ""))
    if not xsrf:
        raise RuntimeError("Could not get XSRF-TOKEN after login")
    return xsrf


def _get_scan_clause(session: requests.Session) -> str | None:
    """Extract the compiled scan clause (atlas_query) from the authenticated screener page."""
    r = session.get(SCREENER_URL, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    scanner = soup.find("scanner")
    if not scanner:
        return None

    try:
        scan_json = json.loads(scanner.get(":scan-json", "{}"))
        # atlas_query is the compiled text clause Chartink uses internally
        clause = scan_json.get("atlas_query") or scan_json.get("scan_clause")
        return clause or None
    except Exception:
        return None


def _run_scan(session: requests.Session, csrf: str, scan_clause: str | None) -> list[dict]:
    """POST to Chartink process endpoint and return rows."""
    payload: dict = {}
    if scan_clause:
        payload["scan_clause"] = scan_clause
    else:
        # Fall back to scan_id — works when logged in for saved screeners
        payload["scan_id"] = str(SCAN_ID)

    resp = session.post(PROCESS_URL, headers={
        **HEADERS,
        "Accept":            "application/json, text/javascript, */*; q=0.01",
        "X-XSRF-TOKEN":      csrf,
        "X-Requested-With":  "XMLHttpRequest",
        "Content-Type":      "application/x-www-form-urlencoded",
        "Referer":           SCREENER_URL,
        "Origin":            "https://chartink.com",
    }, data=payload, timeout=30)

    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response: {resp.text[:300]}")

    return data.get("data", [])


def _classify_cap(row: dict) -> str:
    """Map Chartink market cap field to our standard labels."""
    mc = str(row.get("mcap", "") or row.get("Market Cap", "") or "").lower()
    if "small" in mc:
        return "Smallcap"
    if "mid" in mc:
        return "Midcap"
    if "large" in mc:
        return "Largecap"
    return ""


def _store(db_path: Path, signal_date: str, rows: list[dict]) -> int:
    """Upsert breakout rows into volume_breakouts table. Returns count inserted."""
    if not rows:
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS volume_breakouts (
            symbol      TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            marketcap   TEXT,
            sector      TEXT,
            close       REAL,
            per_chg     REAL,
            volume      INTEGER,
            company     TEXT,
            PRIMARY KEY (symbol, signal_date)
        )
    """)
    # Add columns if they don't exist yet (safe migration)
    for col, typ in [("close","REAL"),("per_chg","REAL"),("volume","INTEGER"),("company","TEXT")]:
        try:
            conn.execute(f"ALTER TABLE volume_breakouts ADD COLUMN {col} {typ}")
        except Exception:
            pass

    # Pull sector from stock_sectors (most complete source), fall back to companies
    sector_map: dict[str, str] = {}
    try:
        for r in conn.execute("SELECT symbol, sector FROM stock_sectors").fetchall():
            if r[1]:
                sector_map[r[0]] = r[1]
    except Exception:
        pass
    for r in conn.execute("SELECT symbol, sector FROM companies WHERE sector IS NOT NULL").fetchall():
        if r[0] not in sector_map:
            sector_map[r[0]] = r[1] or ""

    # Marketcap from companies.market_cap_cr (numeric → label)
    cap_map: dict[str, str] = {}
    for r in conn.execute("SELECT symbol, market_cap_cr FROM companies WHERE market_cap_cr IS NOT NULL").fetchall():
        cr = r[1] or 0
        if cr >= 20000:
            cap_map[r[0]] = "Largecap"
        elif cr >= 5000:
            cap_map[r[0]] = "Midcap"
        elif cr > 0:
            cap_map[r[0]] = "Smallcap"

    inserted = 0
    for row in rows:
        sym = (row.get("nsecode") or row.get("symbol") or row.get("Symbol") or "").strip().upper()
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
                    (symbol, signal_date, marketcap, sector, close, per_chg, volume, company)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, signal_date) DO UPDATE SET
                    marketcap = excluded.marketcap,
                    sector    = excluded.sector,
                    close     = excluded.close,
                    per_chg   = excluded.per_chg,
                    volume    = excluded.volume,
                    company   = excluded.company
            """, (sym, signal_date, mc, sector, close, per_chg, volume, company))
            inserted += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return inserted


def _trading_days(n: int) -> list[date]:
    """Return last n trading days (Mon-Fri) up to today."""
    result = []
    d = date.today()
    while len(result) < n:
        if d.weekday() < 5:
            result.append(d)
        d -= timedelta(days=1)
    return list(reversed(result))


def fetch(target_date: str, db_path: Path) -> int:
    session = requests.Session()

    if not _login(session):
        return 0

    time.sleep(1)  # be polite

    csrf         = _get_csrf(session)
    scan_clause  = _get_scan_clause(session)
    if scan_clause:
        print(f"Scan clause found ({len(scan_clause)} chars)")
    else:
        print("Scan clause not found — using scan_id fallback")

    rows = _run_scan(session, csrf, scan_clause)
    print(f"Chartink returned {len(rows)} symbols")

    if not rows:
        print("No data returned — Chartink may require manual login or the market was closed")
        return 0

    if rows:
        print("Sample row keys:", list(rows[0].keys()))
        print("Sample row:", rows[0])

    count = _store(db_path, target_date, rows)
    print(f"Stored {count} new breakout signals for {target_date}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Chartink HVY breakouts into DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date",  default="", help="Specific date YYYY-MM-DD (default: today)")
    group.add_argument("--days",  type=int,   help="Fetch last N trading days")
    args = parser.parse_args()

    from config import AppConfig
    cfg = AppConfig.from_env()

    if args.days:
        targets = [d.isoformat() for d in _trading_days(args.days)]
    else:
        targets = [args.date or date.today().isoformat()]

    total = 0
    for t in targets:
        print(f"\n── {t} ──────────────────────────────")
        total += fetch(t, cfg.db_path)

    print(f"\nDone. Total new signals stored: {total}")


if __name__ == "__main__":
    main()
