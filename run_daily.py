"""
run_daily.py — End-of-day automation for EP News.

Chains in order:
  1. fetch_today.py   — pull today's NSE announcements into DB
  2. main.py extract  — download PDFs + extract financial results
  3. fetch_breakouts.py — fetch HVY breakout signals from Chartink
  4. Generate daily market brief (via brief_agent)

Usage:
    python run_daily.py                  # run everything for today
    python run_daily.py --date 2026-08-15  # specific date
    python run_daily.py --skip-extract   # skip slow PDF extraction
    python run_daily.py --skip-brief     # skip brief generation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent


def _run(label: str, cmd: list[str], check: bool = True) -> int:
    print(f"\n{'='*60}", flush=True)
    print(f"[daily] STEP: {label}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[daily] {label} → {status} in {elapsed:.1f}s", flush=True)
    if check and result.returncode != 0:
        print(f"[daily] Aborting — {label} failed.", flush=True)
        sys.exit(result.returncode)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="EP News daily automation")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip PDF extraction (step 2)")
    parser.add_argument("--skip-breakouts", action="store_true",
                        help="Skip Chartink breakout fetch (step 3)")
    parser.add_argument("--skip-brief", action="store_true",
                        help="Skip brief generation (step 4)")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip R2 upload (step 5)")
    args = parser.parse_args()

    target = args.date
    py = sys.executable

    print(f"[daily] Starting EP News daily run for {target}", flush=True)
    t_start = time.time()

    # Step 1 — fetch today's NSE announcements
    _run("Fetch NSE announcements", [py, "fetch_today.py", "--date", target])

    # Step 2 — extract financial PDFs
    if not args.skip_extract:
        _run("Extract financial PDFs", [py, "main.py", "extract"], check=False)
    else:
        print("[daily] Skipping PDF extraction (--skip-extract)", flush=True)

    # Step 3 — fetch breakout signals from Chartink
    if not args.skip_breakouts:
        _run("Fetch Chartink breakouts", [py, "fetch_breakouts.py", "--date", target],
             check=False)
    else:
        print("[daily] Skipping breakout fetch (--skip-breakouts)", flush=True)

    # Step 4 — generate daily brief
    if not args.skip_brief:
        print(f"\n{'='*60}", flush=True)
        print("[daily] STEP: Generate daily market brief", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from config import AppConfig
            from api.brief_agent import generate_daily_brief
            cfg = AppConfig.from_env()
            result = generate_daily_brief(cfg.db_path, target_date=target)
            if result:
                print(f"[daily] Brief generated ({len(result.get('content',''))} chars)", flush=True)
            else:
                print("[daily] Brief skipped (no announcements)", flush=True)
        except Exception as e:
            print(f"[daily] Brief generation error: {e}", flush=True)
    else:
        print("[daily] Skipping brief generation (--skip-brief)", flush=True)

    # Step 5 — sync data to Cloudflare R2 so Railway picks it up
    if not args.skip_sync:
        print(f"\n{'='*60}", flush=True)
        print("[daily] STEP: Sync data to R2", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from storage.r2_sync import upload, is_configured
            if is_configured():
                upload(BASE / "data")
            else:
                print("[daily] R2 not configured — set R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY in .env to enable sync", flush=True)
        except Exception as e:
            print(f"[daily] R2 sync error: {e}", flush=True)
    else:
        print("[daily] Skipping R2 sync (--skip-sync)", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[daily] Done. Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
