"""
fyers_auth.py — Generate Fyers access token and save to .env

Run once per day (tokens expire at midnight):
    python fyers_auth.py

You need:
    FYERS_CLIENT_ID  = XY26680-100       (your App ID + "-100")
    FYERS_SECRET_KEY = your secret key   (from Fyers API dashboard)
    FYERS_REDIRECT_URI = https://trade.fyers.in/api-login/redirect-uri/index.html
                         (or whatever URI you registered)
"""

import os, re, webbrowser
from pathlib import Path
from dotenv import load_dotenv, set_key

load_dotenv()

CLIENT_ID    = os.getenv("FYERS_CLIENT_ID", "XY26680-100")
SECRET_KEY   = os.getenv("FYERS_SECRET_KEY", "").strip()
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI",
                         "https://trade.fyers.in/api-login/redirect-uri/index.html")
ENV_FILE     = Path(".env")

def main():
    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        raise SystemExit("Run:  pip install fyers-apiv3")

    if not SECRET_KEY:
        raise SystemExit(
            "Set FYERS_SECRET_KEY in .env  (from https://myapi.fyers.in/dashboard)"
        )

    # Step 1 — build login URL
    session = fyersModel.SessionModel(
        client_id     = CLIENT_ID,
        secret_key    = SECRET_KEY,
        redirect_uri  = REDIRECT_URI,
        response_type = "code",
        grant_type    = "authorization_code",
    )
    auth_url = session.generate_authcode()
    print(f"\nOpening browser for Fyers login...")
    print(f"URL: {auth_url}\n")
    webbrowser.open(auth_url)

    # Step 2 — get auth_code from --code arg or prompt
    import sys
    code_arg = None
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--code" and i < len(sys.argv) - 1:
            code_arg = sys.argv[i + 1]
        elif a.startswith("--code="):
            code_arg = a.split("=", 1)[1]

    if code_arg:
        raw = code_arg
    else:
        print("After login, you'll be redirected to a URL like:")
        print("  https://www.google.com/?auth_code=eyJ...&state=None\n")
        raw = input("Paste the full redirect URL (or just the auth_code): ").strip()

    # Extract auth_code whether they pasted the full URL or just the code
    match = re.search(r"auth_code=([^&\s]+)", raw)
    auth_code = match.group(1) if match else raw

    # Step 3 — exchange for access token
    session.set_token(auth_code)
    resp = session.generate_token()

    if resp.get("s") != "ok":
        raise SystemExit(f"Token generation failed: {resp}")

    access_token = resp["access_token"]
    print(f"\nAccess token obtained.")

    # Step 4 — save to .env
    set_key(str(ENV_FILE), "FYERS_ACCESS_TOKEN", access_token)
    print(f"Saved FYERS_ACCESS_TOKEN to .env")
    print(f"\nNow run:  python fyers_backfill.py")


if __name__ == "__main__":
    main()
