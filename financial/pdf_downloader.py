"""
PDFDownloader — downloads PDF bytes from NSE / BSE archive URLs.

Design
------
- Uses only stdlib (urllib) — no extra dependencies.
- Fakes a browser User-Agent so NSE CDN doesn't block the request.
- Retries up to 2 times with exponential back-off.
- Returns None on any failure so the caller can skip gracefully.
- Skips non-PDF URLs (.zip, .xlsx, etc.) before touching the network.
"""
from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


# Maximum PDF size we'll process (local extraction only — no Claude token concern)
MAX_PDF_BYTES = 30 * 1024 * 1024


class PDFDownloader:
    """Downloads a single PDF from a public URL and returns its raw bytes."""

    _HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self, timeout: int = 30, max_retries: int = 2) -> None:
        self._timeout    = timeout
        self._max_retries = max_retries
        # Permissive SSL context (NSE archive sometimes has chain issues)
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode    = ssl.CERT_NONE

    # ── public ────────────────────────────────────────────────────────────

    def is_pdf_url(self, url: str) -> bool:
        """Return True only if the URL looks like a direct PDF link."""
        if not url or url == "-":
            return False
        lower = url.lower()
        return lower.endswith(".pdf") and lower.startswith("http")

    def download(self, url: str) -> tuple[bytes | None, str]:
        """
        Download PDF bytes from url.
        Returns (bytes, status) where status is one of:
          "ok", "too_large", "http_error", "timeout", "error"
        """
        req = urllib.request.Request(url, headers=self._HEADERS)

        for attempt in range(self._max_retries + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=self._timeout, context=self._ssl_ctx
                ) as resp:
                    data = resp.read()

                if len(data) > MAX_PDF_BYTES:
                    return None, "too_large"
                if len(data) < 512:        # suspiciously small — probably an error page
                    return None, "error"
                return data, "ok"

            except urllib.error.HTTPError as e:
                if e.code in (403, 404, 410):
                    return None, f"http_{e.code}"
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None, f"http_{e.code}"

            except urllib.error.URLError:
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None, "timeout"

            except Exception:
                return None, "error"

        return None, "error"
