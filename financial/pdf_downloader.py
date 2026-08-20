"""
PDFDownloader — downloads PDF bytes from NSE / BSE archive URLs.

Design
------
- Uses only stdlib (urllib) — no extra dependencies.
- Fakes a browser User-Agent so NSE CDN doesn't block the request.
- Retries up to 2 times with exponential back-off.
- Returns None on any failure so the caller can skip gracefully.
- Skips non-PDF URLs (.zip, .xlsx, etc.) before touching the network.
- Disk cache: successful downloads are stored in data/pdf_cache/ and
  served from there on retry — avoids re-downloading when extraction fails.
"""
from __future__ import annotations

import hashlib
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


# Maximum PDF size we'll process (local extraction only — no Claude token concern)
MAX_PDF_BYTES = 30 * 1024 * 1024

_DEFAULT_CACHE_DIR = Path("data/pdf_cache")


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

    def __init__(
        self,
        timeout:     int  = 30,
        max_retries: int  = 2,
        cache_dir:   Path = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._timeout     = timeout
        self._max_retries = max_retries
        self._cache_dir   = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Permissive SSL context (NSE archive sometimes has chain issues)
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode    = ssl.CERT_NONE

    def _cache_path(self, url: str) -> Path:
        """Deterministic cache filename from URL hash."""
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.pdf"

    # ── public ────────────────────────────────────────────────────────────

    def is_pdf_url(self, url: str) -> bool:
        """Return True only if the URL looks like a direct PDF link."""
        if not url or url == "-":
            return False
        lower = url.lower()
        return lower.endswith(".pdf") and lower.startswith("http")

    def download(self, url: str) -> tuple[bytes | None, str]:
        """
        Download PDF bytes from url, using disk cache when available.
        Returns (bytes, status) where status is one of:
          "ok", "cached", "too_large", "http_error", "timeout", "error"
        """
        # Serve from cache if already downloaded
        cache_file = self._cache_path(url)
        if cache_file.exists():
            try:
                data = cache_file.read_bytes()
                if len(data) >= 512:
                    return data, "cached"
            except OSError:
                pass  # corrupted cache entry — re-download

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

                # Persist to cache for future runs
                try:
                    cache_file.write_bytes(data)
                except OSError:
                    pass  # non-fatal: cache write failure

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

    def cache_stats(self) -> dict:
        """Return count and total size of cached PDFs."""
        files = list(self._cache_dir.glob("*.pdf"))
        total = sum(f.stat().st_size for f in files)
        return {"count": len(files), "size_mb": round(total / 1024 / 1024, 1)}
