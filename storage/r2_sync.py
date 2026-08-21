"""
R2 sync — upload local data/ to Cloudflare R2 after ingestion,
download on Railway cold start when data/ is absent.

R2 is S3-compatible so boto3 works with a custom endpoint.

Required env vars (set in .env locally, Railway env vars on cloud):
    R2_ACCOUNT_ID        Cloudflare account ID (from R2 dashboard)
    R2_ACCESS_KEY_ID     R2 API token Access Key ID
    R2_SECRET_ACCESS_KEY R2 API token Secret Access Key
    R2_BUCKET            Bucket name (e.g. "ep-news-data")

What gets stored:
    ep_news.db   → r2://<bucket>/ep_news.db          (single file)
    chroma/      → r2://<bucket>/chroma.tar.gz        (directory packed)
"""
from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path


# ── credentials ───────────────────────────────────────────────────────────────

def _cfg() -> dict | None:
    """Return R2 config dict, or None if not configured."""
    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket     = os.getenv("R2_BUCKET", "ep-news-data").strip()
    if not (account_id and access_key and secret_key):
        return None
    return {
        "endpoint": f"https://{account_id}.r2.cloudflarestorage.com",
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
    }


def is_configured() -> bool:
    return _cfg() is not None


def _client(cfg: dict):
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name="auto",
    )


# ── upload ─────────────────────────────────────────────────────────────────────

def upload(data_dir: Path) -> None:
    """
    Upload ep_news.db and chroma/ from data_dir to R2.
    Called at the end of run_daily.py.
    """
    cfg = _cfg()
    if cfg is None:
        print("[r2] R2 not configured — skipping upload", flush=True)
        return

    client = _client(cfg)
    bucket = cfg["bucket"]

    # 1. SQLite DB
    db_path = data_dir / "ep_news.db"
    if db_path.exists():
        size_mb = db_path.stat().st_size / 1_048_576
        print(f"[r2] Uploading ep_news.db ({size_mb:.1f} MB) …", flush=True)
        t0 = time.time()
        client.upload_file(str(db_path), bucket, "ep_news.db")
        print(f"[r2] ep_news.db uploaded in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"[r2] ep_news.db not found at {db_path} — skipping", flush=True)

    # 2. ChromaDB directory → tar.gz in memory
    chroma_dir = data_dir / "chroma"
    if chroma_dir.exists():
        print("[r2] Packing chroma/ …", flush=True)
        t0 = time.time()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(chroma_dir, arcname="chroma")
        size_mb = buf.tell() / 1_048_576
        buf.seek(0)
        print(f"[r2] Uploading chroma.tar.gz ({size_mb:.1f} MB) …", flush=True)
        client.upload_fileobj(buf, bucket, "chroma.tar.gz")
        print(f"[r2] chroma.tar.gz uploaded in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"[r2] chroma/ not found at {chroma_dir} — skipping", flush=True)

    print("[r2] Upload complete.", flush=True)


# ── download ───────────────────────────────────────────────────────────────────

def download_if_needed(data_dir: Path) -> bool:
    """
    Download ep_news.db and chroma.tar.gz from R2 if data_dir is empty.
    Returns True if a download was performed, False otherwise.
    Called on Railway startup before the scheduler runs.
    """
    cfg = _cfg()
    if cfg is None:
        return False

    db_path    = data_dir / "ep_news.db"
    chroma_dir = data_dir / "chroma"

    if db_path.exists() and chroma_dir.exists():
        return False  # already have data

    print("[r2] data/ is missing — downloading from R2 …", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _client(cfg)
    bucket = cfg["bucket"]

    # ep_news.db
    try:
        t0 = time.time()
        client.download_file(bucket, "ep_news.db", str(db_path))
        size_mb = db_path.stat().st_size / 1_048_576
        print(f"[r2] ep_news.db downloaded ({size_mb:.1f} MB) in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[r2] ep_news.db download failed: {e}", flush=True)

    # chroma.tar.gz
    try:
        t0 = time.time()
        buf = io.BytesIO()
        client.download_fileobj(bucket, "chroma.tar.gz", buf)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(path=data_dir)
        print(f"[r2] chroma/ extracted in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[r2] chroma.tar.gz download failed: {e}", flush=True)

    print("[r2] Download complete.", flush=True)
    return True
