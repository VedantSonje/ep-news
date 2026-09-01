"""
eval/pdf_latency_eval.py — Measure PDF extraction latency per layer.

Layers timed:
  1. pdfplumber  — PDF bytes → plain text
  2. rule        — regex extractor on text
  3. ollama      — local LLM extraction (Ollama must be running)

Uses already-cached PDFs from data/pdfs/ so no network requests needed.
Picks a random sample from the cache, skipping tiny files (<5 KB).

Usage:
    python eval/pdf_latency_eval.py                 # sample 20 PDFs
    python eval/pdf_latency_eval.py --n 50          # sample 50 PDFs
    python eval/pdf_latency_eval.py --n 20 --no-ollama  # skip Ollama layer
    python eval/pdf_latency_eval.py --symbol HFCL   # only that symbol's PDFs

Output: per-PDF row table + p50/p95 summary per layer.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_PATH    = Path("data/ep_news.db")
PDF_CACHE  = Path("data/pdfs")
MIN_BYTES  = 5_000   # skip stub/empty PDFs


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    return round(sv[max(0, int(len(sv) * p / 100) - 1)], 1)


def _text_from_pdf(pdf_bytes: bytes) -> tuple[str, float]:
    """Return (text, elapsed_ms) using pdfplumber."""
    import pdfplumber, io
    t0 = time.perf_counter()
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            text = "\n".join(p.extract_text() or "" for p in doc.pages[:10])
    except Exception:
        text = ""
    return text, round((time.perf_counter() - t0) * 1000, 1)


def _rule_extract(text: str, extract_type: str, symbol: str, url: str) -> tuple[dict, float]:
    """Return (result_dict, elapsed_ms) using rule extractor."""
    from financial.rule_extractor import try_rule_extract
    t0 = time.perf_counter()
    try:
        result = try_rule_extract(extract_type, text, symbol, "", "", url) or {}
        if hasattr(result, "__dict__"):
            result = vars(result)
    except Exception:
        result = {}
    return result, round((time.perf_counter() - t0) * 1000, 1)


def _ollama_extract(text: str, extract_type: str, symbol: str) -> tuple[dict, float]:
    """Return (result_dict, elapsed_ms) using LocalExtractor (Ollama)."""
    from financial.local_extractor import LocalExtractor
    extractor = LocalExtractor()
    t0 = time.perf_counter()
    try:
        result = extractor.extract(text, extract_type, symbol=symbol) or {}
        if hasattr(result, "__dict__"):
            result = result.__dict__
    except Exception as e:
        result = {"error": str(e)}
    return result, round((time.perf_counter() - t0) * 1000, 1)


def _sample_pdfs(symbol: str | None, n: int) -> list[dict]:
    """
    Return up to n rows: {symbol, url, cache_path, extract_type}
    sampled from announcements that have a cached PDF.
    """
    import hashlib
    from screener.filter_config import FilterConfig

    pdf_subjects = FilterConfig().pdf_subjects
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """
        SELECT symbol, attachment, subject FROM announcements
        WHERE attachment LIKE '%.pdf'
          AND subject IN ({})
        {}
        ORDER BY RANDOM() LIMIT ?
        """.format(
            ",".join("?" * len(pdf_subjects)),
            "AND symbol = ?" if symbol else "",
        ),
        list(pdf_subjects.keys()) + ([symbol] if symbol else []) + [n * 5],
    ).fetchall()
    conn.close()

    result = []
    seen = set()
    for sym, url, subj in rows:
        if url in seen:
            continue
        cache_path = PDF_CACHE / (hashlib.md5(url.encode()).hexdigest() + ".pdf")
        if not cache_path.exists():
            continue
        if cache_path.stat().st_size < MIN_BYTES:
            continue
        seen.add(url)
        result.append({
            "symbol":       sym,
            "url":          url,
            "cache_path":   cache_path,
            "extract_type": pdf_subjects.get(subj, "order_win"),
        })
        if len(result) >= n:
            break
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",         type=int, default=20,  help="Number of PDFs to sample")
    ap.add_argument("--no-ollama", action="store_true",   help="Skip Ollama layer")
    ap.add_argument("--symbol",    default=None,          help="Restrict to one symbol")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    samples = _sample_pdfs(args.symbol, args.n)
    if not samples:
        print("No cached PDFs found. Run extraction first.")
        return

    print(f"Sampled {len(samples)} PDFs from cache\n")

    pdf_times:    list[float] = []
    rule_times:   list[float] = []
    ollama_times: list[float] = []
    rule_hits:    int = 0
    ollama_hits:  int = 0

    hdr = f"{'Symbol':<12} {'Type':<18} {'pdf_ms':>7}  {'rule_ms':>7}  {'rule_hit':>8}"
    if not args.no_ollama:
        hdr += f"  {'ollama_ms':>9}  {'olm_hit':>7}"
    print(hdr)
    print("-" * len(hdr))

    for item in samples:
        pdf_bytes = item["cache_path"].read_bytes()
        sym       = item["symbol"]
        etype     = item["extract_type"]

        text, t_pdf = _text_from_pdf(pdf_bytes)
        pdf_times.append(t_pdf)

        rule_res, t_rule = _rule_extract(text, etype, sym, item["url"])
        rule_times.append(t_rule)
        r_hit = bool(rule_res.get("revenue_cr") or rule_res.get("pat_cr") or rule_res.get("order_value_cr"))
        if r_hit:
            rule_hits += 1

        row = f"{sym:<12} {etype:<18} {t_pdf:>7.0f}  {t_rule:>7.0f}  {'YES' if r_hit else 'no':>8}"

        if not args.no_ollama:
            olm_res, t_olm = _ollama_extract(text, etype, sym)
            ollama_times.append(t_olm)
            o_hit = bool(
                olm_res.get("revenue_cr") or olm_res.get("pat_cr") or
                olm_res.get("order_value_cr") or olm_res.get("raw_summary")
            )
            if o_hit:
                ollama_hits += 1
            row += f"  {t_olm:>9.0f}  {'YES' if o_hit else 'no':>7}"

        print(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(samples)
    print()
    print("=" * 65)
    print(f"  PDF EXTRACTION LATENCY  ({n} PDFs sampled)")
    print("=" * 65)
    print(f"  {'Layer':<14}  {'p50 ms':>8}  {'p95 ms':>8}  {'max ms':>8}  {'hit rate':>9}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    print(f"  {'pdfplumber':<14}  {_pct(pdf_times,50):>8.1f}  {_pct(pdf_times,95):>8.1f}  {max(pdf_times):>8.1f}  {'n/a':>9}")
    print(f"  {'rule':<14}  {_pct(rule_times,50):>8.1f}  {_pct(rule_times,95):>8.1f}  {max(rule_times):>8.1f}  {rule_hits/n*100:>8.0f}%")
    if not args.no_ollama and ollama_times:
        print(f"  {'ollama':<14}  {_pct(ollama_times,50):>8.1f}  {_pct(ollama_times,95):>8.1f}  {max(ollama_times):>8.1f}  {ollama_hits/n*100:>8.0f}%")

    print("=" * 65)
    print(f"\n  Note: 'hit rate' = fraction of PDFs where layer returned revenue/PAT/order data")


if __name__ == "__main__":
    main()
