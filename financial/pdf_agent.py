"""
PDFAgent — orchestrates the full PDF extraction pipeline.

Improvements:
1. PDF cache (data/pdfs/) — avoids re-downloading on restart
2. Phase-1 queue persistence (data/phase1_queue.json) — Ollama queue survives restarts
3. Pipelined Phase 1 → Phase 2 — Ollama workers start before Phase 1 finishes
4. OLLAMA_SUMMARY_MODEL — faster model for summary_only (set in local_extractor)
5. 4 Ollama workers — better GPU utilisation (was 2)
"""
from __future__ import annotations

import hashlib
import json
import queue as _queue
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from financial.local_extractor import LocalExtractor
from financial.models import ExtractionStatus, FinancialResult
from financial.pdf_downloader import PDFDownloader
from financial.statement_extractor import StatementStorage, extract_full_statement
from financial.storage import FinancialStorage
from screener.filter_config import FilterConfig


# ── Run summary ───────────────────────────────────────────────────────────────

@dataclass
class RunSummary:
    total_candidates: int = 0
    extracted:        int = 0
    skipped_done:     int = 0
    failed_download:  int = 0
    failed_extract:   int = 0
    not_pdf:          int = 0
    queued_ollama:    int = 0
    results:          list[FinancialResult] = field(default_factory=list)

    def print(self) -> None:
        print(f"\n{'=' * 68}")
        print(f"  PDF EXTRACTION SUMMARY")
        print(f"{'=' * 68}")
        print(f"  Candidates found      : {self.total_candidates}")
        print(f"  Successfully extracted: {self.extracted}")
        print(f"  Already in DB (skip)  : {self.skipped_done}")
        print(f"  Queued for Ollama     : {self.queued_ollama}")
        print(f"  Download failed       : {self.failed_download}")
        print(f"  Extraction failed     : {self.failed_extract}")
        print(f"  Not a PDF (skipped)   : {self.not_pdf}")
        print(f"{'=' * 68}\n")

        if self.results:
            print("  Extracted Results:")
            for r in self.results:
                dtype = r.period_type.replace("_", " ").title()
                if r.revenue_cr or r.pat_cr:
                    rev = f"Rev Rs.{r.revenue_cr:.0f}cr" if r.revenue_cr else ""
                    pat = f"PAT Rs.{r.pat_cr:.0f}cr"     if r.pat_cr    else ""
                    print(f"  {r.symbol:<12} [{r.period:<14}]  {rev}  {pat}")
                else:
                    hl = r.key_highlights[0] if r.key_highlights else r.raw_summary[:80]
                    print(f"  {r.symbol:<12} [{dtype:<18}]  {hl}")
            print()


# ── PDFAgent ──────────────────────────────────────────────────────────────────

class PDFAgent:
    _WORKERS = 32
    _OLLAMA_SKIP    = {"press_release", "concall"}
    _DOWNLOAD_SKIP  = {"press_release", "concall"}
    _SUMMARY_TYPES  = {"summary_only"}
    _OLLAMA_WORKERS = 4  # was 2 — improvement 5

    def __init__(self, db_path: Path | str, chroma_path: Path | str) -> None:
        self._db_path       = Path(db_path)
        self._chroma_path   = Path(chroma_path)
        self._downloader    = PDFDownloader()
        self._extractor     = LocalExtractor()
        self._fin_storage   = FinancialStorage(db_path, chroma_path)
        self._stmt_storage  = StatementStorage(db_path)
        self._pdf_subjects  = FilterConfig().pdf_subjects
        # Improvement 1: local PDF cache
        self._pdf_cache_dir = self._db_path.parent / "pdfs"
        self._pdf_cache_dir.mkdir(exist_ok=True)
        # Improvement 2: Phase 1 queue persistence
        self._phase1_queue_file = self._db_path.parent / "phase1_queue.json"

    # ── PDF cache (improvement 1) ─────────────────────────────────────────────

    def _cache_path(self, url: str) -> Path:
        return self._pdf_cache_dir / (hashlib.md5(url.encode()).hexdigest() + ".pdf")

    def _cached_pdf(self, url: str) -> bytes | None:
        p = self._cache_path(url)
        return p.read_bytes() if p.exists() else None

    def _save_to_cache(self, url: str, data: bytes) -> None:
        try:
            self._cache_path(url).write_bytes(data)
        except Exception:
            pass

    # ── Phase 1 queue persistence (improvement 2) ─────────────────────────────

    def _load_phase1_queue(self) -> dict[str, str]:
        if self._phase1_queue_file.exists():
            try:
                return json.loads(self._phase1_queue_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_phase1_queue(self, q: dict[str, str]) -> None:
        try:
            self._phase1_queue_file.write_text(json.dumps(q), encoding="utf-8")
        except Exception:
            pass

    def _add_to_phase1_queue(self, url: str, extract_type: str) -> None:
        q = self._load_phase1_queue()
        q[url] = extract_type
        self._save_phase1_queue(q)

    def _remove_from_phase1_queue(self, url: str) -> None:
        q = self._load_phase1_queue()
        if url in q:
            del q[url]
            self._save_phase1_queue(q)

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        limit:     int | None = None,
        symbol:    str | None = None,
        date:      str | None = None,
        subject:   str | None = None,
        reextract: bool = False,
        workers:   int | None = None,
    ) -> RunSummary:
        n_workers = workers or self._WORKERS

        candidates = self._fetch_candidates(symbol=symbol, date=date, subject=subject, limit=limit)
        summary = RunSummary(total_candidates=len(candidates))

        skipped_type = 0
        to_process: list[dict] = []
        for row in candidates:
            url  = row["attachment"] or ""
            sym  = row["symbol"]
            subj = row["subject"]
            extract_type = self._pdf_subjects.get(subj, "order_win")

            if not self._downloader.is_pdf_url(url):
                summary.not_pdf += 1
            elif extract_type in self._DOWNLOAD_SKIP:
                skipped_type += 1
            elif not reextract and self._already_stored(sym, url):
                summary.skipped_done += 1
            else:
                to_process.append(row)

        if summary.skipped_done:
            print(f"  Skipped (done): {summary.skipped_done}")
        if skipped_type:
            print(f"  Skipped (no financial data): {skipped_type}")
        if summary.not_pdf:
            print(f"  Skipped (not PDF): {summary.not_pdf}")

        if not to_process:
            self._fin_storage.close()
            summary.print()
            return summary

        # ── Improvement 2: load persisted Phase 1 → Ollama queue ──────────────
        persisted_queue: dict[str, str] = {} if reextract else self._load_phase1_queue()
        # Drop entries already saved to financial_results (Phase 2 completed for them)
        persisted_queue = {
            url: et for url, et in persisted_queue.items()
            if not self._already_stored("", url)
        }

        url_to_row = {(row["attachment"] or ""): row for row in to_process}
        persisted_urls = set(persisted_queue.keys())

        resume_from_cache: list[tuple[dict, str]] = []  # (row, extract_type)
        still_need_phase1: list[dict] = []

        for url, et in persisted_queue.items():
            row = url_to_row.get(url)
            if row is None:
                continue
            if self._cached_pdf(url) is not None:
                resume_from_cache.append((row, et))
            else:
                still_need_phase1.append(row)  # persisted but cache miss → re-download

        for row in to_process:
            if (row["attachment"] or "") not in persisted_urls:
                still_need_phase1.append(row)

        if resume_from_cache:
            print(f"  Resuming {len(resume_from_cache)} PDFs from local cache (skipping re-download)")

        # ── Thread-safe shared state ───────────────────────────────────────────
        _lock          = threading.Lock()
        pending_saves: list[tuple[dict, FinancialResult, bytes | None]] = []
        ollama_total   = [len(resume_from_cache)]  # grows as Phase 1 queues items
        ollama_done    = [0]

        # Improvement 3: pipelined queue — Phase 2 workers start NOW
        ollama_q: _queue.Queue = _queue.Queue()

        def _phase2_worker() -> None:
            while True:
                item = ollama_q.get()
                if item is None:
                    ollama_q.task_done()
                    break
                row, text, extract_type, pdf_bytes = item
                sym = row["symbol"]
                result = self._extractor.ollama_extract(
                    text            = text,
                    symbol          = sym,
                    company         = row["company"],
                    broadcast_dt    = row["broadcast_dt"] or "",
                    source_url      = row["attachment"] or "",
                    extraction_type = extract_type,
                    pdf_bytes       = pdf_bytes,
                )
                url = row["attachment"] or ""
                self._remove_from_phase1_queue(url)
                with _lock:
                    ollama_done[0] += 1
                    idx   = ollama_done[0]
                    total = ollama_total[0]
                    if result is None:
                        print(f"  [{idx}/{total}] {sym}  FAIL", flush=True)
                        summary.failed_extract += 1
                    else:
                        rev = f"Rev {result.revenue_cr:.0f}cr" if result.revenue_cr else ""
                        pat = f"PAT {result.pat_cr:.0f}cr"     if result.pat_cr    else ""
                        print(f"  [{idx}/{total}] {sym}  OK  [{result.period}]  {rev}  {pat}", flush=True)
                        pending_saves.append((row, result, pdf_bytes))
                        summary.extracted += 1
                        summary.results.append(result)
                ollama_q.task_done()

        phase2_threads = [
            threading.Thread(target=_phase2_worker, daemon=True, name=f"ollama-{i}")
            for i in range(self._OLLAMA_WORKERS)
        ]
        for t in phase2_threads:
            t.start()

        # ── Feed resumed cached items straight to Phase 2 ─────────────────────
        if resume_from_cache:
            print(f"\nPhase 2 — resuming {len(resume_from_cache)} cached PDFs ({self._OLLAMA_WORKERS} workers)...")
            for row, extract_type in resume_from_cache:
                url       = row["attachment"] or ""
                pdf_bytes = self._cached_pdf(url)
                if pdf_bytes is None:
                    continue
                try:
                    result, text = self._extractor.fast_extract(
                        pdf_bytes       = pdf_bytes,
                        symbol          = row["symbol"],
                        company         = row["company"],
                        broadcast_dt    = row["broadcast_dt"] or "",
                        source_url      = url,
                        extraction_type = extract_type,
                    )
                except Exception:
                    result, text = None, ""
                if result is not None:
                    with _lock:
                        pending_saves.append((row, result, pdf_bytes if extract_type == "financial_results" else None))
                        summary.extracted += 1
                        summary.results.append(result)
                    self._remove_from_phase1_queue(url)
                elif text:
                    bytes_for_stmt = pdf_bytes if extract_type == "financial_results" else None
                    ollama_q.put((row, text, extract_type, bytes_for_stmt))
                    with _lock:
                        summary.queued_ollama += 1

        # ── Phase 1: parallel download + fast extract ──────────────────────────
        if still_need_phase1:
            print(f"\nPhase 1 — download + fast extract: {len(still_need_phase1)} PDFs ({n_workers} workers)...")
            completed = 0

            def _download_and_fast_extract(row: dict):
                url  = row["attachment"] or ""
                sym  = row["symbol"]
                co   = row["company"]
                dt   = row["broadcast_dt"] or ""
                subj = row["subject"]
                extract_type = self._pdf_subjects.get(subj, "order_win")

                # Improvement 1: hit cache before network
                pdf_bytes = self._cached_pdf(url)
                if pdf_bytes is None:
                    pdf_bytes, dl_status = self._downloader.download(url)
                    if pdf_bytes is None:
                        return row, None, None, None, f"dl_fail:{dl_status}"
                    self._save_to_cache(url, pdf_bytes)

                try:
                    result, text = self._extractor.fast_extract(
                        pdf_bytes       = pdf_bytes,
                        symbol          = sym,
                        company         = co,
                        broadcast_dt    = dt,
                        source_url      = url,
                        extraction_type = extract_type,
                    )
                except Exception as exc:
                    return row, None, None, None, f"extract_err:{exc}"
                kb = len(pdf_bytes) // 1024
                bytes_for_stmt = pdf_bytes if extract_type == "financial_results" else None
                return row, result, text, bytes_for_stmt, f"ok:{kb}kb"

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_download_and_fast_extract, row): row for row in still_need_phase1}
                for future in as_completed(futures):
                    row, result, text, pdf_bytes, status = future.result()
                    sym  = row["symbol"]
                    subj = row["subject"]
                    url  = row["attachment"] or ""
                    with _lock:
                        completed += 1
                        idx = completed

                    if status.startswith("dl_fail"):
                        print(f"  [{idx}/{len(still_need_phase1)}] DL FAIL  {sym}", flush=True)
                        with _lock:
                            summary.failed_download += 1
                    elif result is not None:
                        rev = f"Rev {result.revenue_cr:.0f}cr" if result.revenue_cr else ""
                        pat = f"PAT {result.pat_cr:.0f}cr"     if result.pat_cr    else ""
                        print(f"  [{idx}/{len(still_need_phase1)}] FAST  {sym}  [{result.period}]  {rev}  {pat}", flush=True)
                        with _lock:
                            pending_saves.append((row, result, pdf_bytes))
                            summary.extracted += 1
                            summary.results.append(result)
                    elif text is not None:
                        extract_type = self._pdf_subjects.get(subj, "order_win")
                        if extract_type in self._OLLAMA_SKIP:
                            print(f"  [{idx}/{len(still_need_phase1)}] SKIP  {sym}", flush=True)
                            with _lock:
                                summary.failed_extract += 1
                        else:
                            print(f"  [{idx}/{len(still_need_phase1)}] QUEUE  {sym}  -> Ollama ({extract_type})", flush=True)
                            # Improvement 2: persist queue entry so restart can resume
                            self._add_to_phase1_queue(url, extract_type)
                            bytes_for_stmt = pdf_bytes if extract_type == "financial_results" else None
                            # Improvement 3: feed directly to live Phase 2 workers
                            ollama_q.put((row, text, extract_type, bytes_for_stmt))
                            with _lock:
                                summary.queued_ollama += 1
                                ollama_total[0] += 1
                    else:
                        kb_info = status.split(":")[1] if ":" in status else ""
                        print(f"  [{idx}/{len(still_need_phase1)}] FAIL  {sym}  {kb_info}", flush=True)
                        with _lock:
                            summary.failed_extract += 1

        # Signal Phase 2 workers to stop and wait
        for _ in range(self._OLLAMA_WORKERS):
            ollama_q.put(None)
        if ollama_total[0] > 0:
            print(f"\nPhase 2 — Ollama: {ollama_total[0]} PDFs ({self._OLLAMA_WORKERS} workers) — waiting for completion...")
        for t in phase2_threads:
            t.join()

        # ── Serial DB + ChromaDB writes ────────────────────────────────────────
        print(f"\nSaving {len(pending_saves)} results...")
        for _row, result, pdf_bytes in pending_saves:
            if reextract:
                self._fin_storage.save_or_update(result)
            else:
                self._fin_storage.save(result)
            if pdf_bytes is not None:
                try:
                    stmt = extract_full_statement(
                        pdf_bytes    = pdf_bytes,
                        symbol       = result.symbol,
                        company      = result.company or "",
                        broadcast_dt = result.broadcast_dt or "",
                        source_url   = result.source_url or "",
                    )
                    if stmt and len(stmt.line_items) >= 3:
                        self._stmt_storage.save(stmt)
                except Exception:
                    pass

        # Clean up phase1_queue if empty
        remaining = self._load_phase1_queue()
        if not remaining:
            self._phase1_queue_file.unlink(missing_ok=True)

        self._fin_storage.close()
        self._stmt_storage.close()
        self._fin_storage  = FinancialStorage(self._db_path, self._chroma_path)
        self._stmt_storage = StatementStorage(self._db_path)
        summary.print()
        return summary

    # ── private ───────────────────────────────────────────────────────────────

    def _fetch_candidates(
        self,
        symbol:  str | None,
        date:    str | None,
        subject: str | None,
        limit:   int | None,
    ) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        subjects = list(self._pdf_subjects.keys())
        placeholders = ",".join("?" * len(subjects))
        params: list[Any] = subjects[:]
        clauses = [f"subject IN ({placeholders})", "attachment LIKE '%.pdf'"]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if date:
            clauses.append("broadcast_dt LIKE ?")
            params.append(f"{date}%")
        if subject:
            matched = [s for s, t in self._pdf_subjects.items()
                       if t == subject or s == subject]
            if matched:
                ph = ",".join("?" * len(matched))
                clauses.append(f"subject IN ({ph})")
                params.extend(matched)
        where = " AND ".join(clauses)
        lim   = f" LIMIT {limit}" if limit else ""
        sql = (
            f"SELECT symbol, company, subject, attachment, broadcast_dt "
            f"FROM announcements WHERE {where} "
            f"ORDER BY broadcast_dt DESC{lim}"
        )
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        conn.close()
        return rows

    def _already_stored(self, symbol: str, source_url: str) -> bool:
        if symbol:
            row = self._fin_storage._conn.execute(
                "SELECT 1 FROM financial_results WHERE symbol=? AND source_url=?",
                (symbol, source_url),
            ).fetchone()
        else:
            row = self._fin_storage._conn.execute(
                "SELECT 1 FROM financial_results WHERE source_url=?",
                (source_url,),
            ).fetchone()
        return row is not None
