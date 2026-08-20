"""
PDFAgent — orchestrates the full PDF extraction pipeline.

Steps per announcement
----------------------
1. Read `announcements` SQL table, filter by subjects listed in FilterConfig.pdf_subjects
   AND attachment is a .pdf URL.
2. Skip any symbol+URL already present in financial_results.
3. Download the PDF via PDFDownloader.
4. Route to the correct extractor based on extraction_type:
   - "financial_results" → FinancialExtractor (Claude Opus, full P&L schema)
   - all others         → GeneralExtractor   (Claude Haiku, key facts + summary)
5. Persist FinancialResult to FinancialStorage (SQL + ChromaDB).
6. Print live progress and return a run summary.

Extraction types and their subjects
------------------------------------
  financial_results  — Outcome of Board Meeting
  order_win          — Bagging/Receiving of orders/contracts, Awarding of order(s)/contract(s)
  acquisition        — Acquisition
  restructuring      — Amalgamation/Merger, Demerger, Scheme of Arrangement, Scheme of Amalgamation - NCLT Order
  fundraising        — QIP, Rights Issue, Fund Raising - Preferential Issue
  buyback            — Buyback
  open_offer         — Public Announcement-Open Offer
  credit_rating      — Credit Rating- Revision, Credit Rating- New
  cirp               — Corporate Insolvency Resolution Process, Resolution Plan (for CIRP companies)
"""
from __future__ import annotations

import sqlite3
import time
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
    queued_ollama:    int = 0   # PDFs that needed Phase 2 Ollama
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
    """
    Reads subject-matched announcements from SQL, downloads each PDF,
    routes to the correct extractor, and persists the result.
    """

    _WORKERS = 32    # parallel download+extract workers (I/O-bound, safe to double)
    # Types where fast-extract almost always succeeds; skip Ollama if fast fails
    _OLLAMA_SKIP = {"press_release", "concall"}
    # Types to skip entirely — don't even download, no financial data in them
    _DOWNLOAD_SKIP = {"press_release", "concall"}
    # summary_only: download + summarize, never run financial JSON schema
    _SUMMARY_TYPES = {"summary_only"}
    # Parallel Ollama workers — 2 is safe on a single-GPU machine (one runs,
    # one waits in Ollama's queue; overlap hides I/O latency without OOM risk)
    _OLLAMA_WORKERS = 2

    def __init__(
        self,
        db_path:    Path | str,
        chroma_path: Path | str,
    ) -> None:
        self._db_path       = Path(db_path)
        self._chroma_path   = Path(chroma_path)
        self._downloader    = PDFDownloader()
        self._extractor     = LocalExtractor()
        self._fin_storage   = FinancialStorage(db_path, chroma_path)
        self._stmt_storage  = StatementStorage(db_path)
        self._pdf_subjects  = FilterConfig().pdf_subjects

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        limit:   int | None = None,
        symbol:  str | None = None,
        date:    str | None = None,
        subject: str | None = None,
        reextract: bool = False,
        workers: int | None = None,
    ) -> RunSummary:
        """
        Run the PDF extraction pipeline and return a RunSummary.

        Parameters
        ----------
        limit     : process at most this many PDFs (None = all)
        symbol    : restrict to one stock symbol
        date      : restrict to broadcast_dt starting with this date string
        subject   : restrict to one extraction_type (e.g. 'order_win') or exact subject name
        reextract : re-process PDFs already in DB and overwrite their records
        workers   : parallel workers for download+extract (default: _WORKERS)
        """
        n_workers = workers or self._WORKERS

        candidates = self._fetch_candidates(
            symbol=symbol, date=date, subject=subject, limit=limit
        )
        summary = RunSummary(total_candidates=len(candidates))

        # ── Filter: skip non-PDF, already-stored, and no-value types ────────────
        skipped_type = 0
        to_process = []
        for row in candidates:
            url  = row["attachment"] or ""
            sym  = row["symbol"]
            subj = row["subject"]
            extract_type = self._pdf_subjects.get(subj, "order_win")

            if not self._downloader.is_pdf_url(url):
                summary.not_pdf += 1
            elif extract_type in self._DOWNLOAD_SKIP:
                skipped_type += 1   # no financial data — skip download entirely
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

        # ── Phase 1: parallel download + fast extraction (no Ollama) ─────────
        # Downloads and table/regex extraction run across n_workers threads.
        # Ollama is deliberately excluded — single-GPU inference can't run in
        # parallel anyway, so blocking worker threads on it wastes concurrency.
        print(f"\nPhase 1 — download + fast extract: {len(to_process)} PDFs ({n_workers} workers)...")

        completed = 0
        pending_saves: list[tuple[dict, FinancialResult, bytes | None]] = []
        # (row, extracted_text, extraction_type, pdf_bytes_for_stmt)
        needs_ollama: list[tuple[dict, str, str, bytes | None]] = []

        def _download_and_fast_extract(
            row: dict,
        ) -> tuple[dict, "FinancialResult | None", "str | None", "bytes | None", str]:
            url  = row["attachment"] or ""
            sym  = row["symbol"]
            co   = row["company"]
            dt   = row["broadcast_dt"] or ""
            subj = row["subject"]
            extract_type = self._pdf_subjects.get(subj, "order_win")

            pdf_bytes, dl_status = self._downloader.download(url)
            if pdf_bytes is None:
                return row, None, None, None, f"dl_fail:{dl_status}"

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
            # Keep raw bytes only for financial_results — needed for statement extraction
            bytes_for_stmt = pdf_bytes if extract_type == "financial_results" else None
            return row, result, text, bytes_for_stmt, f"ok:{kb}kb"

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_download_and_fast_extract, row): row for row in to_process}
            for future in as_completed(futures):
                completed += 1
                row, result, text, pdf_bytes, status = future.result()
                sym  = row["symbol"]
                subj = row["subject"]

                if status.startswith("dl_fail"):
                    print(f"  [{completed}/{len(to_process)}] DL FAIL  {sym}", flush=True)
                    summary.failed_download += 1
                elif result is not None:
                    rev = f"Rev {result.revenue_cr:.0f}cr" if result.revenue_cr else ""
                    pat = f"PAT {result.pat_cr:.0f}cr"     if result.pat_cr    else ""
                    print(f"  [{completed}/{len(to_process)}] FAST  {sym}  [{result.period}]  {rev}  {pat}", flush=True)
                    pending_saves.append((row, result, pdf_bytes))
                    summary.extracted += 1
                    summary.results.append(result)
                elif text is not None:
                    extract_type = self._pdf_subjects.get(subj, "order_win")
                    if extract_type in self._OLLAMA_SKIP:
                        print(f"  [{completed}/{len(to_process)}] SKIP  {sym}  (no fast-extract for {extract_type})", flush=True)
                        summary.failed_extract += 1
                    else:
                        print(f"  [{completed}/{len(to_process)}] QUEUE  {sym}  -> Ollama ({extract_type})", flush=True)
                        needs_ollama.append((row, text, extract_type, pdf_bytes))
                        summary.queued_ollama += 1
                else:
                    kb_info = status.split(":")[1] if ":" in status else ""
                    print(f"  [{completed}/{len(to_process)}] FAIL  {sym}  {kb_info}", flush=True)
                    summary.failed_extract += 1

        # ── Phase 2: parallel Ollama pass ─────────────────────────────────────
        # _OLLAMA_WORKERS=2 overlaps one GPU request with one waiting in Ollama's
        # queue — safe on 6GB VRAM with llama3:latest (4.7GB model).
        if needs_ollama:
            print(f"\nPhase 2 — Ollama: {len(needs_ollama)} PDFs queued ({self._OLLAMA_WORKERS} workers)...")
            ollama_done = 0
            ollama_lock = __import__("threading").Lock()

            def _run_ollama(
                item: tuple[dict, str, str, "bytes | None"],
            ) -> tuple[dict, "FinancialResult | None", "bytes | None"]:
                row, text, extract_type, pdf_bytes = item
                result = self._extractor.ollama_extract(
                    text            = text,
                    symbol          = row["symbol"],
                    company         = row["company"],
                    broadcast_dt    = row["broadcast_dt"] or "",
                    source_url      = row["attachment"] or "",
                    extraction_type = extract_type,
                    pdf_bytes       = pdf_bytes,
                )
                return row, result, pdf_bytes

            with ThreadPoolExecutor(max_workers=self._OLLAMA_WORKERS) as pool:
                futures = {pool.submit(_run_ollama, item): item for item in needs_ollama}
                for future in as_completed(futures):
                    row, result, pdf_bytes = future.result()
                    sym = row["symbol"]
                    with ollama_lock:
                        ollama_done += 1
                        idx = ollama_done
                    if result is None:
                        print(f"  [{idx}/{len(needs_ollama)}] {sym}  FAIL", flush=True)
                        summary.failed_extract += 1
                    else:
                        rev = f"Rev {result.revenue_cr:.0f}cr" if result.revenue_cr else ""
                        pat = f"PAT {result.pat_cr:.0f}cr"     if result.pat_cr    else ""
                        print(f"  [{idx}/{len(needs_ollama)}] {sym}  OK  [{result.period}]  {rev}  {pat}", flush=True)
                        pending_saves.append((row, result, pdf_bytes))
                        summary.extracted += 1
                        summary.results.append(result)

        # ── Serial DB + ChromaDB writes (fast — not the bottleneck) ───────────
        print(f"\nSaving {len(pending_saves)} results...")
        for _row, result, pdf_bytes in pending_saves:
            if reextract:
                self._fin_storage.save_or_update(result)
            else:
                self._fin_storage.save(result)
            # Also extract the full P&L table for the Compare tab
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

        self._fin_storage.close()
        self._stmt_storage.close()
        # Reopen storage so this agent instance can be called again (run() reopens fresh)
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
        """
        Query announcements table for rows whose subject is in pdf_subjects
        AND have a PDF attachment.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        # Build IN clause from config
        subjects = list(self._pdf_subjects.keys())
        placeholders = ",".join("?" * len(subjects))
        params: list[Any] = subjects[:]

        clauses = [
            f"subject IN ({placeholders})",
            "attachment LIKE '%.pdf'",
        ]

        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())

        if date:
            clauses.append("broadcast_dt LIKE ?")
            params.append(f"{date}%")

        if subject:
            # Allow filtering by extraction_type (e.g. "order_win") or exact subject
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
        """Check if this exact PDF URL was already processed."""
        row = self._fin_storage._conn.execute(
            "SELECT 1 FROM financial_results WHERE symbol=? AND source_url=?",
            (symbol, source_url),
        ).fetchone()
        return row is not None
