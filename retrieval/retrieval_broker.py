"""
RetrievalBroker — routes a ParsedQuery to the correct retrieval strategy,
enriches candidates with SQL metadata, and builds a full PipelineTrace.

Routing table:
  STRUCTURED  -> SQL exact query
  SEMANTIC    -> vector-only
  HYBRID      -> BM25 + vector + RRF
  FINANCIAL   -> financial_results table
  COMPOUND    -> hybrid + financial merged

After retrieval, _enrich_from_sql() loads sector_tags, novelty, materiality,
and government/export flags from SQL for every candidate, populating
CandidateDoc.evidence so the reranker and LLM both have full context.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from retrieval.query_classifier import ParsedQuery, QueryIntent
from retrieval.hybrid_search import CandidateDoc, HybridSearch
from retrieval.reranker import Reranker
from retrieval.pipeline_logger import PipelineTrace, StageLog, make_stage


class RetrievalBroker:

    def __init__(
        self,
        conn: sqlite3.Connection,
        vector_store,
        fin_storage=None,
    ) -> None:
        self._conn    = conn
        self._vector  = vector_store
        self._fin     = fin_storage
        self._hybrid  = HybridSearch(conn, vector_store)
        self._reranker = Reranker()

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, parsed: ParsedQuery, top_n: int = 10) -> list[CandidateDoc]:
        candidates, _ = self.retrieve_with_trace(parsed, top_n)
        return candidates

    def retrieve_with_trace(
        self,
        parsed: ParsedQuery,
        top_n: int = 10,
    ) -> tuple[list[CandidateDoc], PipelineTrace]:
        trace = PipelineTrace(query=parsed.raw)

        # ── Classifier log ────────────────────────────────────────────────
        trace.classifier = StageLog(
            name    = "classifier",
            count   = 1,
            notes   = (
                f"intent={parsed.intent.value}  "
                f"themes={parsed.themes}  "
                f"symbols={parsed.symbols}  "
                f"date={parsed.date_filter}  "
                f"min_score={parsed.min_score}"
            ),
        )

        # ── Retrieval ─────────────────────────────────────────────────────
        intent = parsed.intent
        bm25_log = vector_log = sql_log = None

        t0 = time.time()
        if intent == QueryIntent.STRUCTURED:
            candidates = self._sql_search(parsed)
            sql_log    = make_stage("sql", candidates, elapsed_ms=round((time.time()-t0)*1000,1),
                                    score_attr="rrf_score",
                                    notes=f"filters: symbols={parsed.symbols} date={parsed.date_filter} score>={parsed.min_score}")
        elif intent == QueryIntent.SEMANTIC:
            candidates, _, vector_log = self._hybrid.search(
                parsed.raw, limit=top_n * 4,
                min_score=parsed.min_score,
                date_filter=parsed.date_filter,
                symbols=parsed.symbols or None,
                themes=parsed.themes or None,
            )
            # For pure semantic, only use vector hits
            candidates = [c for c in candidates if "vector" in c.sources]
        elif intent == QueryIntent.FINANCIAL:
            candidates = self._financial_search(parsed)
        elif intent == QueryIntent.COMPOUND:
            ann, bm25_log, vector_log = self._hybrid.search(
                parsed.raw, limit=top_n * 4,
                min_score=parsed.min_score,
                date_filter=parsed.date_filter,
                symbols=parsed.symbols or None,
                themes=parsed.themes or None,
            )
            fin  = self._financial_search(parsed, limit=top_n)
            candidates = _merge(ann, fin)
        else:  # HYBRID (default)
            candidates, bm25_log, vector_log = self._hybrid.search(
                parsed.raw, limit=top_n * 4,
                min_score=parsed.min_score,
                date_filter=parsed.date_filter,
                symbols=parsed.symbols or None,
                themes=parsed.themes or None,
            )

        fused_log = make_stage("fused", candidates, score_attr="rrf_score",
                               notes=f"pre-rerank pool size={len(candidates)}")

        # ── Enrich from SQL ───────────────────────────────────────────────
        self._enrich_from_sql(candidates)

        # ── Rerank ────────────────────────────────────────────────────────
        ranked, reranked_log = self._reranker.rerank(candidates, parsed.raw, top_n=top_n)

        # ── Assemble trace ────────────────────────────────────────────────
        trace.sql      = sql_log
        trace.bm25     = bm25_log
        trace.vector   = vector_log
        trace.fused    = fused_log
        trace.reranked = reranked_log
        trace.stop()
        return ranked, trace

    # ── Enrichment ────────────────────────────────────────────────────────

    def _enrich_from_sql(self, candidates: list[CandidateDoc]) -> None:
        """
        Load enrichment columns from SQL for every candidate and write them
        into doc.evidence.  Missing rows are silently skipped.
        """
        if not candidates:
            return
        symbols = list({c.symbol for c in candidates if c.symbol})
        if not symbols:
            return

        ph = ",".join("?" * len(symbols))
        rows = self._conn.execute(
            f"""SELECT symbol, broadcast_dt, subject,
                       sector_tags, is_novel, relative_size, catalyst_type,
                       government_linked, export_component, order_value_cr
                FROM announcements WHERE symbol IN ({ph})""",
            symbols,
        ).fetchall()

        # Build lookup: (symbol, dt_prefix, subject) -> enrichment row
        lookup: dict[tuple, dict] = {}
        for r in rows:
            key = (r[0], str(r[1])[:10], r[2])
            lookup[key] = {
                "sector_tags":       r[3] or "",
                "is_novel":          bool(r[4]) if r[4] is not None else True,
                "relative_size":     r[5] or "unknown",
                "catalyst_type":     r[6] or "general",
                "government_linked": bool(r[7]),
                "export_component":  bool(r[8]),
                "order_value_cr":    r[9],
            }

        for doc in candidates:
            key = (doc.symbol, doc.broadcast_dt[:10], doc.subject)
            if key in lookup:
                doc.evidence.update(lookup[key])
            else:
                # Candidate may come from financial or vector without exact match
                doc.evidence.setdefault("sector_tags", "")
                doc.evidence.setdefault("is_novel", True)
                doc.evidence.setdefault("relative_size", "unknown")
                doc.evidence.setdefault("catalyst_type", "general")
                doc.evidence.setdefault("government_linked", False)
                doc.evidence.setdefault("export_component", False)
                doc.evidence.setdefault("order_value_cr", None)

    # ── Private strategy methods ──────────────────────────────────────────

    def _sql_search(self, parsed: ParsedQuery, limit: int = 50) -> list[CandidateDoc]:
        where, params = _build_sql_filters(parsed)
        rows = self._conn.execute(
            f"SELECT id, symbol, company, subject, details, score, broadcast_dt "
            f"FROM announcements WHERE {where} "
            f"ORDER BY score DESC, broadcast_dt DESC LIMIT {limit}",
            params,
        ).fetchall()
        return [
            CandidateDoc(
                id=r[0], symbol=r[1], company=r[2], subject=r[3],
                details=r[4] or "", score=r[5], broadcast_dt=r[6] or "",
                rrf_score=r[5] / 13.0, sources=["sql"],
            )
            for r in rows
        ]

    def _financial_search(self, parsed: ParsedQuery, limit: int = 20) -> list[CandidateDoc]:
        if self._fin is None:
            return []
        where_parts = ["1=1"]
        params: list = []
        if parsed.symbols:
            ph = ",".join("?" * len(parsed.symbols))
            where_parts.append(f"symbol IN ({ph})")
            params.extend(parsed.symbols)
        if parsed.date_filter:
            where_parts.append("DATE(broadcast_dt) >= ?")
            params.append(parsed.date_filter)
        sql = (
            f"SELECT symbol, company, period, revenue_cr, pat_cr, pat_growth_pct, "
            f"ebitda_margin_pct, eps, key_highlights, broadcast_dt "
            f"FROM financial_results WHERE {' AND '.join(where_parts)} "
            f"ORDER BY broadcast_dt DESC LIMIT {limit}"
        )
        try:
            rows = self._fin.query(sql)
        except Exception:
            return []
        return [
            CandidateDoc(
                id=f"fin_{r.get('symbol','')}_{r.get('period','')}",
                symbol=r.get("symbol", ""),
                company=r.get("company", ""),
                subject=f"Financial Results {r.get('period','')}",
                details=(
                    f"Rev Rs.{r.get('revenue_cr','')} cr | "
                    f"PAT Rs.{r.get('pat_cr','')} cr ({r.get('pat_growth_pct','')}% YoY) | "
                    f"EBITDA {r.get('ebitda_margin_pct','')}% | EPS {r.get('eps','')}"
                ),
                score=8,
                broadcast_dt=r.get("broadcast_dt", ""),
                rrf_score=0.5,
                sources=["financial"],
            )
            for r in rows
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_sql_filters(parsed: ParsedQuery) -> tuple[str, list]:
    parts  = ["1=1"]
    params: list = []
    if parsed.symbols:
        ph = ",".join("?" * len(parsed.symbols))
        parts.append(f"symbol IN ({ph})")
        params.extend(parsed.symbols)
    if parsed.date_filter:
        parts.append("DATE(broadcast_dt) = ?")
        params.append(parsed.date_filter)
    if parsed.min_score is not None:
        parts.append("score >= ?")
        params.append(parsed.min_score)
    if parsed.themes:
        for theme in parsed.themes:
            parts.append("sector_tags LIKE ?")
            params.append(f"%{theme}%")
    return " AND ".join(parts), params


def _merge(a: list[CandidateDoc], b: list[CandidateDoc]) -> list[CandidateDoc]:
    seen: set = set()
    out:  list[CandidateDoc] = []
    for doc in a + b:
        key = str(doc.id)
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out
