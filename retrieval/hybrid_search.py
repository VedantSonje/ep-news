"""
HybridSearch — fuses BM25 (FTS5) and ChromaDB vector results via
Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = sum_over_lists( 1 / (k + rank_in_list) )
k=60 is the standard default — dampens the impact of very top ranks.

Returns (fused_candidates, bm25_log, vector_log) so callers can log each source.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from retrieval.bm25_search import BM25Search, BM25Hit
from retrieval.pipeline_logger import StageLog, make_stage


_RRF_K = 60


@dataclass
class CandidateDoc:
    id:           int | str
    symbol:       str
    company:      str
    subject:      str
    details:      str
    score:        int
    broadcast_dt: str
    rrf_score:    float = 0.0
    sources:      list[str] = field(default_factory=list)
    # ranks within each retrieval source (0-indexed, None = not retrieved)
    bm25_rank:    int | None = None
    vector_rank:  int | None = None
    # populated by broker._enrich_from_sql + reranker
    evidence: dict = field(default_factory=dict)


def _rrf(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank + 1)   # rank is 0-indexed


class HybridSearch:
    """
    Combines BM25 and semantic (ChromaDB) search results using RRF.
    Either source can return zero results — the fusion degrades gracefully.
    Returns (candidates, bm25_log, vector_log).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        vector_store,
        bm25_weight:   float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        self._bm25   = BM25Search(conn)
        self._vector = vector_store
        self._bw     = bm25_weight
        self._vw     = vector_weight

    def search(
        self,
        query: str,
        limit: int = 20,
        min_score: int | None = None,
        date_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        symbols: list[str] | None = None,
        themes: list[str] | None = None,
    ) -> tuple[list[CandidateDoc], StageLog, StageLog]:
        t0 = time.time()
        bm25_hits = self._bm25.search(
            query,
            limit=limit * 2,
            min_score=min_score,
            date_filter=date_filter,
            symbols=symbols,
            themes=themes,
        )
        bm25_ms = round((time.time() - t0) * 1000, 1)

        t1 = time.time()
        vector_hits = self._vector.search(
            query,
            n_results=limit * 2,
            min_score=min_score,
            date_from=date_from or (date_filter if date_filter else None),
            date_to=date_to,
        )
        vec_ms = round((time.time() - t1) * 1000, 1)

        bm25_log   = make_stage("bm25",   bm25_hits,   elapsed_ms=bm25_ms,
                                score_attr="bm25_rank")
        vector_log = make_stage("vector", vector_hits, elapsed_ms=vec_ms)

        # Map id -> CandidateDoc for merging
        docs: dict[str, CandidateDoc] = {}

        # ── BM25 contribution ─────────────────────────────────────────────
        for rank, hit in enumerate(bm25_hits):
            key = str(hit.id)
            if key not in docs:
                docs[key] = CandidateDoc(
                    id=hit.id, symbol=hit.symbol, company=hit.company,
                    subject=hit.subject, details=hit.details,
                    score=hit.score, broadcast_dt=hit.broadcast_dt,
                    bm25_rank=rank,
                )
            docs[key].rrf_score += self._bw * _rrf(rank)
            if "bm25" not in docs[key].sources:
                docs[key].sources.append("bm25")

        # ── Vector contribution ───────────────────────────────────────────
        for rank, hit in enumerate(vector_hits):
            sym  = hit.get("symbol", "")
            subj = hit.get("subject", "")
            dt   = hit.get("broadcast_dt", "")
            key  = _find_matching_key(docs, sym, dt, subj)
            if key not in docs:
                docs[key] = CandidateDoc(
                    id=key,
                    symbol=sym,
                    company=hit.get("company", ""),
                    subject=subj,
                    details=hit.get("tags", ""),
                    score=int(hit.get("score", 0)),
                    broadcast_dt=dt,
                    vector_rank=rank,
                )
            else:
                if docs[key].vector_rank is None:
                    docs[key].vector_rank = rank
            docs[key].rrf_score += self._vw * _rrf(rank)
            if "vector" not in docs[key].sources:
                docs[key].sources.append("vector")

        # ── Apply post-fusion filters ─────────────────────────────────────
        results = list(docs.values())
        if min_score is not None:
            results = [d for d in results if d.score >= min_score]
        if date_filter:
            results = [d for d in results if d.broadcast_dt.startswith(date_filter)]
        if symbols:
            sym_set = {s.upper() for s in symbols}
            results = [d for d in results if d.symbol.upper() in sym_set]

        results.sort(key=lambda d: d.rrf_score, reverse=True)
        return results[:limit], bm25_log, vector_log


def _find_matching_key(
    docs: dict[str, CandidateDoc],
    sym: str,
    dt: str,
    subj: str,
) -> str:
    dt10 = dt[:10]
    if sym and dt10:
        for key, doc in docs.items():
            if doc.symbol == sym and doc.broadcast_dt.startswith(dt10) and doc.subject == subj:
                return key
    return f"{sym}|{dt10}|{subj[:20]}" if sym else str(hash((sym, dt, subj)))
