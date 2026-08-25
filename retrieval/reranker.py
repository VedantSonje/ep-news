"""
Reranker — two-stage scoring pipeline.

Stage 1 (always):
  Heuristic proxy score:
  1. Keyword overlap  — Jaccard between query tokens and doc tokens
  2. EP score boost   — announcements with score >= 10 get a lift
  3. Recency boost    — announcements from the past 7 days get a lift
  4. Source diversity — docs in both BM25 and vector get a lift
  5. Order value boost — docs with high order_value_cr get a lift for order queries

Stage 2 (optional, when sentence-transformers CrossEncoder is available):
  Loads cross-encoder/ms-marco-MiniLM-L-6-v2 and rescores the proxy top-K
  with (query, document) pair scoring. Falls back to proxy if unavailable.
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime

from retrieval.hybrid_search import CandidateDoc
from retrieval.pipeline_logger import StageLog, make_stage


_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "by", "with", "from", "is", "are", "was", "were",
    "which", "that", "this", "has", "have", "had", "be", "been",
    "its", "it", "as", "if", "so", "but", "not", "no",
})

_RECENT_DAYS = 7
_CE_TOP_K    = 30   # run cross-encoder on at most this many candidates


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]{2,}", text)
            if w.lower() not in _STOP_WORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _days_ago(broadcast_dt: str) -> int:
    try:
        dt = datetime.fromisoformat(broadcast_dt[:10])
        return (date.today() - dt.date()).days
    except (ValueError, AttributeError):
        return 999


# ── Cross-encoder (loaded once at module level) ───────────────────────────────

_ce_model = None
_ce_tried = False


def _get_cross_encoder():
    global _ce_model, _ce_tried
    if _ce_tried:
        return _ce_model
    _ce_tried = True
    try:
        import os as _os
        # Never download from network — only use if already cached locally.
        # On Railway (no cache) this raises immediately; heuristic proxy is used instead.
        _os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        print("[reranker] CrossEncoder loaded (BAAI/bge-reranker-v2-m3)", flush=True)
    except Exception as e:
        print(f"[reranker] CrossEncoder unavailable ({e}), using heuristic proxy", flush=True)
        _ce_model = None
    return _ce_model


class Reranker:
    """
    Re-ranks a list of CandidateDoc objects.
    Uses CrossEncoder when available, heuristic proxy otherwise.
    Returns (ranked_candidates, stage_log).
    """

    def __init__(
        self,
        keyword_weight:      float = 0.35,
        score_weight:        float = 0.25,
        recency_weight:      float = 0.15,
        diversity_weight:    float = 0.10,
        order_value_weight:  float = 0.15,
    ) -> None:
        self._kw   = keyword_weight
        self._sc   = score_weight
        self._re   = recency_weight
        self._div  = diversity_weight
        self._ov   = order_value_weight
        # Cross-encoder loaded lazily on first rerank() — don't block startup

    def rerank(
        self,
        candidates: list[CandidateDoc],
        query: str,
        top_n: int = 10,
    ) -> tuple[list[CandidateDoc], StageLog]:
        if not candidates:
            return [], StageLog(name="reranked", count=0)

        t0 = time.time()
        query_tokens = _tokens(query)

        # ── Stage 1: heuristic proxy ─────────────────────────────────────────
        # Determine whether this is an order-value query
        is_order_query = any(w in query.lower() for w in
                             ["order", "contract", "win", "wins", "bagged", "awarded"])
        # Max order value in this result set for normalisation
        max_ov = max(
            (doc.evidence.get("order_value_cr") or 0.0 for doc in candidates),
            default=0.0,
        )
        if max_ov <= 0:
            max_ov = 1.0

        for doc in candidates:
            base_rrf   = doc.rrf_score
            doc_tokens = _tokens(f"{doc.subject} {doc.details} {doc.symbol} {doc.company}")

            kw_boost  = self._kw  * _jaccard(query_tokens, doc_tokens)
            sc_boost  = self._sc  * min(doc.score / 13.0, 1.0)
            age       = _days_ago(doc.broadcast_dt)
            re_boost  = self._re  * max(0.0, 1.0 - age / _RECENT_DAYS)
            div_boost = self._div if len(doc.sources) > 1 else 0.0

            # Order value boost — scaled to max in this result set
            ov = doc.evidence.get("order_value_cr") or 0.0
            ov_boost = (self._ov * min(ov / max_ov, 1.0)) if (is_order_query and ov > 0) else 0.0

            doc.rrf_score = base_rrf * (1.0 + kw_boost + sc_boost + re_boost + div_boost + ov_boost)

            doc.evidence.update({
                "base_rrf":         round(base_rrf, 5),
                "proxy_rrf":        round(doc.rrf_score, 5),
                "keyword_boost":    round(kw_boost, 4),
                "score_boost":      round(sc_boost, 4),
                "recency_boost":    round(re_boost, 4),
                "diversity_boost":  round(div_boost, 4),
                "order_val_boost":  round(ov_boost, 4),
                "age_days":         age,
                "bm25_rank":        doc.bm25_rank,
                "vector_rank":      doc.vector_rank,
            })

        candidates.sort(key=lambda d: d.rrf_score, reverse=True)

        # ── Stage 2: cross-encoder re-scoring (top-K candidates only) ────────
        ce = _get_cross_encoder()
        if ce is not None and len(candidates) > 0:
            ce_pool = candidates[:_CE_TOP_K]
            pairs   = [
                (query, f"{doc.subject}. {doc.details[:400]}")
                for doc in ce_pool
            ]
            try:
                ce_scores = ce.predict(pairs)
                for doc, ce_score in zip(ce_pool, ce_scores):
                    doc.rrf_score = float(ce_score)
                    doc.evidence["ce_score"] = round(float(ce_score), 4)
                candidates.sort(key=lambda d: d.rrf_score, reverse=True)
            except Exception as e:
                # Cross-encoder failed mid-run — proxy ranking already applied
                candidates[0].evidence["ce_error"] = str(e)

        ranked = candidates[:top_n]
        elapsed = round((time.time() - t0) * 1000, 1)
        log = make_stage(
            "reranked", ranked, elapsed_ms=elapsed,
            score_attr="rrf_score",
            notes=f"ce={'yes' if ce else 'proxy'} query_tokens={len(query_tokens)}",
        )
        return ranked, log
