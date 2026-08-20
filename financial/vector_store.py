"""
VectorStore — hybrid BM25 + dense retrieval with ChromaDB.

Architecture
------------
Tier 1  Dense vector search  ChromaDB all-MiniLM-L6-v2 (cosine, HNSW)
Tier 2  BM25 sparse search   rank_bm25 in-memory index, rebuilt at startup / backfill
Fusion  Reciprocal Rank Fusion (k=60)  — no score normalisation needed

Why hybrid?
  Dense alone misses exact ticker symbols (ATHERENERG embedding ≠ ATHERENERG docs)
  BM25 alone misses semantic queries ("good margins" ≠ "ebitda_margin_pct")
  RRF of both covers both failure modes.

Two collections
---------------
ep_announcements  All filtered BSE/NSE announcements.
ep_financials     Extracted financial facts (order wins, quarterly results, …)

Metadata fields
---------------
ep_announcements:
  symbol, company, subject, broadcast_dt, date_int (YYYYMMDD int),
  sector_tags, catalyst_type, order_value_cr, is_novel, has_pdf

ep_financials:
  symbol, company, period, period_type, broadcast_dt, date_int,
  revenue_cr, pat_cr, ebitda_margin_pct, revenue_growth_pct, pat_growth_pct,
  eps, order_value_cr, sector, client_type
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from financial.models import FinancialResult


_ANN_COL  = "ep_announcements"
_FIN_COL  = "ep_financials"
_MISSING  = -1.0   # placeholder for None floats — ChromaDB metadata can't be None
_RRF_K    = 60     # standard RRF constant


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v) -> float:
    if v is None:
        return _MISSING
    try:
        return float(v)
    except (TypeError, ValueError):
        return _MISSING


def _date_to_int(date_str: str) -> int:
    if not date_str or len(date_str) < 10:
        return 0
    try:
        return int(date_str[:10].replace("-", ""))
    except ValueError:
        return 0


def _parse_order_value(highlights: list[str]) -> float:
    for h in highlights:
        if not h:
            continue
        if "order value" in h.lower():
            m = re.search(r"Rs\.?([\d,]+(?:\.\d+)?)\s*Cr", h, re.I)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
    return _MISSING


def _parse_sector(highlights: list[str]) -> str:
    for h in highlights:
        if not h:
            continue
        if h.lower().startswith("sector:"):
            return h.split(":", 1)[1].strip()
    return ""


def _parse_client_type(highlights: list[str]) -> str:
    for h in highlights:
        if not h:
            continue
        if h.lower().startswith("client type:"):
            return h.split(":", 1)[1].strip()
    return ""


def _ann_document(company: str, symbol: str, subject: str, details: str) -> str:
    return f"{company} ({symbol}): {subject}. {details[:600]}"


def _ann_id(symbol: str, dt: str, subject: str, details: str) -> str:
    return hashlib.md5(
        f"{symbol}|{dt}|{subject}|{details}".encode()
    ).hexdigest()


def _fin_id(symbol: str, period: str, period_type: str) -> str:
    return hashlib.md5(
        f"{symbol}|{period}|{period_type}".encode()
    ).hexdigest()


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Tokenise for BM25.
    Splits on whitespace/punctuation, lowercases everything.
    Ticker-like tokens (ATHERENERG) are kept as-is AND added lowercase.
    """
    tokens: list[str] = []
    for tok in re.findall(r'[A-Za-z0-9&]+', text):
        lo = tok.lower()
        tokens.append(lo)
        # Also add raw uppercase so both "RVNL" and "rvnl" match
        if tok != lo:
            tokens.append(tok)
    return tokens


def _eval_where(meta: dict, where: dict) -> bool:
    """
    Evaluate a ChromaDB-style where clause against a metadata dict.
    Supports: $and, $or, $eq, $ne, $gte, $lte, $gt, $lt
    """
    if "$and" in where:
        return all(_eval_where(meta, c) for c in where["$and"])
    if "$or" in where:
        return any(_eval_where(meta, c) for c in where["$or"])
    for field, condition in where.items():
        if field.startswith("$"):
            continue
        val = meta.get(field)
        if isinstance(condition, dict):
            for op, cval in condition.items():
                if op == "$eq"  and val != cval:                return False
                if op == "$ne"  and val == cval:                return False
                if op == "$gte" and (val is None or val < cval):  return False
                if op == "$lte" and (val is None or val > cval):  return False
                if op == "$gt"  and (val is None or val <= cval): return False
                if op == "$lt"  and (val is None or val >= cval): return False
        else:
            if val != condition:
                return False
    return True


class _BM25Index:
    """In-memory BM25 index over one ChromaDB collection."""

    def __init__(self) -> None:
        self._ids:   list[str]  = []
        self._metas: list[dict] = []
        self._bm25  = None

    def build(self, ids: list[str], texts: list[str], metas: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._ids   = ids
        self._metas = metas
        self._bm25  = BM25Okapi([_tokenize(t) for t in texts])

    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(
        self,
        query: str,
        where: dict | None,
        n: int,
    ) -> list[tuple[str, float]]:
        """Return (id, score) pairs sorted by BM25 score descending."""
        if not self._bm25 or not self._ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Filter + collect
        hits = [
            (self._ids[i], float(scores[i]))
            for i in range(len(self._ids))
            if scores[i] > 0.0
            and (where is None or _eval_where(self._metas[i], where))
        ]
        hits.sort(key=lambda x: -x[1])
        return hits[:n]


def _rrf_fuse(
    bm25_hits:  list[tuple[str, float]],   # (id, score) sorted desc
    dense_ids:  list[str],                  # sorted by dense rank
    n:          int,
    k:          int = _RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    Returns [(id, rrf_score)] sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    for rank, (id_, _) in enumerate(bm25_hits):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
    for rank, id_ in enumerate(dense_ids):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:n]


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Dual-collection ChromaDB interface with BM25 hybrid search.
    BM25 index is built at startup from all stored docs and rebuilt after backfill.
    """

    def __init__(self, chroma_path: Path | str) -> None:
        import os
        import chromadb

        # Force ONNX embeddings to CPU to avoid competing with Ollama for VRAM.
        # Set ONNX_DEVICE=cuda in .env to use GPU embeddings instead.
        onnx_device = os.getenv("ONNX_DEVICE", "cpu")
        os.environ.setdefault("ONNX_DEVICE", onnx_device)

        path = Path(chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._ann = self._client.get_or_create_collection(
            name=_ANN_COL,
            metadata={"hnsw:space": "cosine"},
        )
        self._fin = self._client.get_or_create_collection(
            name=_FIN_COL,
            metadata={"hnsw:space": "cosine"},
        )
        self._ann_bm25 = _BM25Index()
        self._fin_bm25 = _BM25Index()
        self._build_bm25_indexes()

    # ── BM25 index management ─────────────────────────────────────────────────

    def _build_bm25_indexes(self) -> None:
        """Load all docs from ChromaDB and build BM25 indexes. Called at startup."""
        if self._ann.count() > 0:
            data = self._ann.get(include=["documents", "metadatas"])
            self._ann_bm25.build(data["ids"], data["documents"], data["metadatas"])
        if self._fin.count() > 0:
            data = self._fin.get(include=["documents", "metadatas"])
            self._fin_bm25.build(data["ids"], data["documents"], data["metadatas"])

    def rebuild_bm25(self) -> None:
        """Explicit rebuild — call after bulk upserts (e.g. post-backfill)."""
        self._build_bm25_indexes()

    # ── Save single item ──────────────────────────────────────────────────────

    def save_announcement(self, row: dict) -> None:
        symbol  = str(row.get("symbol") or "")
        company = str(row.get("company") or "")
        subject = str(row.get("subject") or "")
        details = str(row.get("details") or "")
        dt      = str(row.get("broadcast_dt") or "")
        self._ann.upsert(
            ids       = [_ann_id(symbol, dt, subject, details)],
            documents = [_ann_document(company, symbol, subject, details)],
            metadatas = [{
                "symbol":         symbol,
                "company":        company,
                "subject":        subject,
                "broadcast_dt":   dt[:16],
                "date_int":       _date_to_int(dt),
                "sector_tags":    str(row.get("sector_tags") or ""),
                "catalyst_type":  str(row.get("catalyst_type") or "general"),
                "order_value_cr": _f(row.get("order_value_cr")),
                "is_novel":       int(row.get("is_novel") or 0),
                "has_pdf":        1 if str(row.get("attachment") or "").endswith(".pdf") else 0,
            }],
        )

    def save_financial(self, result: FinancialResult) -> None:
        highlights  = result.key_highlights or []
        order_val   = _parse_order_value(highlights)
        # Prefer dedicated fields; fall back to parsing highlights for legacy rows
        sector      = result.order_sector or result.sector or _parse_sector(highlights)
        client_type = result.client_type or _parse_client_type(highlights)
        self._fin.upsert(
            ids       = [_fin_id(result.symbol, result.period, result.period_type)],
            documents = [result.to_embed_doc()],
            metadatas = [{
                "symbol":             result.symbol,
                "company":            result.company,
                "period":             result.period,
                "period_label":       result.period_label or result.period or "",
                "period_type":        result.period_type,
                "broadcast_dt":       str(result.broadcast_dt or "")[:16],
                "date_int":           _date_to_int(str(result.broadcast_dt or "")),
                "revenue_cr":         _f(result.revenue_cr),
                "pat_cr":             _f(result.pat_cr),
                "ebitda_margin_pct":  _f(result.ebitda_margin_pct),
                "revenue_growth_pct": _f(result.revenue_growth_pct),
                "pat_growth_pct":     _f(result.pat_growth_pct),
                "eps":                _f(result.eps),
                "order_value_cr":     order_val,
                "sector":             sector or "",
                "client_type":        client_type or "",
                "client_name":        result.client_name or "",
                "order_sector":       result.order_sector or "",
                "execution_months":   result.execution_months or 0,
            }],
        )

    # ── Backfill from SQLite ──────────────────────────────────────────────────

    def backfill_from_sqlite(self, db_path: Path | str, since: str | None = None) -> tuple[int, int]:
        """Push SQLite rows into ChromaDB.

        since : ISO date string (YYYY-MM-DD). Only rows with broadcast_dt >= since
                are upserted. Pass None (or omit) for a full backfill.
                The caller (main.py) can pass the last-known max broadcast_dt to
                make incremental runs fast.
        """
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ann_count = self._backfill_announcements(conn, since=since)
        fin_count = self._backfill_financials(conn, since=since)
        ow_count  = self._backfill_order_wins(conn, since=since)
        conn.close()
        # Rebuild BM25 after all upserts
        print("  Rebuilding BM25 indexes...", flush=True)
        self._build_bm25_indexes()
        print(f"  BM25 ready — ann:{len(self._ann_bm25._ids)}  fin:{len(self._fin_bm25._ids)}", flush=True)
        return ann_count, fin_count + ow_count

    def _backfill_announcements(self, conn: sqlite3.Connection, since: str | None = None) -> int:
        where = "WHERE broadcast_dt >= ?" if since else ""
        params = (since,) if since else ()
        rows = conn.execute(
            "SELECT symbol, company, subject, details, broadcast_dt, "
            "       sector_tags, catalyst_type, order_value_cr, is_novel, attachment "
            f"FROM announcements {where}",
            params,
        ).fetchall()
        total = len(rows)
        print(f"  Backfilling {total} announcements...", flush=True)
        _BATCH = 500
        saved = 0
        for i in range(0, total, _BATCH):
            batch = rows[i : i + _BATCH]
            ids, docs, metas = [], [], []
            for row in batch:
                r       = dict(row)
                symbol  = str(r.get("symbol") or "")
                company = str(r.get("company") or "")
                subject = str(r.get("subject") or "")
                details = str(r.get("details") or "")
                dt      = str(r.get("broadcast_dt") or "")
                ids.append(_ann_id(symbol, dt, subject, details))
                docs.append(_ann_document(company, symbol, subject, details))
                metas.append({
                    "symbol":         symbol,
                    "company":        company,
                    "subject":        subject,
                    "broadcast_dt":   dt[:16],
                    "date_int":       _date_to_int(dt),
                    "sector_tags":    str(r.get("sector_tags") or ""),
                    "catalyst_type":  str(r.get("catalyst_type") or "general"),
                    "order_value_cr": _f(r.get("order_value_cr")),
                    "is_novel":       int(r.get("is_novel") or 0),
                    "has_pdf":        1 if str(r.get("attachment") or "").endswith(".pdf") else 0,
                })
            self._ann.upsert(ids=ids, documents=docs, metadatas=metas)
            saved += len(batch)
            print(f"    {saved}/{total}", end="\r", flush=True)
        print(f"    {saved}/{total} done        ", flush=True)
        return saved

    def _backfill_financials(self, conn: sqlite3.Connection, since: str | None = None) -> int:
        where = "WHERE broadcast_dt >= ?" if since else ""
        params = (since,) if since else ()
        rows = conn.execute(
            "SELECT symbol, company, period, period_label, period_type, broadcast_dt, "
            "       revenue_cr, revenue_growth_pct, ebitda_cr, ebitda_margin_pct, "
            "       pat_cr, pat_growth_pct, eps, key_highlights, guidance, raw_summary, "
            "       sector, client_name, client_type, order_sector, execution_months "
            f"FROM financial_results {where}",
            params,
        ).fetchall()
        total = len(rows)
        print(f"  Backfilling {total} financial results...", flush=True)
        for row in rows:
            r = dict(row)
            try:
                highlights = json.loads(r.get("key_highlights") or "[]")
            except (json.JSONDecodeError, TypeError):
                highlights = []
            order_val   = _parse_order_value(highlights)
            db_sector   = str(r.get("sector") or "")
            db_ct       = str(r.get("client_type") or "")
            db_os       = str(r.get("order_sector") or "")
            sector      = db_os or db_sector or _parse_sector(highlights)
            client_type = db_ct or _parse_client_type(highlights)
            symbol      = str(r.get("symbol") or "")
            company     = str(r.get("company") or "")
            period      = str(r.get("period") or "")
            period_lbl  = str(r.get("period_label") or period)
            pt          = str(r.get("period_type") or "")
            dt          = str(r.get("broadcast_dt") or "")
            rev  = r.get("revenue_cr")
            pat  = r.get("pat_cr")
            hl_str   = " | ".join(str(h) for h in highlights if h is not None)
            raw_s    = str(r.get("raw_summary") or "")
            guidance = str(r.get("guidance") or "")
            rev_s    = f"Revenue Rs.{rev:.0f}Cr" if rev else ""
            pat_s    = f"PAT Rs.{pat:.0f}Cr"     if pat else ""
            document = " ".join(filter(None, [
                f"{company} ({symbol}) {period_lbl}:",
                raw_s,
                " ".join(filter(None, [rev_s, pat_s])),
                hl_str, guidance,
            ]))
            self._fin.upsert(
                ids       = [_fin_id(symbol, period, pt)],
                documents = [document],
                metadatas = [{
                    "symbol":             symbol,
                    "company":            company,
                    "period":             period,
                    "period_label":       period_lbl,
                    "period_type":        pt,
                    "broadcast_dt":       dt[:16],
                    "date_int":           _date_to_int(dt),
                    "revenue_cr":         _f(rev),
                    "pat_cr":             _f(pat),
                    "ebitda_margin_pct":  _f(r.get("ebitda_margin_pct")),
                    "revenue_growth_pct": _f(r.get("revenue_growth_pct")),
                    "pat_growth_pct":     _f(r.get("pat_growth_pct")),
                    "eps":                _f(r.get("eps")),
                    "order_value_cr":     order_val,
                    "sector":             sector or "",
                    "client_type":        client_type or "",
                    "client_name":        str(r.get("client_name") or ""),
                    "order_sector":       db_os,
                    "execution_months":   int(r.get("execution_months") or 0),
                }],
            )
        print(f"    {total}/{total} done", flush=True)
        return total

    def _backfill_order_wins(self, conn: sqlite3.Connection, since: str | None = None) -> int:
        """Push order_wins table rows into ep_financials ChromaDB collection."""
        where  = "WHERE broadcast_dt >= ?" if since else ""
        params = (since,) if since else ()
        rows = conn.execute(
            "SELECT symbol, company, broadcast_dt, order_value_cr, "
            "       client_name, client_type, order_sector, execution_months, "
            "       raw_summary, source_url "
            f"FROM order_wins {where}",
            params,
        ).fetchall()
        total = len(rows)
        if total == 0:
            return 0
        print(f"  Backfilling {total} order wins...", flush=True)
        for row in rows:
            r       = dict(row)
            symbol  = str(r.get("symbol") or "")
            company = str(r.get("company") or "")
            dt      = str(r.get("broadcast_dt") or "")
            ov      = _f(r.get("order_value_cr"))
            ct      = str(r.get("client_type") or "")
            os_     = str(r.get("order_sector") or "")
            cn      = str(r.get("client_name") or "")
            em      = int(r.get("execution_months") or 0)
            raw_s   = str(r.get("raw_summary") or "")
            doc = " ".join(filter(None, [
                f"{company} ({symbol}) order win:",
                raw_s,
                f"Client: {cn}" if cn else "",
                f"Sector: {os_}" if os_ else "",
                f"Value: Rs.{ov:.0f}Cr" if ov > 0 else "",
            ]))
            self._fin.upsert(
                ids       = [_fin_id(symbol, dt[:10], "order_win")],
                documents = [doc],
                metadatas = [{
                    "symbol":          symbol,
                    "company":         company,
                    "period":          dt[:10],
                    "period_label":    dt[:10],
                    "period_type":     "order_win",
                    "broadcast_dt":    dt[:16],
                    "date_int":        _date_to_int(dt),
                    "revenue_cr":      _MISSING,
                    "pat_cr":          _MISSING,
                    "ebitda_margin_pct": _MISSING,
                    "revenue_growth_pct": _MISSING,
                    "pat_growth_pct":  _MISSING,
                    "eps":             _MISSING,
                    "order_value_cr":  ov,
                    "sector":          os_,
                    "client_type":     ct,
                    "client_name":     cn,
                    "order_sector":    os_,
                    "execution_months": em,
                }],
            )
        print(f"    {total}/{total} done", flush=True)
        return total

    # ── Search methods ────────────────────────────────────────────────────────

    def search_order_wins(
        self,
        query:     str,
        min_cr:    float | None = None,
        sector:    str | None = None,
        from_date: str | None = None,
        to_date:   str | None = None,
        n:         int = 10,
    ) -> list[dict]:
        conditions: list[dict] = [{"period_type": {"$eq": "order_win"}}]
        if min_cr is not None:
            conditions.append({"order_value_cr": {"$gte": float(min_cr)}})
        if sector:
            conditions.append({"sector": {"$eq": sector}})
        if from_date:
            conditions.append({"date_int": {"$gte": _date_to_int(from_date)}})
        if to_date:
            conditions.append({"date_int": {"$lte": _date_to_int(to_date)}})
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        return self._query(self._fin, self._fin_bm25, query, where, n)

    def search_financials(
        self,
        query:          str,
        period_type:    str | None = None,
        min_revenue:    float | None = None,
        revenue_growth: bool = False,
        n:              int = 10,
    ) -> list[dict]:
        conditions: list[dict] = []
        if period_type:
            conditions.append({"period_type": {"$eq": period_type}})
        if min_revenue is not None:
            conditions.append({"revenue_cr": {"$gte": float(min_revenue)}})
        if revenue_growth:
            conditions.append({"revenue_growth_pct": {"$gte": 0.1}})
        where = self._build_where(conditions)
        return self._query(self._fin, self._fin_bm25, query, where, n)

    def search_announcements(
        self,
        query:         str,
        subject:       str | None = None,
        catalyst_type: str | None = None,
        from_date:     str | None = None,
        to_date:       str | None = None,
        n:             int = 10,
    ) -> list[dict]:
        conditions: list[dict] = []
        if subject:
            conditions.append({"subject": {"$eq": subject}})
        if catalyst_type:
            conditions.append({"catalyst_type": {"$eq": catalyst_type}})
        if from_date:
            conditions.append({"date_int": {"$gte": _date_to_int(from_date)}})
        if to_date:
            conditions.append({"date_int": {"$lte": _date_to_int(to_date)}})
        where = self._build_where(conditions)
        return self._query(self._ann, self._ann_bm25, query, where, n)

    def search_all(self, query: str, n: int = 10) -> list[dict]:
        """Search both collections, fuse with RRF across collections."""
        # Run hybrid search on each collection independently
        ann_res = self._query(self._ann, self._ann_bm25, query, None, n)
        fin_res = self._query(self._fin, self._fin_bm25, query, None, n)
        # Re-fuse the two already-fused lists by their rrf_score
        combined = ann_res + fin_res
        combined.sort(key=lambda r: r.get("_rrf_score", 0.0), reverse=True)
        return combined[:n]

    def counts(self) -> dict[str, int]:
        return {
            "ep_announcements": self._ann.count(),
            "ep_financials":    self._fin.count(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_where(conditions: list[dict]) -> dict | None:
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _query(
        self,
        col,
        bm25_idx:  _BM25Index,
        query:     str,
        where:     dict | None,
        n:         int,
    ) -> list[dict]:
        total = col.count()
        if total == 0:
            return []

        fetch_n = min(n * 3, total)   # fetch more candidates before RRF

        # ── Dense search ──────────────────────────────────────────────────────
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results":   fetch_n,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            res = col.query(**kwargs)
        except Exception as e:
            print(f"  [vector] query error: {e}")
            return []

        dense_ids  = res["ids"][0]
        dense_docs = res["documents"][0]
        dense_meta = res["metadatas"][0]
        dense_dist = res["distances"][0]

        # Build fast lookup: id → (doc, meta, dist)
        dense_lookup: dict[str, tuple] = {
            id_: (doc, meta, dist)
            for id_, doc, meta, dist in zip(dense_ids, dense_docs, dense_meta, dense_dist)
        }

        # ── BM25 search ───────────────────────────────────────────────────────
        bm25_hits: list[tuple[str, float]] = []
        if bm25_idx.is_built():
            bm25_hits = bm25_idx.search(query, where, fetch_n)

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = _rrf_fuse(bm25_hits, dense_ids, n)

        # ── Fetch docs for BM25-only IDs not in dense results ─────────────────
        bm25_only_ids = [id_ for id_, _ in fused if id_ not in dense_lookup]
        if bm25_only_ids:
            try:
                extra = col.get(
                    ids=bm25_only_ids,
                    include=["documents", "metadatas"],
                )
                for id_, doc, meta in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                    dense_lookup[id_] = (doc, meta, 1.0)  # no distance available
            except Exception:
                pass

        # ── Build output in fused order ───────────────────────────────────────
        out: list[dict] = []
        for id_, rrf_score in fused:
            if id_ not in dense_lookup:
                continue
            doc, meta, dist = dense_lookup[id_]
            row = dict(meta)
            row["_id"]        = id_
            row["_snippet"]   = doc[:300]
            row["_distance"]  = round(dist, 4)
            row["_rrf_score"] = round(rrf_score, 6)
            out.append(row)

        return out
