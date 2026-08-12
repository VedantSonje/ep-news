"""
ChromaDBStorage — vector / RAG storage for semantic search.

Metadata stored per document
-----------------------------
  symbol, company, subject, score, tags, broadcast_dt, attachment
  sector_tags    — comma-separated sector labels (from enrichment)
  catalyst_type  — order_win / results / dividend / acquisition / …
  date_str       — ISO date only "YYYY-MM-DD"  (for range filtering)

Semantic search supports:
  - min_score    — ChromaDB metadata filter ($gte)
  - date_from    — inclusive lower bound on date_str
  - date_to      — inclusive upper bound on date_str

Freshness / TTL
---------------
  purge_stale_for_date(symbol, date_str) — deletes all ChromaDB entries
      for a (symbol, date) pair, so re-ingesting a CSV replaces stale
      embeddings instead of accumulating orphaned entries.

  purge_by_ids(stale_ids) — bulk delete by exact ChromaDB document IDs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from models import ScoredAnnouncement
from storage.base import BaseStorage


class ChromaDBStorage(BaseStorage):
    """
    Persistent ChromaDB vector store.
    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via onnxruntime).
    """

    _COLLECTION = "ep_announcements"

    def __init__(self, chroma_path: Path | str) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise ImportError("chromadb is required: pip install chromadb") from e

        self._path = Path(chroma_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._col = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ── BaseStorage interface ──────────────────────────────────────────────

    def save(self, items: list[ScoredAnnouncement], source_file: str = "") -> int:
        if not items:
            return 0

        seen_ids: dict[str, int] = {}
        ids, docs, metas = [], [], []

        for sa in items:
            ann = sa.announcement
            base_id = hashlib.md5(
                f"{ann.symbol}|{ann.broadcast_raw}|{ann.subject}|{ann.details}".encode()
            ).hexdigest()

            if base_id in seen_ids:
                seen_ids[base_id] += 1
                doc_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                doc_id = base_id

            document = (
                f"{ann.company} ({ann.symbol}): {ann.subject}. "
                f"{ann.details[:500]}"
            )
            ids.append(doc_id)
            docs.append(document)

            # Enrichment fields — present on ScoredAnnouncement if pipeline ran enrichment
            sector_tags  = getattr(sa, "sector_tags", "") or ""
            catalyst_type = getattr(sa, "catalyst_type", "general") or "general"
            date_str     = sa.dt_str[:10] if sa.dt_str else ""

            # date_int: YYYYMMDD as int — enables ChromaDB numeric range filtering
            date_int = _date_to_int(date_str)

            metas.append({
                "symbol":        ann.symbol,
                "company":       ann.company,
                "subject":       ann.subject,
                "score":         sa.score,
                "tags":          sa.result.tag_str,
                "broadcast_dt":  sa.dt_str,
                "attachment":    ann.attachment or "",
                "sector_tags":   sector_tags,
                "catalyst_type": catalyst_type,
                "date_str":      date_str,
                "date_int":      date_int,
            })

        self._col.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        pass

    # ── Semantic search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: int | None = None,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search with optional metadata pre-filters.

        date_from / date_to use ISO date format "YYYY-MM-DD" and filter
        on the date_str metadata field, reducing the candidate set before
        embedding similarity is computed.
        """
        where = _build_chroma_where(min_score, date_from, date_to)
        n     = min(n_results, self._col.count() or 1)

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results":   n,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        output: list[dict[str, Any]] = []
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for meta, dist in zip(metas, dists):
            row = dict(meta)
            row["similarity_distance"] = round(dist, 4)
            output.append(row)
        return output

    # ── Freshness / TTL ────────────────────────────────────────────────────

    def purge_stale_for_date(self, symbol: str, date_str: str) -> int:
        """
        Delete all ChromaDB entries for (symbol, date_str).
        Call before re-upserting so updated filings replace stale embeddings.
        Returns the number of entries deleted.
        """
        date_int = _date_to_int(date_str)
        try:
            existing = self._col.get(
                where={"$and": [
                    {"symbol":   {"$eq": symbol}},
                    {"date_int": {"$eq": date_int}},
                ]},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self._col.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def purge_by_ids(self, stale_ids: list[str]) -> int:
        """Delete ChromaDB entries by exact document ID."""
        if not stale_ids:
            return 0
        try:
            self._col.delete(ids=stale_ids)
            return len(stale_ids)
        except Exception:
            return 0

    def get_ids_for_date(self, date_str: str) -> list[str]:
        """Return all ChromaDB document IDs for a given date."""
        date_int = _date_to_int(date_str)
        try:
            result = self._col.get(
                where={"date_int": {"$eq": date_int}},
                include=[],
            )
            return result.get("ids", [])
        except Exception:
            return []

    def migrate_add_date_int(self) -> int:
        """
        One-time migration: add date_int to existing entries that have
        broadcast_dt but no date_int.  Safe to run multiple times.
        Returns count of updated entries.
        """
        try:
            all_items = self._col.get(include=["metadatas"])
        except Exception:
            return 0

        ids_to_update: list[str] = []
        new_metas: list[dict] = []

        for doc_id, meta in zip(all_items.get("ids", []),
                                all_items.get("metadatas", [])):
            if meta and "date_int" not in meta:
                dt_str = (meta.get("broadcast_dt") or "")[:10]
                date_str = dt_str if len(dt_str) == 10 else ""
                new_meta = dict(meta)
                new_meta["date_str"]      = date_str
                new_meta["date_int"]      = _date_to_int(date_str)
                new_meta.setdefault("sector_tags",   "")
                new_meta.setdefault("catalyst_type", "general")
                ids_to_update.append(doc_id)
                new_metas.append(new_meta)

        if ids_to_update:
            # Process in batches of 100 to avoid memory issues
            for i in range(0, len(ids_to_update), 100):
                self._col.update(
                    ids=ids_to_update[i:i+100],
                    metadatas=new_metas[i:i+100],
                )
        return len(ids_to_update)

    def update_metadata(self, doc_id: str, **fields) -> None:
        """Patch metadata fields on an existing document (no re-embedding)."""
        if not fields:
            return
        try:
            self._col.update(ids=[doc_id], metadatas=[fields])
        except Exception:
            pass


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    if not date_str or len(date_str) < 10:
        return 0
    try:
        return int(date_str[:10].replace("-", ""))
    except ValueError:
        return 0


def _build_chroma_where(
    min_score: int | None,
    date_from: str | None,
    date_to:   str | None,
) -> dict | None:
    """
    Build a ChromaDB `where` filter dict.
    ChromaDB supports $eq, $gte, $lte on int/float and $eq on strings.
    Date range uses date_int (YYYYMMDD as int) for numeric comparison.
    """
    conditions: list[dict] = []

    if min_score is not None:
        conditions.append({"score": {"$gte": min_score}})

    if date_from:
        conditions.append({"date_int": {"$gte": _date_to_int(date_from)}})

    if date_to:
        conditions.append({"date_int": {"$lte": _date_to_int(date_to)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
