"""
PipelineTrace — structured log of every retrieval stage.

Each stage records: what ran, how many hits, the top-5 snapshots, and elapsed time.
The trace is attached to RetrievalResponse so callers can inspect or print it.

Render modes:
  trace.render("full")    — all stages with hit tables
  trace.render("compact") — one line per stage
  trace.render("json")    — machine-readable
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HitSnapshot:
    rank:            int
    symbol:          str
    subject:         str
    score:           int
    retrieval_score: float   # bm25 rank / cosine dist / rrf score (depends on stage)
    sources:         list[str] = field(default_factory=list)

    def one_line(self) -> str:
        src = "+".join(self.sources) if self.sources else "-"
        return (
            f"  #{self.rank:<2} [{self.score:>2}] {self.symbol:<12} "
            f"{self.subject[:40]:<40}  score={self.retrieval_score:.4f}  src={src}"
        )


@dataclass
class StageLog:
    name:       str
    count:      int
    elapsed_ms: float = 0.0
    notes:      str   = ""
    top_hits:   list[HitSnapshot] = field(default_factory=list)

    def render_header(self) -> str:
        return f"[{self.name.upper():<12}]  hits={self.count:<4}  {self.elapsed_ms:.1f}ms  {self.notes}"

    def render_full(self) -> str:
        lines = [self.render_header()]
        for h in self.top_hits:
            lines.append(h.one_line())
        return "\n".join(lines)


@dataclass
class PipelineTrace:
    query:       str
    classifier:  StageLog | None = None
    sql:         StageLog | None = None
    bm25:        StageLog | None = None
    vector:      StageLog | None = None
    fused:       StageLog | None = None
    reranked:    StageLog | None = None
    explanation: str   = ""
    total_ms:    float = 0.0

    _start: float = field(default_factory=time.time, repr=False, compare=False)

    def stop(self) -> None:
        self.total_ms = round((time.time() - self._start) * 1000, 1)

    def render(self, mode: Literal["full", "compact", "json"] = "full") -> str:
        if mode == "json":
            return self._to_json()
        stages = [s for s in [
            self.classifier, self.sql, self.bm25,
            self.vector, self.fused, self.reranked,
        ] if s is not None]

        if mode == "compact":
            lines = [f"Query: {self.query!r}  total={self.total_ms:.0f}ms"]
            for s in stages:
                lines.append(f"  {s.render_header()}")
            return "\n".join(lines)

        # full
        sep = "-" * 72
        lines = [
            sep,
            f"  PIPELINE TRACE  |  {self.query!r}  |  total {self.total_ms:.0f}ms",
            sep,
        ]
        for s in stages:
            lines.append("")
            lines.append(s.render_full())
        if self.explanation:
            lines.append("")
            lines.append("[EXPLANATION]")
            lines.append(self.explanation[:300] + ("..." if len(self.explanation) > 300 else ""))
        lines.append(sep)
        return "\n".join(lines)

    def _to_json(self) -> str:
        def _stage(s: StageLog | None) -> dict | None:
            if s is None:
                return None
            return {
                "name": s.name,
                "count": s.count,
                "elapsed_ms": s.elapsed_ms,
                "notes": s.notes,
                "top_hits": [
                    {"rank": h.rank, "symbol": h.symbol, "score": h.score,
                     "retrieval_score": h.retrieval_score, "sources": h.sources}
                    for h in s.top_hits
                ],
            }
        return json.dumps({
            "query":      self.query,
            "total_ms":   self.total_ms,
            "classifier": _stage(self.classifier),
            "sql":        _stage(self.sql),
            "bm25":       _stage(self.bm25),
            "vector":     _stage(self.vector),
            "fused":      _stage(self.fused),
            "reranked":   _stage(self.reranked),
        }, indent=2)


# ── Helpers used by broker/reranker ──────────────────────────────────────────

def make_stage(
    name: str,
    items,          # list[CandidateDoc] | list[BM25Hit] | list[dict]
    elapsed_ms: float = 0.0,
    notes: str = "",
    score_attr: str = "rrf_score",
    max_hits: int = 5,
) -> StageLog:
    """Build a StageLog from any list of hit objects."""
    snapshots: list[HitSnapshot] = []
    for rank, item in enumerate(items[:max_hits]):
        if isinstance(item, dict):
            snapshots.append(HitSnapshot(
                rank=rank,
                symbol=item.get("symbol", ""),
                subject=item.get("subject", "")[:40],
                score=int(item.get("score", 0)),
                retrieval_score=round(float(item.get("similarity_distance", 0)), 4),
                sources=["vector"],
            ))
        else:
            r_score = getattr(item, score_attr, getattr(item, "bm25_rank", 0.0))
            snapshots.append(HitSnapshot(
                rank=rank,
                symbol=getattr(item, "symbol", ""),
                subject=getattr(item, "subject", "")[:40],
                score=int(getattr(item, "score", 0)),
                retrieval_score=round(float(r_score), 4),
                sources=getattr(item, "sources", []),
            ))
    return StageLog(name=name, count=len(items), elapsed_ms=elapsed_ms,
                    notes=notes, top_hits=snapshots)
