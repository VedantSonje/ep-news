"""
eval/query_latency_eval.py — Measure ask2 pipeline latency per stage.

Usage:
    python eval/query_latency_eval.py               # retrieval only (no LLM cost)
    python eval/query_latency_eval.py --full        # include LLM explain call
    python eval/query_latency_eval.py --runs 3      # repeat each query N times

Output: per-query table + p50/p95 summary per stage.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.retrieval_agent import RetrievalAgent
from retrieval.pipeline_logger import PipelineTrace

DB_PATH    = Path("data/ep_news.db")
CHROMA_DIR = Path("data/chroma")

# ── Test queries — cover all five intent types ────────────────────────────────
QUERIES = [
    # STRUCTURED (date / symbol filter)
    "defence orders this week",
    "railway contracts last 7 days",
    "HFCL announcements",

    # SEMANTIC (concept search)
    "export orders defence sector",
    "government hospital infrastructure contracts",
    "renewable energy solar power wins",

    # FINANCIAL
    "HFCL financial results revenue",
    "quarterly results revenue growth",

    # HYBRID
    "large defence orders government linked novel",

    # COMPOUND
    "top EP candidates defence railway this month",
    "order wins above 100 crore government",
    "pharma healthcare export announcements",
]


def _stages(trace: PipelineTrace) -> dict[str, float]:
    """Extract per-stage elapsed_ms from a PipelineTrace."""
    return {
        "classifier": trace.classifier.elapsed_ms if trace.classifier else 0.0,
        "sql":        trace.sql.elapsed_ms        if trace.sql        else 0.0,
        "bm25":       trace.bm25.elapsed_ms       if trace.bm25       else 0.0,
        "vector":     trace.vector.elapsed_ms     if trace.vector     else 0.0,
        "fused":      trace.fused.elapsed_ms      if trace.fused      else 0.0,
        "reranked":   trace.reranked.elapsed_ms   if trace.reranked   else 0.0,
        "total":      trace.total_ms,
    }


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return round(sorted_v[idx], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full",  action="store_true", help="Include LLM explain call (costs tokens)")
    ap.add_argument("--runs",  type=int, default=1, help="Repeat each query N times (default 1)")
    ap.add_argument("--top-n", type=int, default=8, help="Candidates for retrieval (default 8)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    import os

    from storage.sql_storage import SQLiteStorage
    from storage.vector_storage import ChromaDBStorage
    from financial.storage import FinancialStorage
    from retrieval.bm25_search import setup_fts

    sql_store = SQLiteStorage(DB_PATH)
    setup_fts(sql_store.connection)
    vec_store = ChromaDBStorage(CHROMA_DIR)
    fin_store = FinancialStorage(DB_PATH, CHROMA_DIR)

    agent = RetrievalAgent(
        conn         = sql_store.connection,
        vector_store = vec_store,
        fin_storage  = fin_store,
        api_key      = os.getenv("ANTHROPIC_API_KEY", ""),
    )

    all_rows: list[dict] = []   # {query, run, stage→ms, ...}
    stage_data: dict[str, list[float]] = {
        "classifier": [], "sql": [], "bm25": [],
        "vector": [], "fused": [], "reranked": [], "total": [],
    }
    llm_times: list[float] = []

    col_w = 38
    hdr = f"{'Query':<{col_w}} {'Run':>3}  {'cls':>5}  {'sql':>5}  {'bm25':>5}  {'vec':>6}  {'fused':>5}  {'rerank':>6}  {'total':>6}"
    if args.full:
        hdr += f"  {'llm':>6}"
    print(hdr)
    print("-" * len(hdr))

    for query in QUERIES:
        for run in range(1, args.runs + 1):
            trace = agent.trace_only(query, top_n=args.top_n)
            s = _stages(trace)

            llm_ms = 0.0
            if args.full:
                import os
                from anthropic import Anthropic
                t0 = time.perf_counter()
                resp = agent.ask(query, top_n=args.top_n)
                llm_ms = round((time.perf_counter() - t0) * 1000 - s["total"], 1)
                llm_times.append(llm_ms)

            row = f"{'…'+query[-col_w+1:] if len(query)>col_w else query:<{col_w}} {run:>3}  " \
                  f"{s['classifier']:>5.0f}  {s['sql']:>5.0f}  {s['bm25']:>5.0f}  " \
                  f"{s['vector']:>6.0f}  {s['fused']:>5.0f}  {s['reranked']:>6.0f}  {s['total']:>6.0f}"
            if args.full:
                row += f"  {llm_ms:>6.0f}"
            print(row)

            for k, v in s.items():
                stage_data[k].append(v)
            all_rows.append({"query": query, "run": run, **s, "llm_ms": llm_ms})

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  LATENCY SUMMARY  ({len(QUERIES)} queries × {args.runs} run(s))")
    print("=" * 65)
    print(f"  {'Stage':<12}  {'p50 ms':>8}  {'p95 ms':>8}  {'max ms':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for stage, values in stage_data.items():
        if not any(v > 0 for v in values):
            continue
        print(f"  {stage:<12}  {_pct(values,50):>8.1f}  {_pct(values,95):>8.1f}  {max(values):>8.1f}")
    if llm_times:
        print(f"  {'llm_call':<12}  {_pct(llm_times,50):>8.1f}  {_pct(llm_times,95):>8.1f}  {max(llm_times):>8.1f}")
    print("=" * 65)

    sql_store.close()
    fin_store.close()


if __name__ == "__main__":
    main()
