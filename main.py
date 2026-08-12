"""
EP News Screener — unified CLI entry point.

Commands
--------
  python main.py ingest <csv_file>          Parse, score, store in SQL + ChromaDB
  python main.py ask "<question>"           Ask the AI agent (tool-use loop)
  python main.py ask2 "<question>"          Ask the retrieval agent (faster, cheaper)
  python main.py enrich                     Run enrichment on existing DB rows
  python main.py top [--min-score N]        Show top EP candidates (no agent)
  python main.py search "<query>"           Semantic search (no agent)
  python main.py extract                    Extract financial PDFs via Claude
  python main.py financials                 View extracted financial results
  python main.py stats                      Database statistics
"""
from __future__ import annotations

import argparse
import sys

# Safe UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from config import AppConfig
from screener.filter_config import FilterConfig
from screener.csv_parser import CsvParser
from screener.scoring_engine import AnnouncementFilter
from screener.pipeline import EPScreener
from storage.database_manager import DatabaseManager
from storage.sql_storage import SQLiteStorage
from agent.ep_agent import EPAgent
from agent.retrieval_agent import RetrievalAgent
from financial.pdf_agent import PDFAgent
from financial.storage import FinancialStorage
from financial.vector_store import VectorStore
from enrichment.sector_tagger import SectorTagger
from enrichment.novelty_detector import NoveltyDetector
from enrichment.materiality import MaterialityExtractor
from retrieval.bm25_search import setup_fts


# ── Helpers ───────────────────────────────────────────────────────────────

def _sep(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _print_row(row: dict) -> None:
    score   = row.get("score", "?")
    symbol  = str(row.get("symbol", "")).ljust(12)
    company = str(row.get("company", ""))[:32].ljust(32)
    tags    = str(row.get("tags", row.get("subject", "")))
    dt      = str(row.get("broadcast_dt", ""))[:16]
    bar     = "#" * min(int(score), 13) if isinstance(score, int) else ""
    print(f"[{score:>2}] {bar:<13}  {symbol} {company}")
    print(f"       {dt}  |  {tags}")


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Parse a CSV, filter by subject, enrich, and persist to both databases."""
    csv_path = args.csv_file

    print(f"Ingesting: {csv_path}")

    config   = FilterConfig()
    parser   = CsvParser(csv_path)
    fltr     = AnnouncementFilter(config)
    screener = EPScreener(parser, fltr)

    # Open SQL connection early so novelty detector can query recent rows
    sql_store = SQLiteStorage(cfg.db_path)
    setup_fts(sql_store.connection)

    results = screener.run(conn=sql_store.connection)
    print(f"Screener: {len(results)} announcements passed subject filter.")

    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        counts = db.save_all(results, source_file=str(csv_path))

    sql_store.close()

    print(f"Saved -> SQL: {counts['sql']} new rows | ChromaDB: {counts['vector']} upserted")
    _sep()

    print(f"\nFirst 10 from this batch:\n")
    for sa in results[:10]:
        ann     = sa.announcement
        sectors = getattr(sa, "sector_tags", "") or ""
        novel   = "NEW" if getattr(sa, "is_novel", True) else "repeat"
        size    = getattr(sa, "relative_size", "")
        print(f"  {ann.symbol:<12} {ann.company[:32]}")
        print(f"  {sa.dt_str}  |  {ann.subject[:40]}")
        print(f"  {size:<8}  {novel}  |  {sectors}\n")


def cmd_ask2(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Ask the retrieval agent (deterministic retrieval + single LLM explain call)."""
    trace_only = getattr(args, "trace_only", False)
    if not trace_only and not cfg.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    question = args.question
    print(f"Question: {question}")
    _sep()

    from storage.vector_storage import ChromaDBStorage
    sql_store = SQLiteStorage(cfg.db_path)
    setup_fts(sql_store.connection)
    vec_store = ChromaDBStorage(cfg.chroma_path)
    fin_store = FinancialStorage(cfg.db_path, cfg.chroma_path)

    agent = RetrievalAgent(
        conn         = sql_store.connection,
        vector_store = vec_store,
        fin_storage  = fin_store,
        api_key      = cfg.anthropic_api_key,
    )

    if getattr(args, "trace_only", False):
        # Show pipeline trace without calling the LLM
        trace = agent.trace_only(question, top_n=args.top_n or 8)
        print(trace.render("full"))
        sql_store.close()
        fin_store.close()
        return

    resp = agent.ask(question, top_n=args.top_n or 8)

    # Print pipeline trace first if requested
    if getattr(args, "trace", False) and resp.trace:
        print(resp.trace.render("full"))
        _sep()

    print(resp.answer)
    _sep("-")
    print(
        f"Intent: {resp.intent}  |  Candidates: {resp.candidates_used}  |  "
        f"Cache read: {resp.cache_read} tok  |  Cache write: {resp.cache_write} tok  |  "
        f"Pipeline: {resp.trace.total_ms:.0f}ms" if resp.trace else ""
    )
    sql_store.close()
    fin_store.close()


def cmd_enrich(args: argparse.Namespace, cfg: AppConfig) -> None:
    """
    Run enrichment (sector tags, novelty, materiality) on existing DB rows
    that have not yet been enriched (sector_tags IS NULL).
    """
    sql_store = SQLiteStorage(cfg.db_path)
    conn      = sql_store.connection
    setup_fts(conn)

    tagger   = SectorTagger()
    novelty  = NoveltyDetector(lookback_days=30)
    material = MaterialityExtractor()

    rows = conn.execute(
        "SELECT id, symbol, company, subject, details, broadcast_dt "
        "FROM announcements WHERE sector_tags IS NULL"
    ).fetchall()

    if not rows:
        print("All rows already enriched.")
        sql_store.close()
        return

    print(f"Enriching {len(rows)} rows...")
    updated = 0
    for row in rows:
        rid, symbol, company, subject, details, broadcast_dt = (
            row[0], row[1], row[2], row[3], row[4] or "", row[5] or ""
        )
        sector_str = tagger.tag_str(subject, details, company)
        mat        = material.extract(subject, details, company)

        sql_store.update_enrichment(
            rid,
            sector_tags       = sector_str,
            is_novel          = 1,   # already passed screener; novelty fires on new ingestion
            relative_size     = mat.relative_size,
            catalyst_type     = mat.catalyst_type,
            government_linked = int(mat.government_linked),
            export_component  = int(mat.export_component),
            order_value_cr    = mat.order_value_cr,
        )
        updated += 1

    print(f"Enriched {updated} rows.")
    _sep()

    # Quick summary
    sector_counts = conn.execute(
        "SELECT sector_tags, COUNT(*) as n FROM announcements "
        "WHERE sector_tags != '' GROUP BY sector_tags ORDER BY n DESC LIMIT 10"
    ).fetchall()
    novel_count   = conn.execute(
        "SELECT COUNT(*) FROM announcements WHERE is_novel = 1"
    ).fetchone()[0]
    repeat_count  = conn.execute(
        "SELECT COUNT(*) FROM announcements WHERE is_novel = 0"
    ).fetchone()[0]

    print(f"  Novel announcements : {novel_count}")
    print(f"  Repeat/update       : {repeat_count}")
    print()
    print("  Top sector tag combinations:")
    for r in sector_counts[:8]:
        print(f"    {r[1]:>4}x  {r[0]}")

    sql_store.close()


def cmd_ask(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Ask the AI agent a natural-language question."""
    if not cfg.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    question = args.question
    print(f"Question: {question}")
    _sep()

    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        agent    = EPAgent(cfg.anthropic_api_key, cfg.model, db)
        response = agent.ask(question)

    print(response.answer)
    _sep("-")
    print(
        f"Model: {response.model_used}  |  "
        f"Tool calls: {response.tool_calls_made}  |  "
        f"Cache read: {response.cache_read_tokens} tok  |  "
        f"Cache write: {response.cache_write_tokens} tok"
    )


def cmd_top(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Show top EP candidates directly from the database (no agent)."""
    min_score = args.min_score or 8
    limit     = args.limit or 15

    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        rows = db.get_top_ep_candidates(min_score=min_score, limit=limit)

    if not rows:
        print(f"No candidates with score >= {min_score}. Run 'ingest' first.")
        return

    _sep()
    print(f"  TOP EP CANDIDATES  (score >= {min_score}, limit {limit})")
    _sep()
    print()
    for row in rows:
        _print_row(row)
        print()


def cmd_search(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Semantic similarity search (no agent)."""
    query = args.query
    n     = args.n or 8

    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        rows = db.semantic_search(query, n_results=n)

    if not rows:
        print("No results. Run 'ingest' first.")
        return

    _sep()
    print(f"  SEMANTIC SEARCH: \"{query}\"")
    _sep()
    print()
    for row in rows:
        _print_row(row)
        dist = row.get("similarity_distance", "?")
        print(f"       similarity distance: {dist}")
        print()


def cmd_extract(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Download PDFs for matched subjects and extract key facts using Docling + Ollama (local, free)."""
    print(f"\nPDF Extraction  (Docling + Ollama qwen2.5:14b — no API cost)")
    print(f"  Limit  : {args.limit or 'all'}")
    print(f"  Symbol : {args.symbol or 'all'}")
    print(f"  Date   : {args.date or 'all'}")
    print(f"  Type   : {args.type or 'all'}")
    _sep()

    agent = PDFAgent(
        db_path     = cfg.db_path,
        chroma_path = cfg.chroma_path,
    )
    agent.run(
        limit      = args.limit,
        symbol     = args.symbol,
        date       = args.date,
        subject    = args.type,
        reextract  = getattr(args, "reextract", False),
        workers    = getattr(args, "workers", None),
    )


def cmd_financials(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Query extracted financial results (no agent needed)."""
    with FinancialStorage(cfg.db_path, cfg.chroma_path) as fs:
        if args.symbol:
            rows = fs.get_by_symbol(args.symbol)
            label = f"Financial results for {args.symbol.upper()}"
        else:
            rows = fs.get_top_by_pat(limit=args.limit or 15)
            label = f"Top {args.limit or 15} by PAT"

    if not rows:
        print("No financial results found. Run 'extract' first.")
        return

    _sep()
    print(f"  {label.upper()}")
    _sep()
    print(f"  {'SYMBOL':<12} {'PERIOD':<14} {'REV(Cr)':>9} {'PAT(Cr)':>9} "
          f"{'PAT%':>8} {'EBDA%':>7} {'EPS':>6}")
    print(f"  {'-'*72}")
    for r in rows:
        rev  = f"{r.get('revenue_cr', 0) or 0:>9.0f}" if r.get("revenue_cr") else "         -"
        pat  = f"{r.get('pat_cr', 0)  or 0:>9.0f}"   if r.get("pat_cr")     else "         -"
        patg = f"{r.get('pat_growth_pct') or 0:>+7.1f}%" if r.get("pat_growth_pct") is not None else "        -"
        ebm  = f"{r.get('ebitda_margin_pct') or 0:>6.1f}%" if r.get("ebitda_margin_pct") else "       -"
        eps  = f"{r.get('eps') or 0:>6.2f}" if r.get("eps") else "      -"
        sym  = str(r.get("symbol", ""))[:12].ljust(12)
        per  = str(r.get("period", ""))[:14].ljust(14)
        print(f"  {sym} {per} {rev} {pat} {patg} {ebm} {eps}")
    print()

    # Show key highlights for first result
    if rows and rows[0].get("key_highlights"):
        try:
            import json as _json
            highlights = _json.loads(rows[0]["key_highlights"])
            sym = rows[0].get("symbol", "")
            print(f"  Key Highlights — {sym}:")
            for h in highlights:
                print(f"    • {h}")
            print()
        except Exception:
            pass


def cmd_backfill(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Push all existing SQLite rows into ChromaDB (safe to re-run — uses upsert)."""
    since: str | None = getattr(args, "since", None)

    # --auto-since: find the latest broadcast_dt already in ChromaDB, backfill only newer rows
    if getattr(args, "auto_since", False) and not since:
        import sqlite3 as _sq
        conn = _sq.connect(str(cfg.db_path))
        row = conn.execute(
            "SELECT MAX(DATE(broadcast_dt)) FROM announcements"
        ).fetchone()
        conn.close()
        # Use 7 days ago as the auto-since cutoff so we never miss stragglers
        from datetime import date as _date, timedelta as _td
        max_dt = row[0] if (row and row[0]) else str(_date.today())
        since = str(_date.fromisoformat(max_dt) - _td(days=7))

    print(f"\nBackfilling SQLite → ChromaDB")
    print(f"  DB     : {cfg.db_path}")
    print(f"  Chroma : {cfg.chroma_path}")
    if since:
        print(f"  Since  : {since}  (incremental)")
    else:
        print(f"  Mode   : full backfill")
    _sep()

    store = VectorStore(cfg.chroma_path)
    ann_count, fin_count = store.backfill_from_sqlite(cfg.db_path, since=since)

    counts = store.counts()
    _sep()
    print(f"  Upserted announcements : {ann_count}")
    print(f"  Upserted financials    : {fin_count}")
    print(f"  ChromaDB ep_announcements : {counts['ep_announcements']} docs")
    print(f"  ChromaDB ep_financials    : {counts['ep_financials']} docs")
    _sep()


def cmd_query(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Semantic + metadata search across announcements and financial results."""
    q    = args.query
    n    = args.n or 8
    mode = (args.type or "all").lower()

    store = VectorStore(cfg.chroma_path)
    counts = store.counts()

    if counts["ep_announcements"] == 0 and counts["ep_financials"] == 0:
        print("ChromaDB is empty. Run: python main.py backfill")
        return

    _sep()
    print(f"  QUERY: \"{q}\"  [{mode}]")
    _sep()
    print()

    if mode in ("order_win", "orders"):
        results = store.search_order_wins(
            q,
            min_cr=args.min_cr,
            n=n,
        )
        _print_fin_results(results)

    elif mode in ("financials", "results", "quarterly"):
        results = store.search_financials(
            q,
            period_type=args.period_type,
            revenue_growth=(args.growth or False),
            n=n,
        )
        _print_fin_results(results)

    elif mode in ("announcements", "ann"):
        results = store.search_announcements(
            q,
            subject=args.subject,
            n=n,
        )
        _print_ann_results(results)

    else:
        # Default: search both, show combined
        ann = store.search_announcements(q, n=n // 2 + 1)
        fin = store.search_financials(q, n=n // 2 + 1)

        if fin:
            print("  Financial Results:")
            _print_fin_results(fin)
        if ann:
            print("  Announcements:")
            _print_ann_results(ann)


def _print_fin_results(results: list[dict]) -> None:
    if not results:
        print("  No results found.\n")
        return
    for r in results:
        sym  = str(r.get("symbol", "")).ljust(12)
        co   = str(r.get("company", ""))[:30].ljust(30)
        pt   = str(r.get("period_type", ""))
        per  = str(r.get("period", ""))
        dist = r.get("_distance", "?")
        print(f"  {sym} {co}  [{pt}]  {per}  (dist: {dist})")

        # Show key numbers
        ov  = r.get("order_value_cr", -1)
        rev = r.get("revenue_cr", -1)
        pat = r.get("pat_cr", -1)
        sec = r.get("sector", "")
        ct  = r.get("client_type", "")

        if ov and ov > 0:
            print(f"       Order value: Rs.{ov:,.0f} Cr"
                  + (f"  |  Sector: {sec}" if sec else "")
                  + (f"  |  {ct}" if ct else ""))
        if rev and rev > 0:
            print(f"       Revenue: Rs.{rev:,.0f} Cr"
                  + (f"  |  PAT: Rs.{pat:,.0f} Cr" if pat and pat > 0 else ""))
        snippet = r.get("_snippet", "")
        if snippet:
            print(f"       {snippet[:120]}...")
        print()


def _print_ann_results(results: list[dict]) -> None:
    if not results:
        print("  No results found.\n")
        return
    for r in results:
        sym     = str(r.get("symbol", "")).ljust(12)
        co      = str(r.get("company", ""))[:30].ljust(30)
        subj    = str(r.get("subject", ""))[:45]
        dt      = str(r.get("broadcast_dt", ""))[:10]
        dist    = r.get("_distance", "?")
        sectors = str(r.get("sector_tags", ""))
        print(f"  {sym} {co}  {dt}  (dist: {dist})")
        print(f"       {subj}")
        if sectors:
            print(f"       Sectors: {sectors}")
        snippet = r.get("_snippet", "")
        if snippet:
            print(f"       {snippet[snippet.find(':')+1:].strip()[:110]}...")
        print()


def cmd_stats(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Print database statistics."""
    with DatabaseManager(cfg.db_path, cfg.chroma_path) as db:
        s = db.stats()

    _sep()
    print(f"  DATABASE STATS")
    _sep()
    print(f"  SQL total rows    : {s['sql_total']}")
    print(f"  ChromaDB docs     : {s['vector_total']}")
    print()
    if s.get("top_symbols"):
        print("  Top symbols by score:")
        for r in s["top_symbols"]:
            print(f"    {r['symbol']:<12}  max score: {r['score']}")


# ── CLI setup ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ep_news",
        description="EP News Screener — Stockbee-style EP filter for Indian equities",
    )
    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ingest
    p_ingest = sub.add_parser("ingest", help="Parse CSV, filter by subject, and store in both DBs")
    p_ingest.add_argument("csv_file", help="Path to BSE/NSE announcements CSV")

    # ask (tool-use loop agent)
    p_ask = sub.add_parser("ask", help="Ask the AI agent (tool-use loop)")
    p_ask.add_argument("question", help="Your question in plain English")

    # ask2 (retrieval agent — deterministic + single LLM call)
    p_ask2 = sub.add_parser("ask2", help="Ask the retrieval agent (faster, cheaper)")
    p_ask2.add_argument("question", help="Your question in plain English")
    p_ask2.add_argument("--top-n", type=int, default=8, metavar="N",
                        help="Number of candidates to surface (default: 8)")
    p_ask2.add_argument("--trace", action="store_true",
                        help="Print full pipeline trace before the answer")
    p_ask2.add_argument("--trace-only", action="store_true",
                        help="Print pipeline trace without calling the LLM")

    # enrich
    sub.add_parser("enrich", help="Run enrichment (sectors, novelty, materiality) on DB rows")

    # top
    p_top = sub.add_parser("top", help="Show top EP candidates from the database")
    p_top.add_argument("--min-score", type=int, default=8, metavar="N")
    p_top.add_argument("--limit", type=int, default=15, metavar="N")

    # search
    p_search = sub.add_parser("search", help="Semantic search in the vector store")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument("-n", type=int, default=8, metavar="N")

    # extract
    p_extract = sub.add_parser(
        "extract",
        help="Download PDFs for matched subjects and extract key facts using Claude",
    )
    p_extract.add_argument("--limit",  type=int, default=None, metavar="N",
                           help="Max PDFs to process (default: all)")
    p_extract.add_argument("--symbol", type=str, default=None, metavar="SYM",
                           help="Restrict to one stock symbol")
    p_extract.add_argument("--date",   type=str, default=None, metavar="YYYY-MM-DD",
                           help="Filter by broadcast date prefix")
    p_extract.add_argument("--type",   type=str, default=None, metavar="TYPE",
                           help="Extraction type: financial_results / order_win / acquisition / "
                                "restructuring / fundraising / buyback / open_offer / "
                                "credit_rating / cirp")
    p_extract.add_argument("--reextract", action="store_true",
                           help="Re-process PDFs already in DB (update existing records)")
    p_extract.add_argument("--workers", type=int, default=None, metavar="N",
                           help="Parallel workers for download+extract (default: 8)")

    # financials
    p_fin = sub.add_parser("financials", help="View extracted financial results")
    p_fin.add_argument("--symbol", type=str, default=None, metavar="SYM")
    p_fin.add_argument("--limit",  type=int, default=15,   metavar="N")

    # stats
    sub.add_parser("stats", help="Print database statistics")

    # backfill
    p_backfill = sub.add_parser(
        "backfill",
        help="Push all SQLite rows into ChromaDB (safe to re-run)",
    )
    p_backfill.add_argument(
        "--since", metavar="YYYY-MM-DD",
        help="Only upsert rows with broadcast_dt >= this date (incremental). "
             "Omit for full backfill.",
    )
    p_backfill.add_argument(
        "--auto-since", action="store_true",
        help="Auto-detect the date of the oldest un-indexed row and backfill only from there.",
    )

    # query
    p_query = sub.add_parser(
        "query",
        help="Semantic + metadata search (e.g. 'defence order wins above 100cr')",
    )
    p_query.add_argument("query", help="Natural language search query")
    p_query.add_argument(
        "--type", type=str, default="all", metavar="TYPE",
        help="Collection to search: all / order_win / financials / announcements (default: all)",
    )
    p_query.add_argument("-n", type=int, default=8, metavar="N",
                         help="Number of results (default: 8)")
    p_query.add_argument("--min-cr", type=float, default=None, metavar="CR",
                         help="Minimum order/revenue value in Crore (order_win mode)")
    p_query.add_argument("--period-type", type=str, default=None,
                         help="Filter by period_type: quarterly / annual / order_win")
    p_query.add_argument("--subject", type=str, default=None,
                         help="Filter by exact BSE subject name")
    p_query.add_argument("--growth", action="store_true",
                         help="Filter to companies with positive revenue growth")

    return root


def main() -> None:
    cfg    = AppConfig.from_env()
    cfg.ensure_dirs()
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "ingest":     cmd_ingest,
        "ask":        cmd_ask,
        "ask2":       cmd_ask2,
        "enrich":     cmd_enrich,
        "top":        cmd_top,
        "search":     cmd_search,
        "stats":      cmd_stats,
        "extract":    cmd_extract,
        "financials": cmd_financials,
        "backfill":   cmd_backfill,
        "query":      cmd_query,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
