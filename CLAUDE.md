# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

### Run the web server (primary interface)
```bash
python api_server.py
# Serves on http://localhost:8000
# Auto-fetches NSE data on cold start if DB is stale; runs full update daily at 22:00
```

### CLI pipeline (`main.py`)
```bash
python fetch_today.py                          # fetch today's NSE announcements → auto-ingest
python main.py ingest <csv_file>               # ingest a downloaded CSV manually
python main.py ask2 "defence orders today"     # AI Q&A (fast, cheap — use this)
python main.py ask "defence orders today"      # AI Q&A (tool-use loop — powerful but more API calls)
python main.py top --min-score 10              # show top EP candidates from DB
python main.py extract --limit 20              # download PDFs + extract financials
python main.py financials --symbol HFCL        # view extracted financial results
python main.py stats                           # DB row counts
python main.py backfill --auto-since           # sync SQLite → ChromaDB
```

### Standalone screener (no DB, quick scan)
```bash
python ep_screener.py CF-AN-equities-26-08-2026.csv --min-score 8 --verbose
```

---

## Architecture

### Two separate entry points — different purposes

**`ep_screener.py`** (root, self-contained): standalone scorer. Has its own `FilterConfig`, `CsvParser`, `AnnouncementFilter`, `ScoringEngine`, `EPScreener`, `ReportPrinter`. Outputs to stdout/CSV only. No database. Full 1–13 scoring logic lives here.

**`main.py`** (modular, production): uses the `screener/` package. The `screener/pipeline.py` `EPScreener` applies the subject filter and enrichment but always sets `score=0` — the subject allowlist is the sole gate. Persists everything that passes to both SQLite and ChromaDB. The score column in the DB reflects enrichment-derived materiality, not the 1–13 scoring engine.

### Storage: two databases in sync

Every ingest writes to both backends simultaneously via `DatabaseManager` (facade in `storage/database_manager.py`):

- **SQLite** (`data/ep_news.db`) — structured queries, FTS5 full-text search (set up via `retrieval/bm25_search.py`), enrichment columns (`sector_tags`, `order_value_cr`, `government_linked`, `export_component`, `is_novel`).
- **ChromaDB** (`data/chroma/`) — two collections: `ep_announcements` (semantic search on announcements) and `ep_financials` (semantic search on financial results). On re-ingest, stale vectors for the same (symbol, date) are purged before upserting.

### LLM routing — two separate stacks

**API server** (`api_server.py` + `api/`): uses `api/llm_client.py`. Priority: Groq (`GROQ_API_KEY`) → Ollama (`llama3.1:latest`). Claude is never called here.

**CLI agents** (`main.py ask` / `main.py ask2`): use the Anthropic SDK directly. Require `ANTHROPIC_API_KEY`. Model configured via `EP_MODEL` env var (default `claude-haiku-4-5`).

### PDF extraction — three layers, in order

`financial/pdf_agent.py` orchestrates:
1. **Rule-based** (`financial/rule_extractor.py`): pure regex on text extracted by pdfplumber/PyMuPDF. Free, instant.
2. **Local LLM** (`financial/local_extractor.py`): Ollama (`llama3.1:latest`). Primary extraction path — section-aware chunking, Pydantic validation on JSON output, handles financial tables via `financial/financial_table_extractor.py`.
3. **Vision fallback** (`financial/vision_extractor.py`): Gemini Flash (`GEMINI_API_KEY`). For image-only/scanned PDFs pdfplumber can't parse.
4. **Claude** (`financial/extractor.py` for financials, `financial/general_extractor.py` for order wins/press releases): highest accuracy, requires `ANTHROPIC_API_KEY`. Sends base64-encoded PDF directly to Claude.

### Two AI agents — different trade-offs

**`EPAgent`** (`ask` command): manual Claude tool-use loop, up to 10 iterations, 4 tools (`query_sql`, `semantic_search`, `get_top_ep_candidates`, `get_company_announcements`). System prompt cached with `cache_control: ephemeral`.

**`RetrievalAgent`** (`ask2` command): deterministic retrieval → single LLM call. Five-stage pipeline:
1. `QueryClassifier` (pure regex) → intent: `STRUCTURED` / `SEMANTIC` / `HYBRID` / `FINANCIAL` / `COMPOUND`
2. `RetrievalBroker` → routes to SQL / BM25 / vector / financial search
3. `HybridSearch` → BM25 (FTS5) + ChromaDB fused via RRF (k=60)
4. SQL enrichment → sector, novelty, materiality per candidate
5. `Reranker` → keyword boost + score boost + recency boost → single Claude explain call

Use `--trace` / `--trace-only` flags on `ask2` to debug the retrieval pipeline without LLM cost.

### Enrichment (inline on ingest)

`screener/pipeline.py` runs three enrichers on every announcement before storage:
- `SectorTagger`: regex keyword → sector label (defence, railway, pharma, etc.)
- `MaterialityExtractor`: extracts order value in ₹Cr, classifies size (tiny/small/medium/large/mega), flags `government_linked` and `export_component`
- `NoveltyDetector`: 30-day lookback in SQLite to detect repeat filings

### Environment variables that affect behavior

| Variable | Effect |
|---|---|
| `GROQ_API_KEY` | Switches API server LLM from Ollama to Groq |
| `ANTHROPIC_API_KEY` | Required for `ask`, `ask2`, Claude-based PDF extraction |
| `GEMINI_API_KEY` | Enables vision fallback for scanned PDFs |
| `ONNX_DEVICE=cpu` | Forces ChromaDB embeddings to CPU — required on 6 GB VRAM to leave full VRAM for Ollama |
| `EP_MODEL` | Claude model for CLI agents (default `claude-haiku-4-5`) |
| `OLLAMA_MODEL` | Ollama model for local extraction and API server (default `llama3.1:latest`) |

### Key data flow: `fetch_today.py` → daily run
`fetch_today.py` hits the NSE API, saves JSON as CSV to `data/fetched/`, then runs the standard ingest pipeline from `main.py`. `api_server.py` calls this automatically at 22:00 and on cold start.
