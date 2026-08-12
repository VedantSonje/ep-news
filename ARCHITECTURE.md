# EP News Screener — Architecture

---

## 1. System Overview

```mermaid
flowchart TB
    classDef input    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef screen   fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef storage  fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef fin      fill:#fce7f3,stroke:#db2777,color:#831843
    classDef agent    fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef cli      fill:#f1f5f9,stroke:#64748b,color:#0f172a

    %% ── Inputs ──────────────────────────────────────────────────────
    CSV[/"📄 BSE / NSE Announcements CSV\nCF-AN-equities-DD-MM-YYYY.csv\n700+ rows per day"/]:::input
    USER["👤 User / Trader"]:::input

    %% ── Screener Pipeline ────────────────────────────────────────────
    subgraph SCREEN["🔍 SCREENER PIPELINE   screener/"]
        direction LR
        FC["FilterConfig\n──────────\n• 28 drop subjects\n• 16 keep subjects\n• 6 compiled regexes\n• base scores 1-13"]
        CP["CsvParser\n──────────\n• ColumnMapper\n• BSE/NSE\n  auto-detect\n• datetime parse"]
        AF["AnnouncementFilter\n──────────────────\nsubject gate\n(first pass, cheap)"]
        SE["ScoringEngine\n─────────────\n• 15 per-subject\n  handlers\n• dispatch table\n• order value Rs.Cr"]
        EP["EPScreener\n───────────\norchestrator\nfilter→score\n→sort"]

        FC --> AF
        FC --> SE
        CP --> AF --> SE --> EP
    end

    %% ── Storage Layer ────────────────────────────────────────────────
    subgraph STORE["💾 STORAGE LAYER   storage/"]
        DM["DatabaseManager\n(facade — single entry point)"]
        SQL[("🗄️ SQLiteStorage\nannouncements table\n177 rows · 2 days\nSELECT queries")]
        VEC[("🧠 ChromaDBStorage\nep_announcements\n180 docs embedded\ncosine similarity")]
        DM --> SQL
        DM --> VEC
    end

    %% ── Financial Pipeline ───────────────────────────────────────────
    subgraph FIN["📊 FINANCIAL PDF PIPELINE   financial/"]
        PA["PDFAgent\n(orchestrator)"]
        DL["PDFDownloader\n──────────────\nurllib · retry\nbrowser UA\n10 MB cap · SSL"]
        FE["FinancialExtractor\n──────────────────\nclaude-opus-4-7\nPDF base64 input\nJSON schema output\nprompt caching"]
        FSQL[("🗄️ financial_results\nSQL table\nrevenue · PAT\nEBITDA · EPS")]
        FVEC[("🧠 ep_financials\nChromaDB collection\nnarrative embeddings")]

        PA --> DL --> FE
        FE --> FSQL
        FE --> FVEC
    end

    %% ── AI Agent ─────────────────────────────────────────────────────
    subgraph AGENT["🤖 AI AGENT   agent/"]
        TOOLS["ToolExecutor\n────────────────────────\n① query_sql\n② semantic_search\n③ get_top_ep_candidates\n④ get_company_announcements"]
        BOT["EPAgent\n──────────────────────────\nclaude-haiku-4-5\nManual agentic loop\nPrompt caching · 5 min TTL\nMax 10 tool iterations"]
        BOT <--> TOOLS
    end

    %% ── CLI ──────────────────────────────────────────────────────────
    subgraph CLI["⌨️  CLI   main.py"]
        direction LR
        C1["ingest"]
        C2["ask"]
        C3["top"]
        C4["search"]
        C5["extract"]
        C6["financials"]
        C7["stats"]
    end

    %% ── Data Flow ────────────────────────────────────────────────────
    CSV  --> CP
    EP   --> |ScoredAnnouncement\nscore + tags| DM

    SQL  --> |rows with\nfinancial results tag| PA

    TOOLS <--> DM
    TOOLS <--> FSQL

    USER --> CLI
    C1 --> SCREEN & STORE
    C2 --> BOT
    C3 --> SQL
    C4 --> VEC
    C5 --> FIN
    C6 --> FSQL
    C7 --> DM

    BOT --> |"natural language\nanswer"| USER
```

---

## 2. Screener Scoring Logic

```mermaid
flowchart LR
    classDef drop  fill:#fee2e2,stroke:#dc2626
    classDef keep  fill:#dcfce7,stroke:#16a34a
    classDef score fill:#dbeafe,stroke:#3b82f6

    ANN["Announcement\n(symbol, subject,\ndetails, attachment)"]

    ANN --> GATE{"Subject\nin drop list?\n28 subjects"}
    GATE -- YES --> DROP["❌ Dropped\n(noise)"]:::drop
    GATE -- NO  --> GATE2{"Subject\nin keep list?\n16 subjects"}
    GATE2 -- NO  --> DROP
    GATE2 -- YES --> SCORE["ScoringEngine\ndispatch"]:::score

    SCORE --> S13["Score 13\nOrder ≥ Rs.1000 cr"]
    SCORE --> S11["Score 11-12\nOrder Rs.100-999 cr"]
    SCORE --> S10["Score 10\nOrder (no value)"]
    SCORE --> S9["Score 9\nAcquisition\nPlant commissioning"]
    SCORE --> S8["Score 8\nBoard results\nCapacity addition\nScheme of arrangement"]
    SCORE --> S5["Score 5-7\nDividend · C-suite change\nHigh-value press release"]
    SCORE --> S4["Score 4\nDividend Rs.1-4/share"]
    SCORE --> S1["Score 1-3\n(below min_score\n→ filtered out)"]:::drop
```

---

## 3. Financial PDF Extraction Flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI   as main.py extract
    participant PA    as PDFAgent
    participant SQL1  as SQLiteStorage
    participant DL    as PDFDownloader
    participant API   as NSE Archive
    participant FE    as FinancialExtractor
    participant CL    as Claude Opus 4.7
    participant FS    as FinancialStorage
    participant SQL2  as financial_results (SQL)
    participant CHR   as ep_financials (ChromaDB)

    CLI  ->> PA   : run(limit, symbol, date, min_score)
    PA   ->> SQL1 : SELECT announcements WHERE tags LIKE '%financial results%'
    SQL1 -->> PA  : list of candidates (symbol, attachment, broadcast_dt)

    loop For each candidate PDF
        PA   ->> PA   : is_pdf_url(attachment)?
        PA   ->> PA   : already_stored(symbol, url)?

        PA   ->> DL   : download(url)
        DL   ->> API  : GET pdf with browser headers + retry
        API  -->> DL  : PDF bytes (100 KB – 10 MB)
        DL  -->> PA   : (bytes, "ok")

        PA   ->> FE   : extract(pdf_bytes, symbol, company, broadcast_dt)
        FE   ->> FE   : base64 encode PDF
        FE   ->> CL   : messages.create(document=base64_pdf, output_config=JSON_schema)
        Note over CL  : Extracts: period, revenue_cr,<br/>revenue_growth_pct, ebitda_cr,<br/>ebitda_margin_pct, pat_cr,<br/>pat_growth_pct, eps,<br/>dividend, highlights, guidance
        CL  -->> FE   : structured JSON (guaranteed valid)
        FE  -->> PA   : FinancialResult dataclass

        PA   ->> FS   : save(result)
        FS   ->> SQL2 : INSERT OR IGNORE financial_results
        FS   ->> CHR  : upsert(id, narrative_doc, metadata)
    end

    PA  -->> CLI  : RunSummary (extracted, failed, skipped)
```

---

## 4. AI Agent Tool-Use Loop

```mermaid
sequenceDiagram
    autonumber
    participant U  as User
    participant AG as EPAgent
    participant CL as Claude Haiku 4.5
    participant TX as ToolExecutor
    participant DB as DatabaseManager

    U  ->> AG : ask("Which stocks got large orders today?")

    AG ->> CL : messages.create(<br/>system=[cached prompt],<br/>tools=[4 tools],<br/>messages=[question])
    Note over CL : system prompt cached<br/>cache_control: ephemeral<br/>5-min TTL → ~90% cost saving

    CL -->> AG : stop_reason="tool_use"<br/>tool: get_top_ep_candidates(min_score=10)

    AG ->> TX : execute("get_top_ep_candidates", {min_score:10})
    TX ->> DB : get_top_by_score(10, 10)
    DB -->> TX : [{AFCONS, score:13}, {KEC, score:13}, ...]
    TX -->> AG : JSON result

    AG ->> CL : messages=[..., tool_result]

    CL -->> AG : stop_reason="tool_use"<br/>tool: semantic_search("defence electronics order")

    AG ->> TX : execute("semantic_search", {query:...})
    TX ->> DB : vector.search("defence electronics order")
    DB -->> TX : [{BEL, dist:0.58}, {PREMEXPLN, dist:0.66}]
    TX -->> AG : JSON result

    AG ->> CL : messages=[..., tool_result]

    CL -->> AG : stop_reason="end_turn"<br/>text="Top EP candidates today:\n[13] AFCONS..."

    AG -->> U  : AgentResponse(answer, tool_calls_made=2,<br/>cache_read_tokens=..., cache_write_tokens=...)
```

---

## 5. Data Model

```mermaid
erDiagram
    ANNOUNCEMENTS {
        int     id              PK
        text    symbol
        text    company
        text    subject
        text    details
        int     score
        text    tags
        text    broadcast_dt
        text    attachment
        text    source_file
        text    ingested_at
    }

    FINANCIAL_RESULTS {
        int     id                  PK
        text    symbol
        text    company
        text    period
        text    period_type
        real    revenue_cr
        real    revenue_growth_pct
        real    ebitda_cr
        real    ebitda_margin_pct
        real    pat_cr
        real    pat_growth_pct
        real    eps
        real    dividend_per_share
        text    key_highlights
        text    guidance
        text    raw_summary
        text    source_url
        text    broadcast_dt
        text    extracted_at
    }

    CHROMA_EP_ANNOUNCEMENTS {
        text    id              PK
        text    document
        text    symbol
        text    company
        text    subject
        int     score
        text    tags
        text    broadcast_dt
        text    attachment
    }

    CHROMA_EP_FINANCIALS {
        text    id              PK
        text    document
        text    symbol
        text    company
        text    period
        text    period_type
        real    revenue_cr
        real    pat_cr
        real    pat_growth_pct
        real    ebitda_margin_pct
        real    eps
        text    broadcast_dt
    }

    ANNOUNCEMENTS      ||--o{ FINANCIAL_RESULTS      : "symbol (join)"
    ANNOUNCEMENTS      ||--o| CHROMA_EP_ANNOUNCEMENTS : "same record"
    FINANCIAL_RESULTS  ||--o| CHROMA_EP_FINANCIALS    : "same record"
```

---

## 6. File Structure

```
ep_news/
│
├── main.py                        ← CLI entry point (7 commands)
├── config.py                      ← AppConfig  (env vars)
├── models.py                      ← Shared dataclasses
│
├── screener/                      ── SCREENER PIPELINE ──────────────
│   ├── filter_config.py           ← FilterConfig (frozen, regexes)
│   ├── csv_parser.py              ← ColumnMapper + CsvParser
│   ├── scoring_engine.py          ← AnnouncementFilter + ScoringEngine
│   └── pipeline.py                ← EPScreener (orchestrator)
│
├── storage/                       ── STORAGE LAYER ──────────────────
│   ├── base.py                    ← BaseStorage (ABC)
│   ├── sql_storage.py             ← SQLiteStorage
│   ├── vector_storage.py          ← ChromaDBStorage
│   └── database_manager.py        ← DatabaseManager (facade)
│
├── financial/                     ── FINANCIAL PDF PIPELINE ─────────
│   ├── models.py                  ← FinancialResult + ExtractionStatus
│   ├── pdf_downloader.py          ← PDFDownloader (urllib + retry)
│   ├── extractor.py               ← FinancialExtractor (Claude)
│   ├── storage.py                 ← FinancialStorage (SQL + ChromaDB)
│   └── pdf_agent.py               ← PDFAgent (orchestrator)
│
├── agent/                         ── AI AGENT LAYER ─────────────────
│   ├── tools.py                   ← TOOL_DEFINITIONS + ToolExecutor
│   └── ep_agent.py                ← EPAgent (Claude agentic loop)
│
├── data/                          ── PERSISTENT DATA ────────────────
│   ├── ep_news.db                 ← SQLite (announcements + financials)
│   └── chroma/                    ← ChromaDB (ep_announcements + ep_financials)
│
├── requirements.txt
└── .env
```

---

## 7. Technology Choices

| Layer | Local / Dev | Production Upgrade |
|---|---|---|
| **SQL** | SQLite (zero-config, file-based) | PostgreSQL — change connection string only |
| **Vector / RAG** | ChromaDB (persistent, local) | Qdrant (Docker / cloud) |
| **Scraper model** | claude-haiku-4-5 (fast, cheap) | claude-opus-4-7 (better accuracy) |
| **Extraction model** | claude-opus-4-7 (accuracy first) | Keep as-is |
| **PDF download** | urllib (stdlib, no deps) | aiohttp for async parallel downloads |

---

## 8. Cost Optimisations

| Technique | Where used | Effect |
|---|---|---|
| **Prompt caching** (`cache_control: ephemeral`) | EPAgent system prompt, FinancialExtractor system prompt | ~90% cost reduction on repeated queries |
| **Subject gate first** | AnnouncementFilter before ScoringEngine | Drops ~80% of rows before regex matching |
| **Frozen FilterConfig** | All regexes compiled once at startup | Zero re-compilation per announcement |
| **INSERT OR IGNORE** | SQLiteStorage + ChromaDB upsert | Safe to re-ingest same CSV — no duplicates |
| **claude-haiku-4-5 for agent** | EPAgent Q&A loop | $1/M tokens vs $5/M for Opus |
| **claude-opus-4-7 for extraction** | FinancialExtractor | Accuracy over cost for one-time PDF parse |
