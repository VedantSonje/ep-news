"""Generate interview-ready PDF for EP News Screener project."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "EP News Screener - Project Documentation", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(25, 60, 120)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def code(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        for line in text.split("\n"):
            self.cell(0, 4.5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def table_row(self, cells, widths, bold=False, fill=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 8.5)
        if fill:
            self.set_fill_color(230, 240, 250)
        h = 5.5
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, str(cell), border=1, fill=fill)
        self.ln(h)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(4, 5, chr(8226))
        self.multi_cell(0, 5, text)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ============================================================================
# PAGE 1: TITLE
# ============================================================================
pdf.add_page()
pdf.ln(30)
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 15, "EP News Screener", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "AI-Powered Stock Announcement Screening System", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "for Indian Equity Markets (BSE / NSE)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(50, 50, 50)
pdf.cell(0, 7, "Tech Stack: Python 3.12 | SQLite + FTS5 | ChromaDB | Claude AI (Haiku + Opus)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "36 files | ~5,000 lines | 7 packages | 2 AI agents | 3 search methods", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_draw_color(25, 60, 120)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 7, "1,268 announcements processed | May 4 - June 23, 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ============================================================================
# PAGE 2: PROBLEM & SOLUTION
# ============================================================================
pdf.add_page()
pdf.section_title("1. Problem Statement")
pdf.body(
    "Every trading day, BSE and NSE publish 700-4000 corporate announcements. "
    "Most are noise (AGM notices, newspaper copies, record dates). Hidden in this flood are "
    "EP (Episodic Pivot) catalysts - large orders, earnings surprises, acquisitions, capacity "
    "commissioning - that can drive 10-50% stock moves.\n\n"
    "Manual screening takes 2-3 hours per day and misses cross-referencing opportunities. "
    "This system automates the entire pipeline from raw CSV to AI-powered analysis."
)

pdf.section_title("2. Solution Overview")
pdf.body(
    "End-to-end pipeline that ingests raw exchange CSVs, filters noise (80% dropped), "
    "scores catalysts (1-13), enriches with sector/materiality/novelty metadata, stores in "
    "dual databases (SQL + vector), and answers natural language questions via AI agents."
)

pdf.sub_title("Architecture Flow")
pdf.code(
    "CSV (4000 rows)\n"
    "  |-- Screener: filter + score --> 200-750 candidates\n"
    "  |-- Enrichment: sector tags + materiality + novelty\n"
    "  |-- Storage: SQLite (structured) + ChromaDB (semantic)\n"
    "  |-- Retrieval: BM25 + Vector + RRF fusion + Reranker\n"
    "  |-- Agent: Claude explains ranked candidates\n"
    "  --> User gets actionable EP watchlist"
)

pdf.section_title("3. Technology Stack")
w = [55, 50, 85]
pdf.table_row(["Layer", "Technology", "Why This Choice"], w, bold=True, fill=True)
pdf.table_row(["Language", "Python 3.12", "Data + ML ecosystem"], w)
pdf.table_row(["SQL Database", "SQLite + FTS5", "Zero-config, built-in BM25 search"], w)
pdf.table_row(["Vector Database", "ChromaDB", "all-MiniLM-L6-v2 via ONNX, cosine sim"], w)
pdf.table_row(["LLM (Agent)", "Claude Haiku 4.5", "Fast, cheap ($1/M tokens)"], w)
pdf.table_row(["LLM (PDF Extract)", "Claude Opus 4.7", "Accuracy-first for financial tables"], w)
pdf.table_row(["Search Fusion", "RRF (k=60)", "Merges BM25 keyword + vector semantic"], w)
pdf.table_row(["Reranking", "Proxy (Jaccard+score)", "No extra model dependency"], w)
pdf.table_row(["PDF Download", "urllib (stdlib)", "Browser UA, retry, SSL, 10MB cap"], w)

# ============================================================================
# PAGE 3: PACKAGES
# ============================================================================
pdf.add_page()
pdf.section_title("4. Package Architecture (7 Packages, 36 Files)")

pdf.sub_title("Package 1: screener/ -- CSV Screening Pipeline")
pdf.body(
    "FilterConfig: 28 drop subjects, 16 keep subjects, 6 compiled regexes\n"
    "CsvParser.parse(): reads CSV, maps BSE/NSE column variants, 7 datetime formats\n"
    "AnnouncementFilter.should_keep(ann): first gate, drops ~80% noise\n"
    "ScoringEngine.score(ann): dispatch table of 15 subject handlers, scores 1-13\n"
    "EPScreener.run(min_score, conn): orchestrator - filter > score > enrich > sort"
)

pdf.sub_title("Package 2: enrichment/ -- Metadata Enrichment (Zero LLM)")
pdf.body(
    "SectorTagger.tag(): 13 sector labels via regex (defence, railway, solar, pharma...)\n"
    "MaterialityExtractor.extract(): order value (Rs.Cr/Lakh/Mn/USD), relative_size "
    "(tiny/small/medium/large/mega), catalyst_type, government_linked, export_component\n"
    "NoveltyDetector.detect(): Jaccard similarity vs recent DB rows, keyword signals"
)

pdf.sub_title("Package 3: storage/ -- Dual Database Layer")
pdf.body(
    "SQLiteStorage: 18-column schema, FTS5 for BM25, INSERT OR IGNORE (idempotent)\n"
    "ChromaDBStorage: cosine similarity, metadata pre-filtering (date_int, score), "
    "purge_stale_for_date() for freshness/TTL\n"
    "DatabaseManager: facade pattern, atomic dual-write with freshness purge"
)

pdf.sub_title("Package 4: retrieval/ -- Deterministic Candidate Generation")
pdf.body(
    "QueryClassifier.classify(): regex intent detection (STRUCTURED/SEMANTIC/HYBRID/"
    "FINANCIAL/COMPOUND), extracts symbols, date, score, themes\n"
    "BM25Search: SQLite FTS5 with porter stemming, auto-synced via triggers\n"
    "HybridSearch: RRF fusion -- score(d) = 1/(k+rank), k=60\n"
    "Reranker: 4 boost components (keyword 40%, EP score 30%, recency 20%, diversity 10%)\n"
    "RetrievalBroker: routes query, enriches from SQL, applies reranker, builds PipelineTrace"
)

pdf.sub_title("Package 5: financial/ -- PDF Extraction Pipeline")
pdf.body(
    "PDFDownloader: urllib + browser headers + retry + 10MB cap\n"
    "FinancialExtractor: Claude Opus 4.7, base64 PDF, JSON schema output "
    "(revenue, EBITDA, PAT, growth%, EPS, dividend, highlights, guidance)\n"
    "PDFAgent: orchestrator - query SQL > download > extract > store, 1.5s delay"
)

pdf.sub_title("Package 6: agent/ -- Two AI Agents")
pdf.body(
    "EPAgent (tool-use loop): Claude Haiku + 4 tools, up to 10 iterations, 4-6 API calls\n"
    "RetrievalAgent (candidate-first): deterministic retrieval > single LLM explain, "
    "1 API call, 3x cheaper, per-stock evidence breakdown"
)

# ============================================================================
# PAGE 4: SCORING SYSTEM
# ============================================================================
pdf.add_page()
pdf.section_title("5. Scoring System (Dispatch Table Pattern)")

pdf.body("The ScoringEngine uses a dispatch table mapping each subject type to a handler method:")

pdf.code(
    "_DISPATCH = {\n"
    '    "Bagging/Receiving of orders":    self._orders,\n'
    '    "Outcome of Board Meeting":       self._board_meeting,\n'
    '    "Acquisition":                    self._acquisition,\n'
    '    "Dividend":                       self._dividend,\n'
    '    "Commencement of production":     self._commissioning,\n'
    "    ...  # 15 handlers total\n"
    "}"
)

w = [15, 95, 80]
pdf.table_row(["Score", "Meaning", "Example"], w, bold=True, fill=True)
pdf.table_row(["13", "Order >= Rs.1000 crore", "TEXRAIL Rs.4045cr railway order"], w)
pdf.table_row(["11", "Order Rs.100-999 crore", "GOODLUCK Rs.255cr defence order"], w)
pdf.table_row(["10", "Order win (value not disclosed)", "HFCL, RAILTEL, RVNL orders"], w)
pdf.table_row(["9", "Acquisition / Commissioning", "WIPRO acquisition, ACMESOLAR commissioning"], w)
pdf.table_row(["8", "Board results / Capacity addition", "DLF Q4 results, SRF capacity addition"], w)
pdf.table_row(["6-7", "Dividend / C-suite / Press release", "LUPIN Rs.18/share dividend"], w)
pdf.table_row(["4-5", "Low-value dividend / Generic", "Filtered but stored for completeness"], w)

pdf.section_title("6. Enrichment Examples")
pdf.body("Every announcement gets enriched with structured metadata (all regex, zero LLM):")

w = [25, 25, 20, 25, 28, 20, 20, 27]
pdf.set_font("Helvetica", "B", 7.5)
pdf.table_row(["Symbol", "Order Val", "Size", "Catalyst", "Sector", "GOV", "EXP", "Novel"], w, bold=True, fill=True)
pdf.set_font("Helvetica", "", 7.5)
pdf.table_row(["TEXRAIL", "Rs.4045cr", "mega", "order_win", "railway,capital", "No", "Yes", "Yes"], w)
pdf.table_row(["AFCONS", "Rs.2450cr", "mega", "order_win", "infrastructure", "No", "No", "Yes"], w)
pdf.table_row(["GOODLUCK", "Rs.255cr", "medium", "order_win", "defence", "No", "No", "Yes"], w)
pdf.table_row(["HFCL", "Rs.162cr", "medium", "order_win", "infrastructure", "No", "Yes", "Yes"], w)
pdf.table_row(["NIBE", "Rs.156cr", "medium", "order_win", "defence", "Yes", "No", "Yes"], w)
pdf.table_row(["BIOCON", "Rs.4569cr", "mega", "results", "pharma", "No", "No", "Yes"], w)
pdf.table_row(["LUPIN", "n/a", "unknown", "results", "pharma", "No", "No", "Yes"], w)
pdf.table_row(["ACMESOLAR", "n/a", "unknown", "commissioning", "solar", "No", "No", "Yes"], w)

# ============================================================================
# PAGE 5: RETRIEVAL PIPELINE
# ============================================================================
pdf.add_page()
pdf.section_title("7. Retrieval Pipeline (with Example)")

pdf.sub_title("Query: 'defence stocks with large orders'")

pdf.body("Step 1 - QueryClassifier (pure Python, no LLM):")
pdf.code(
    "ParsedQuery(\n"
    "    intent     = SEMANTIC (vector_only)\n"
    "    themes     = ['defence']\n"
    "    symbols    = []\n"
    "    date_filter= None\n"
    "    min_score  = None\n"
    ")"
)

pdf.body("Step 2 - RetrievalBroker routes to vector search (because intent=SEMANTIC):")
pdf.code(
    "ChromaDB semantic search --> 40 hits in 541ms\n"
    "Top vector hits:\n"
    "  #1 BLKASHYAP  Bagging/Receiving of orders  dist=0.621\n"
    "  #2 PREMEXPLN  Bagging/Receiving of orders  dist=0.674\n"
    "  #3 AVANTEL    Bagging/Receiving of orders  dist=0.693"
)

pdf.body("Step 3 - RRF Fusion (BM25 + Vector merged):")
pdf.code(
    "score(doc) = sum_over_lists( 1 / (k + rank) )  where k=60\n"
    "Fused pool: 20 candidates with combined RRF scores"
)

pdf.body("Step 4 - Proxy Reranker (4 components):")
pdf.code(
    "final = base_rrf * (1 + kw_boost + score_boost + recency_boost + diversity)\n"
    "\n"
    "Components:\n"
    "  keyword_boost  = 0.4 * jaccard(query_tokens, doc_tokens)\n"
    "  score_boost    = 0.3 * (ep_score / 13)\n"
    "  recency_boost  = 0.2 * max(0, 1 - age_days/7)\n"
    "  diversity_boost= 0.1 if doc in both BM25 and vector"
)

pdf.body("Step 5 - Per-stock evidence (sent to LLM):")
pdf.code(
    "TEXRAIL [13]  order=Rs.4045cr(mega) | catalyst=order_win\n"
    "              sectors=[capital_goods,railway] | EXPORT\n"
    "              src=sql | rrf=1.300 | boosts(kw=0.000,sc=0.300)\n"
    "\n"
    "GOODLUCK [11] order=Rs.255cr(medium) | catalyst=order_win\n"
    "              sectors=[defence]\n"
    "              src=sql | rrf=1.070 | boosts(kw=0.011,sc=0.254)"
)

pdf.body("Step 6 - Single Claude Haiku call explains the ranked candidates with "
         "'WHY INCLUDED' and 'CONCERN' per stock, ending with 'EP Watch' summary.")

# ============================================================================
# PAGE 6: PIPELINE TRACE
# ============================================================================
pdf.add_page()
pdf.section_title("8. Pipeline Trace (Full Observability)")

pdf.body("Every query produces a PipelineTrace logging all 6 stages:")

pdf.code(
    "------------------------------------------------------------------------\n"
    "  PIPELINE TRACE  |  'order crore score above 10'  |  total 15ms\n"
    "------------------------------------------------------------------------\n"
    "\n"
    "[CLASSIFIER  ]  hits=1   0.0ms  intent=sql_only  themes=[]\n"
    "                                symbols=[]  min_score=10\n"
    "\n"
    "[SQL         ]  hits=21  0.5ms  filters: score>=10\n"
    "  #1 [13] KEC          Bagging/Receiving of orders   score=1.00\n"
    "  #2 [13] AFCONS       Bagging/Receiving of orders   score=1.00\n"
    "  #3 [11] BLKASHYAP    Bagging/Receiving of orders   score=0.85\n"
    "\n"
    "[FUSED       ]  hits=21  0.0ms  pre-rerank pool=21\n"
    "\n"
    "[RERANKED    ]  hits=5   0.0ms  query_tokens=5\n"
    "  #1 [13] AFCONS       rrf=1.318  src=sql\n"
    "  #2 [13] KEC          rrf=1.300  src=sql\n"
    "  #3 [11] EMSLIMITED   rrf=1.075  src=sql\n"
    "------------------------------------------------------------------------"
)

pdf.body("Render modes: trace.render('full') | trace.render('compact') | trace.render('json')\n"
         "CLI: python main.py ask2 --trace  or  --trace-only (no LLM call)")

pdf.section_title("9. Two AI Agents Compared")

w = [40, 75, 75]
pdf.table_row(["Aspect", "Agent 1: EPAgent", "Agent 2: RetrievalAgent"], w, bold=True, fill=True)
pdf.table_row(["Architecture", "Tool-use loop (up to 10 rounds)", "Candidate-first + single explain"], w)
pdf.table_row(["Model", "Claude Haiku 4.5", "Claude Haiku 4.5"], w)
pdf.table_row(["API calls/query", "4-6 calls", "1 call"], w)
pdf.table_row(["Cost", "Higher (multi-round)", "~3x cheaper"], w)
pdf.table_row(["Tools available", "query_sql, semantic_search, etc.", "None (deterministic retrieval)"], w)
pdf.table_row(["Prompt caching", "Yes (5-min TTL)", "Yes (5-min TTL)"], w)
pdf.table_row(["Observability", "Tool call log", "Full PipelineTrace"], w)
pdf.table_row(["Per-stock evidence", "No", "Yes (boost breakdown)"], w)

# ============================================================================
# PAGE 7: INPUT/OUTPUT EXAMPLES
# ============================================================================
pdf.add_page()
pdf.section_title("10. Input / Output Examples")

pdf.sub_title("Example 1: Ingest Command")
pdf.code(
    "$ python main.py ingest CF-AN-equities-16-06-2026-to-23-06-2026.csv\n"
    "\n"
    "Ingesting: CF-AN-equities-16-06-2026-to-23-06-2026.csv  (min_score=4)\n"
    "Screener: 223 candidates passed filtering.\n"
    "Saved -> SQL: 223 new rows | ChromaDB: 223 upserted\n"
    "\n"
    "Top candidates from this batch:\n"
    "  [11]  GOODLUCK   medium  NEW  | defence\n"
    "  [11]  TEXRAIL    medium  NEW  | capital_goods,railway\n"
    "  [10]  MARINE     unknown NEW  | order/contract\n"
    "  [10]  HFCL       unknown NEW  | railway"
)

pdf.sub_title("Example 2: Top EP Candidates")
pdf.code(
    "$ python main.py top --min-score 10 --limit 5\n"
    "\n"
    "  TOP EP CANDIDATES  (score >= 10)\n"
    "  [13] #############  TEXRAIL     Texmaco Rail & Engineering\n"
    "       2026-05-12  |  order/contract, Rs.4045 cr, railway\n"
    "  [13] #############  AFCONS      Afcons Infrastructure\n"
    "       2026-05-04  |  order/contract, Rs.2450 cr, infrastructure\n"
    "  [13] #############  KEC         KEC International\n"
    "       2026-05-05  |  order/contract, Rs.1002 cr, power"
)

pdf.sub_title("Example 3: Semantic Search")
pdf.code(
    '$ python main.py search "solar renewable capacity expansion"\n'
    "\n"
    "  SEMANTIC SEARCH: \"solar renewable capacity expansion\"\n"
    "  [9] ACMESOLAR   Commencement of commercial production\n"
    "      similarity distance: 0.5734\n"
    "  [9] PREMIERENE  Acquisition\n"
    "      similarity distance: 0.6312\n"
    "  [8] BORORENEW   Outcome of Board Meeting\n"
    "      similarity distance: 0.6589"
)

pdf.sub_title("Example 4: AI Agent Query")
pdf.code(
    '$ python main.py ask2 "Which defence stocks received large orders?"\n'
    "\n"
    "Intent: hybrid | Candidates: 5 | Pipeline: 560ms\n"
    "\n"
    "TEXRAIL [13] - Rs.4045cr mega railway order with export\n"
    "  component. Largest order in the database. Strong EP setup.\n"
    "GOODLUCK [11] - Rs.255cr defence order. Mid-cap defence\n"
    "  play, respectable order size for company revenue.\n"
    "NIBE [11] - Rs.156cr defence order via subsidiary.\n"
    "  Government/PSU contract. Interesting defence angle.\n"
    "\n"
    "EP Watch: TEXRAIL stands out with mega Rs.4045cr order -\n"
    "  railway + export + multiple follow-on orders in same week."
)

# ============================================================================
# PAGE 8: DESIGN PATTERNS & COST
# ============================================================================
pdf.add_page()
pdf.section_title("11. Design Patterns Used")

w = [45, 65, 80]
pdf.table_row(["Pattern", "Where", "Benefit"], w, bold=True, fill=True)
pdf.table_row(["Facade", "DatabaseManager over SQL+ChromaDB", "Single entry point, keeps DBs in sync"], w)
pdf.table_row(["Strategy", "RetrievalBroker routes queries", "Easy to add new retrieval strategies"], w)
pdf.table_row(["Dispatch Table", "ScoringEngine (15 handlers)", "Clean separation per subject type"], w)
pdf.table_row(["Abstract Base", "BaseStorage for SQL/ChromaDB", "Swappable backends (Postgres, Qdrant)"], w)
pdf.table_row(["Pipeline/Chain", "CSV>filter>score>enrich>store", "Each stage independently testable"], w)
pdf.table_row(["Idempotent Write", "INSERT OR IGNORE + upsert", "Safe to re-ingest same CSV"], w)
pdf.table_row(["Migration", "ALTER TABLE for new columns", "Backward compat with older DBs"], w)
pdf.table_row(["Prompt Caching", "cache_control: ephemeral", "~90% token cost savings"], w)

pdf.section_title("12. Cost Optimizations")

w = [60, 60, 70]
pdf.table_row(["Technique", "Where", "Savings"], w, bold=True, fill=True)
pdf.table_row(["Subject gate first", "AnnouncementFilter", "Drops 80% before scoring"], w)
pdf.table_row(["Compiled regexes", "FilterConfig (frozen)", "Zero re-compilation per row"], w)
pdf.table_row(["Prompt caching", "Agent system prompts", "~90% token cost reduction"], w)
pdf.table_row(["Haiku for agent", "EPAgent + RetrievalAgent", "$1/M vs $15/M (Opus)"], w)
pdf.table_row(["Deterministic retrieval", "Retrieval package", "Candidate gen costs $0"], w)
pdf.table_row(["Single LLM call", "RetrievalAgent (Agent 2)", "3x cheaper than tool loop"], w)
pdf.table_row(["FTS5 built-in", "BM25Search", "No extra search server"], w)

pdf.section_title("13. Key Numbers")

pdf.body(
    "- 36 Python files, ~5,000 lines of code\n"
    "- 7 packages: screener, enrichment, storage, retrieval, financial, agent, CLI\n"
    "- 1,268 announcements processed (May 4 - June 23, 2026)\n"
    "- 18-column SQL schema with 7 enrichment columns\n"
    "- 13 sector labels, 7 catalyst types, 5 size buckets\n"
    "- 2 AI agents (tool-loop + candidate-first)\n"
    "- 3 search methods: SQL exact, BM25 keyword (FTS5), vector semantic (ChromaDB)\n"
    "- RRF fusion merges BM25 + vector with k=60\n"
    "- 4-component proxy reranker (no extra model)\n"
    "- 6-stage pipeline trace for full observability\n"
    "- Prompt caching: 5-min TTL, ~90% cost savings on repeated queries"
)

# ============================================================================
# PAGE 9: PRODUCTION UPGRADES
# ============================================================================
pdf.add_page()
pdf.section_title("14. Production Upgrade Path")

w = [45, 55, 90]
pdf.table_row(["Component", "Current (Local)", "Production"], w, bold=True, fill=True)
pdf.table_row(["SQL Database", "SQLite", "PostgreSQL (change conn string only)"], w)
pdf.table_row(["Vector Store", "ChromaDB (local)", "Qdrant (Docker/cloud)"], w)
pdf.table_row(["Agent Model", "Claude Haiku 4.5", "Claude Opus (better accuracy)"], w)
pdf.table_row(["PDF Downloads", "urllib (sequential)", "aiohttp (async parallel)"], w)
pdf.table_row(["Reranker", "Proxy (Jaccard)", "Cross-encoder model"], w)
pdf.table_row(["Deployment", "CLI", "FastAPI + Scheduler + Dashboard"], w)

pdf.section_title("15. CLI Commands Reference")

w = [55, 135]
pdf.table_row(["Command", "Description"], w, bold=True, fill=True)
pdf.table_row(["ingest <csv>", "Parse, score, enrich, store in SQL + ChromaDB"], w)
pdf.table_row(["ask '<question>'", "Agent 1: tool-use loop (Claude Haiku)"], w)
pdf.table_row(["ask2 '<question>'", "Agent 2: deterministic retrieval + single LLM explain"], w)
pdf.table_row(["ask2 --trace-only", "Show full pipeline trace without calling LLM"], w)
pdf.table_row(["enrich", "Backfill sector/materiality/novelty on existing DB rows"], w)
pdf.table_row(["top --min-score N", "Top EP candidates directly from SQL (no AI)"], w)
pdf.table_row(["search '<query>'", "Semantic similarity search via ChromaDB"], w)
pdf.table_row(["extract --limit N", "Download PDFs + extract financials via Claude Opus"], w)
pdf.table_row(["financials", "View extracted revenue/PAT/EPS table"], w)
pdf.table_row(["stats", "Database row counts and top symbols"], w)

pdf.ln(8)
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Built with Python 3.12 | Claude AI (Anthropic) | SQLite + ChromaDB", align="C")

# ============================================================================
# SAVE
# ============================================================================
output_path = r"D:\algos\ep news\EP_News_Screener_Project.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
