"""
RAGAS evaluation for the EP-News RAG pipeline.

Runs 50+ curated questions through retrieve() + stream_response() and scores
each answer using RAGAS metrics (faithfulness, answer_relevancy,
context_precision, context_recall) with llama3:latest as the judge LLM.

Also computes ranking metrics (Hit@5, MRR, NDCG@5) for questions that have
a known set of relevant symbols — no LLM judge needed for these.

Usage:
    python eval/ragas_eval.py                  # all questions
    python eval/ragas_eval.py --n 10           # first 10 (quick smoke test)
    python eval/ragas_eval.py --out results.csv
    python eval/ragas_eval.py --skip-llm       # retrieval quality only, no LLM judge
    python eval/ragas_eval.py --retrieval-only # only retrieve, skip generation
    python eval/ragas_eval.py --category adversarial  # run one category
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# langchain-community 0.3.x removed chat_models.vertexai; stub it so ragas can import
import types as _types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from api.chat_handler import ChatHandler

# ── 54 curated test questions ─────────────────────────────────────────────────
# Grounded in actual DB data (data/ep_news.db, July 2026).
# ground_truth  = ideal short answer (used by RAGAS for recall/precision scoring)
# relevant_symbols = symbols that MUST appear in top-5 retrieval results
#                    (used for Hit@5, MRR, NDCG@5 — empty means skip ranking metrics)
# adversarial   = True means the query should return 0 or near-0 useful results

TEST_CASES: list[dict] = [
    # ── ORDER WINS (10) ──────────────────────────────────────────────────────
    {
        "category": "order_win",
        "question": "Which defence companies received orders above Rs.1000 Cr?",
        "ground_truth": (
            "Blue Star Limited received a defence order worth Rs.12402 Cr. "
            "Bharat Electronics Limited (BEL) received a defence order worth Rs.1081 Cr."
        ),
        "relevant_symbols": ["BLUESTARCO", "BEL"],
    },
    {
        "category": "order_win",
        "question": "Show railway sector order wins",
        "ground_truth": (
            "Texmaco Rail & Engineering Limited received railway orders worth Rs.4045 Cr "
            "and Rs.253 Cr. Transrail Lighting Limited received a railway order of Rs.600 Cr."
        ),
        "relevant_symbols": ["TEXRAIL", "TRANSRAILL"],
    },
    {
        "category": "order_win",
        "question": "What orders did Blue Star win recently?",
        "ground_truth": (
            "Blue Star Limited received a defence/PSU order worth Rs.12402 Cr in May 2026."
        ),
        "relevant_symbols": ["BLUESTARCO"],
    },
    {
        "category": "order_win",
        "question": "Power sector order wins above Rs.1000 Cr",
        "ground_truth": (
            "Adani Energy Solutions Limited received a power sector order worth Rs.8500 Cr in July 2026."
        ),
        "relevant_symbols": ["ADANIENSOL"],
    },
    {
        "category": "order_win",
        "question": "Real estate companies with large deal announcements",
        "ground_truth": (
            "Oberoi Realty received orders worth Rs.6000 Cr and Rs.4000 Cr. "
            "Prestige Estates Projects received an order worth Rs.6000 Cr in July 2026."
        ),
        "relevant_symbols": ["OBEROIRLTY", "PRESTIGE"],
    },
    {
        "category": "order_win",
        "question": "Bharat Forge order wins",
        "ground_truth": (
            "Bharat Forge Limited received a defence order worth Rs.425 Cr in June 2026."
        ),
        "relevant_symbols": ["BHARATFORG"],
    },
    {
        "category": "order_win",
        "question": "List the top order wins above Rs.500 Cr",
        "ground_truth": (
            "Top order wins include Prudent Corporate Rs.13900 Cr, Blue Star Rs.12402 Cr, "
            "Adani Energy Solutions Rs.8500 Cr, Oberoi Realty Rs.6000 Cr, "
            "Prestige Estates Rs.6000 Cr, Biocon Rs.4569 Cr, Texmaco Rail Rs.4045 Cr."
        ),
        "relevant_symbols": ["ADANIENSOL", "OBEROIRLTY", "PRESTIGE"],
    },
    {
        "category": "order_win",
        "question": "GRSE order wins",
        "ground_truth": (
            "Garden Reach Shipbuilders & Engineers (GRSE) received a defence order."
        ),
        "relevant_symbols": ["GRSE"],
    },
    {
        "category": "order_win",
        "question": "HFCL telecom order wins",
        "ground_truth": (
            "HFCL Limited received telecom infrastructure orders."
        ),
        "relevant_symbols": ["HFCL"],
    },
    {
        "category": "order_win",
        "question": "Defence and PSU sector orders above Rs.200 Cr",
        "ground_truth": (
            "Blue Star received Rs.12402 Cr, BEL received Rs.1081 Cr, "
            "Bharat Forge received Rs.425 Cr."
        ),
        "relevant_symbols": ["BLUESTARCO", "BEL", "BHARATFORG"],
    },
    # ── FINANCIALS (10) ──────────────────────────────────────────────────────
    {
        "category": "financials",
        "question": "BPCL Q1 FY2027 financial results",
        "ground_truth": (
            "Bharat Petroleum Corporation Limited (BPCL) reported Q1 FY2027 revenue of "
            "Rs.159479 Cr and PAT of Rs.3962 Cr with EBITDA margin of -1.8%."
        ),
        "relevant_symbols": ["BPCL"],
    },
    {
        "category": "financials",
        "question": "Larsen and Toubro Q1 FY2027 results",
        "ground_truth": (
            "Larsen & Toubro (L&T) reported Q1 FY2027 revenue of Rs.63679 Cr, "
            "PAT of Rs.3617 Cr, and EBITDA margin of 9.9%."
        ),
        "relevant_symbols": ["LT"],
    },
    {
        "category": "financials",
        "question": "Nestle India latest quarterly results",
        "ground_truth": (
            "Nestle India reported revenue of Rs.63781.8 Cr and PAT of Rs.9751.2 Cr "
            "with EBITDA margin of 24.1%."
        ),
        "relevant_symbols": ["NESTLEIND"],
    },
    {
        "category": "financials",
        "question": "Axis Bank Q1 FY2027 earnings",
        "ground_truth": (
            "Axis Bank reported Q1 FY2027 revenue of Rs.33985.63 Cr, "
            "PAT of Rs.7113.92 Cr, and EBITDA margin of 34.3%."
        ),
        "relevant_symbols": ["AXISBANK"],
    },
    {
        "category": "financials",
        "question": "NTPC quarterly results and EBITDA margin",
        "ground_truth": (
            "NTPC Limited reported Q1 FY2027 revenue of Rs.43831.86 Cr, "
            "PAT of Rs.5342.36 Cr, and EBITDA margin of 30.4%."
        ),
        "relevant_symbols": ["NTPC"],
    },
    {
        "category": "financials",
        "question": "Shoppers Stop Q4 FY2026 revenue and profit growth",
        "ground_truth": (
            "Shoppers Stop reported Q4 FY2026 revenue of Rs.1134.6 Cr (9.8% YoY growth) "
            "and PAT of Rs.68.2 Cr (34.5% growth) with EBITDA margin of 16.7%."
        ),
        "relevant_symbols": ["SHOPERSTOP"],
    },
    {
        "category": "financials",
        "question": "UltraTech Cement Q1 FY2027 results",
        "ground_truth": (
            "UltraTech Cement reported Q1 FY2027 revenue of Rs.24465 Cr and PAT of Rs.2604 Cr."
        ),
        "relevant_symbols": ["ULTRACEMCO"],
    },
    {
        "category": "financials",
        "question": "Sterlite Technologies Q1 FY2027 financials",
        "ground_truth": (
            "Sterlite Technologies Limited reported Q1 FY2027 revenue of Rs.1910 Cr "
            "and PAT of Rs.197 Cr."
        ),
        "relevant_symbols": ["STLTECH"],
    },
    {
        "category": "financials",
        "question": "Maruti Suzuki quarterly revenue and profit",
        "ground_truth": (
            "Maruti Suzuki quarterly financial results including revenue and PAT."
        ),
        "relevant_symbols": ["MARUTI"],
    },
    {
        "category": "financials",
        "question": "HDFC Bank Q1 FY2027 net profit",
        "ground_truth": (
            "HDFC Bank Q1 FY2027 financial results with net profit and revenue figures."
        ),
        "relevant_symbols": ["HDFCBANK"],
    },
    # ── ANNOUNCEMENTS (8) ────────────────────────────────────────────────────
    {
        "category": "announcements",
        "question": "Recent acquisition announcements",
        "ground_truth": (
            "Recent acquisitions include KPIT Technologies, Premier Energies, "
            "Shoppers Stop, Vikran Engineering, and Kirloskar Pneumatic in May 2026."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "announcements",
        "question": "Which companies announced dividends recently?",
        "ground_truth": (
            "Companies that announced dividends include BHEL, Tata Chemicals, "
            "Ambuja Cements, Godrej Properties, and Sobha Limited."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "announcements",
        "question": "Buyback and open offer announcements in 2026",
        "ground_truth": (
            "Tips Music Limited announced a buyback in July 2026. "
            "Bliss GVS Pharma announced a public open offer in July 2026. "
            "Neogen Chemicals announced a QIP in July 2026."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "announcements",
        "question": "Bharat Electronics BEL recent news and orders",
        "ground_truth": (
            "Bharat Electronics Limited (BEL) received a defence order worth Rs.1081 Cr in June 2026."
        ),
        "relevant_symbols": ["BEL"],
    },
    {
        "category": "announcements",
        "question": "CIPLA management changes and appointments",
        "ground_truth": (
            "CIPLA recent management change or appointment announcements."
        ),
        "relevant_symbols": ["CIPLA"],
    },
    {
        "category": "announcements",
        "question": "Credit rating revision announcements",
        "ground_truth": (
            "Credit rating revision announcements for companies on BSE/NSE."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "announcements",
        "question": "NCLT insolvency and CIRP proceedings",
        "ground_truth": (
            "Corporate insolvency resolution process (CIRP) announcements."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "announcements",
        "question": "Fund raising and preferential allotment announcements",
        "ground_truth": (
            "Fund raising and preferential issue announcements for BSE/NSE companies."
        ),
        "relevant_symbols": [],
    },
    # ── COMPOUND (4) ─────────────────────────────────────────────────────────
    {
        "category": "compound",
        "question": "Which infrastructure companies got orders above Rs.200 Cr?",
        "ground_truth": (
            "Texmaco Rail received Rs.4045 Cr and Rs.253 Cr railway orders. "
            "Transrail Lighting received Rs.600 Cr. "
            "Adani Energy Solutions received Rs.8500 Cr in the power sector."
        ),
        "relevant_symbols": ["TEXRAIL", "ADANIENSOL"],
    },
    {
        "category": "compound",
        "question": "Tata Steel results and recent announcements",
        "ground_truth": (
            "Tata Steel financial results and corporate announcements."
        ),
        "relevant_symbols": ["TATASTEEL"],
    },
    {
        "category": "compound",
        "question": "Companies with both high revenue and order wins",
        "ground_truth": (
            "Companies with strong financials and order win announcements."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "compound",
        "question": "PSU companies results and dividends",
        "ground_truth": (
            "PSU companies financial results and dividend announcements including NTPC, BPCL."
        ),
        "relevant_symbols": ["NTPC", "BPCL"],
    },
    # ── SECTOR QUERIES (6) ───────────────────────────────────────────────────
    {
        "category": "sector",
        "question": "Banking sector Q1 FY2027 results",
        "ground_truth": (
            "Banking sector quarterly results including Axis Bank."
        ),
        "relevant_symbols": ["AXISBANK", "HDFCBANK"],
    },
    {
        "category": "sector",
        "question": "Pharma sector recent news and results",
        "ground_truth": (
            "Pharma sector announcements including CIPLA quarterly results."
        ),
        "relevant_symbols": ["CIPLA"],
    },
    {
        "category": "sector",
        "question": "Solar and renewable energy order wins",
        "ground_truth": (
            "Renewable energy sector order wins including Adani Energy Solutions."
        ),
        "relevant_symbols": ["ADANIENSOL"],
    },
    {
        "category": "sector",
        "question": "FMCG companies revenue growth this quarter",
        "ground_truth": (
            "FMCG sector quarterly revenue including Nestle India."
        ),
        "relevant_symbols": ["NESTLEIND"],
    },
    {
        "category": "sector",
        "question": "IT and technology sector results",
        "ground_truth": (
            "IT sector quarterly financial results."
        ),
        "relevant_symbols": [],
    },
    {
        "category": "sector",
        "question": "Capital goods and engineering order wins",
        "ground_truth": (
            "Capital goods and engineering sector order win announcements."
        ),
        "relevant_symbols": ["BHARATFORG", "BEL"],
    },
    # ── ADVERSARIAL — should return 0 or near-0 useful results (12) ──────────
    {
        "category": "adversarial",
        "question": "What are the order wins for Fictional Corp Ltd?",
        "ground_truth": "No data available. Fictional Corp Ltd is not listed on BSE/NSE.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "XYZABC quarterly results",
        "ground_truth": "No data available for XYZABC in the database.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "Show orders received in 2050",
        "ground_truth": "No data available for the year 2050. Database covers recent announcements only.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "Apple Inc quarterly revenue and iPhone sales",
        "ground_truth": "Apple Inc is not listed on BSE or NSE. No data available.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "Tesla Model Y production numbers",
        "ground_truth": "Tesla is not listed on BSE or NSE. No relevant data available.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "What did Amazon announce last week?",
        "ground_truth": "Amazon is not listed on Indian exchanges. No relevant data available.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "Orders from January 1990",
        "ground_truth": "No data available for 1990. Database covers only recent BSE/NSE announcements.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "GameStop short squeeze analysis",
        "ground_truth": "GameStop is not listed on BSE or NSE. No data available.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "asdfjkl qwerty zxcvbn results",
        "ground_truth": "No matching records found.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "Cryptocurrency Bitcoin price movement",
        "ground_truth": "Bitcoin is not a BSE/NSE-listed equity. No relevant data available.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "US Federal Reserve interest rate decision",
        "ground_truth": "US Federal Reserve is not tracked in this BSE/NSE announcements database.",
        "relevant_symbols": [],
        "adversarial": True,
    },
    {
        "category": "adversarial",
        "question": "ZZZNOTREAL123 acquisition target",
        "ground_truth": "No data available. ZZZNOTREAL123 does not exist in the database.",
        "relevant_symbols": [],
        "adversarial": True,
    },
]


# ── pipeline runner ────────────────────────────────────────────────────────────

def _result_to_ctx(r: dict, intent: str) -> str:
    """Serialise one retrieved result dict as a short context string."""
    if intent == "order_win":
        parts = [
            f"{r.get('symbol')} — {r.get('company')}",
            f"Order value: Rs.{r.get('order_value_cr', 0):,.0f} Cr" if r.get("order_value_cr") else "",
            f"Sector: {r.get('sector_tags', '')}" if r.get("sector_tags") else "",
            f"Date: {str(r.get('broadcast_dt', ''))[:10]}",
            r.get("_snippet", "")[:200],
        ]
    elif intent == "financials":
        parts = [
            f"{r.get('symbol')} — {r.get('company')} [{r.get('period', '')}]",
            f"Revenue: Rs.{r.get('revenue_cr', 0):,.2f} Cr" if r.get("revenue_cr") else "",
            f"PAT: Rs.{r.get('pat_cr', 0):,.2f} Cr" if r.get("pat_cr") else "",
            f"Revenue growth: {r.get('revenue_growth_pct')}%" if r.get("revenue_growth_pct") else "",
            f"EBITDA margin: {r.get('ebitda_margin_pct')}%" if r.get("ebitda_margin_pct") else "",
            f"Date: {str(r.get('broadcast_dt', ''))[:10]}",
        ]
    else:
        parts = [
            f"{r.get('symbol')} — {r.get('company')}",
            f"Subject: {r.get('subject', '')}",
            f"Date: {str(r.get('broadcast_dt', ''))[:10]}",
            r.get("_snippet", "")[:200],
        ]
    return " | ".join(p for p in parts if p)


def _compute_ranking_metrics(
    results: list[dict],
    relevant_symbols: list[str],
    k: int = 5,
) -> dict[str, float] | None:
    """
    Compute Hit@k, MRR, and NDCG@k given retrieval results and a set of
    relevant symbols.  Returns None when relevant_symbols is empty (skip).

    Hit@k  = 1 if any relevant symbol appears in top-k results, else 0
    MRR    = 1/rank_of_first_relevant_result (0 if none in top-k)
    NDCG@k = normalised discounted cumulative gain using binary relevance
    """
    if not relevant_symbols:
        return None

    rel_set = {s.upper() for s in relevant_symbols}
    retrieved = [str(r.get("symbol", "")).upper() for r in results[:k]]

    # Hit@k
    hit_k = 1.0 if any(s in rel_set for s in retrieved) else 0.0

    # MRR
    mrr = 0.0
    for rank, sym in enumerate(retrieved, 1):
        if sym in rel_set:
            mrr = 1.0 / rank
            break

    # NDCG@k
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, sym in enumerate(retrieved, 1)
        if sym in rel_set
    )
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"hit_at_k": hit_k, "mrr": mrr, "ndcg_at_k": ndcg, "k": k}


def run_pipeline(
    handler: ChatHandler, question: str, n: int = 8
) -> tuple[list[str], str, list[dict]]:
    """Returns (contexts_list, answer_text, raw_results)."""
    results, intent, min_cr = handler.retrieve(question, n=n)
    contexts = [_result_to_ctx(r, intent) for r in results] or ["No matching records found."]
    answer   = "".join(handler.stream_response(question, results, intent, [], min_cr))
    return contexts, answer, results


# ── RAGAS helpers ──────────────────────────────────────────────────────────────

def _make_llm():
    """Return a LangChain ChatOllama instance, trying multiple import paths."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3:latest", temperature=0)
    except Exception:
        pass
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore[import]
        return ChatOllama(model="llama3:latest", temperature=0)
    except Exception as e:
        raise RuntimeError(f"Cannot import ChatOllama: {e}")


def _make_embeddings(model: str):
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    except Exception:
        pass
    try:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore[import]
        return OllamaEmbeddings(model=model)
    except Exception:
        return None


def _run_ragas(rows: list[dict], llm, embeddings):
    """
    Supports both ragas 0.1.x (datasets-based) and 0.2.x (EvaluationDataset-based).
    Returns a pandas DataFrame with per-question scores.
    """
    import ragas
    version = tuple(int(x) for x in ragas.__version__.split(".")[:2])

    if version >= (0, 2):
        # ── ragas 0.2.x API ───────────────────────────────────────────────
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy,
            LLMContextPrecisionWithReference, LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        lc_llm = LangchainLLMWrapper(llm)
        metrics = [
            Faithfulness(llm=lc_llm),
            LLMContextPrecisionWithReference(llm=lc_llm),
            LLMContextRecall(llm=lc_llm),
        ]
        if embeddings:
            lc_emb = LangchainEmbeddingsWrapper(embeddings)
            metrics.append(AnswerRelevancy(llm=lc_llm, embeddings=lc_emb))

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
                reference=r["ground_truth"],
            )
            for r in rows
        ]
        result = evaluate(EvaluationDataset(samples=samples), metrics=metrics)
        return result.to_pandas()

    else:
        # ── ragas 0.1.x API ───────────────────────────────────────────────
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        faithfulness.llm          = llm
        answer_relevancy.llm      = llm
        context_precision.llm     = llm
        context_recall.llm        = llm
        if embeddings:
            answer_relevancy.embeddings = embeddings

        dataset = Dataset.from_dict(
            {
                "question":     [r["question"]     for r in rows],
                "answer":       [r["answer"]       for r in rows],
                "contexts":     [r["contexts"]     for r in rows],
                "ground_truth": [r["ground_truth"] for r in rows],
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return result.to_pandas()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for EP-News")
    parser.add_argument("--n",           type=int,  default=None,
                        help="Run only the first N test cases (default: all)")
    parser.add_argument("--out",         type=str,  default="eval/results.csv",
                        help="Output CSV path (default: eval/results.csv)")
    parser.add_argument("--skip-llm",       action="store_true",
                        help="Skip RAGAS LLM scoring; only collect pipeline outputs")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only run retrieve(); skip Ollama generation and RAGAS scoring")
    parser.add_argument("--embed-model", type=str,  default="nomic-embed-text",
                        help="Ollama model for answer_relevancy embeddings")
    parser.add_argument("--category",    type=str,  default=None,
                        help="Run only cases matching this category (e.g. adversarial)")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.category:
        cases = [tc for tc in cases if tc["category"] == args.category]
    if args.n:
        cases = cases[: args.n]

    print(f"\n{'='*62}")
    print(f"  EP-News Evaluation  |  {len(cases)} questions")
    print(f"{'='*62}\n")

    # ── init pipeline ──────────────────────────────────────────────────────
    db_path     = ROOT / "data" / "ep_news.db"
    chroma_path = ROOT / "data" / "chroma"
    if not chroma_path.exists():
        chroma_path = ROOT / "data" / "chroma_db"
    handler = ChatHandler(chroma_path, db_path)
    print("Pipeline initialised.\n")

    # ── collect pipeline outputs ───────────────────────────────────────────
    rows: list[dict] = []
    for i, tc in enumerate(cases, 1):
        adv_flag = " [ADV]" if tc.get("adversarial") else ""
        print(f"[{i:02d}/{len(cases)}] {tc['category']:14s}{adv_flag}  {tc['question'][:60]}")
        t0 = time.time()
        raw_results: list[dict] = []
        try:
            if args.retrieval_only:
                raw_results, intent, min_cr = handler.retrieve(tc["question"])
                contexts = [_result_to_ctx(r, intent) for r in raw_results] or ["No results."]
                answer   = f"[retrieval-only: {len(contexts)} results, intent={intent}]"
            else:
                contexts, answer, raw_results = run_pipeline(handler, tc["question"])
            elapsed = round(time.time() - t0, 1)

            # Ranking metrics
            rank_m = _compute_ranking_metrics(raw_results, tc.get("relevant_symbols", []))
            rank_str = ""
            if rank_m:
                rank_str = (f" | Hit@{rank_m['k']}={rank_m['hit_at_k']:.0f}"
                            f" MRR={rank_m['mrr']:.2f} NDCG={rank_m['ndcg_at_k']:.2f}")

            # For adversarial cases, flag if we got too many results
            if tc.get("adversarial") and len(raw_results) > 3:
                rank_str += " [WARN: many results for adversarial query]"

            print(f"          -> {len(contexts)} ctx  |  {len(answer)} chars  |  {elapsed}s{rank_str}")
            if args.retrieval_only and contexts[0] not in ("No results.", "No matching records found."):
                for ctx in contexts[:2]:
                    print(f"             {ctx[:90]}")
        except Exception as e:
            print(f"          ERR: {e}")
            contexts, answer, rank_m = ["ERROR"], f"Pipeline error: {e}", None

        row = {
            "category":          tc["category"],
            "question":          tc["question"],
            "ground_truth":      tc["ground_truth"],
            "adversarial":       tc.get("adversarial", False),
            "relevant_symbols":  ",".join(tc.get("relevant_symbols", [])),
            "contexts":          contexts,
            "answer":            answer,
        }
        if rank_m:
            row.update({
                "hit_at_k": rank_m["hit_at_k"],
                "mrr":      rank_m["mrr"],
                "ndcg_at_k": rank_m["ndcg_at_k"],
            })
        rows.append(row)

    print()

    # ── Ranking metrics summary (no LLM needed) ───────────────────────────
    _print_ranking_summary(rows)

    if args.skip_llm or args.retrieval_only:
        if args.retrieval_only:
            print("Retrieval-only mode: skipping Ollama generation and RAGAS scoring.")
        else:
            print("Skipping RAGAS LLM scoring (--skip-llm).")
        _write_csv(rows, args.out)
        return

    # ── RAGAS scoring (skip adversarial cases — they have no ground truth) ─
    ragas_rows = [r for r in rows if not r.get("adversarial")]
    import ragas
    print(f"Running RAGAS {ragas.__version__} with llama3:latest as judge …")
    print(f"({len(ragas_rows)} non-adversarial questions, ~5-15 min)\n")

    try:
        llm        = _make_llm()
        embeddings = _make_embeddings(args.embed_model)
        if embeddings is None:
            print(f"  [warn] Embedding model '{args.embed_model}' unavailable; "
                  "answer_relevancy will be skipped.\n")

        scores_df = _run_ragas(ragas_rows, llm, embeddings)

        _score_cols = [c for c in scores_df.columns
                       if c not in ("question", "answer", "contexts", "ground_truth")]

        print("\n" + "="*62)
        print("  RAGAS Score Summary  (1.0 = perfect)")
        print("="*62)
        for col in _score_cols:
            mean_val = scores_df[col].mean()
            bar = "#" * int(mean_val * 20)
            print(f"  {col:35s}  {mean_val:.3f}  {bar}")
        print()

        # Write RAGAS scores back into the correct rows
        ragas_idx = 0
        for row in rows:
            if row.get("adversarial"):
                continue
            for col in _score_cols:
                val = scores_df.iloc[ragas_idx][col]
                row[col] = round(float(val), 4) if val == val else None  # NaN guard
            ragas_idx += 1

    except Exception as e:
        print(f"\n[error] RAGAS scoring failed: {e}")
        import traceback; traceback.print_exc()
        print("Writing pipeline outputs without RAGAS scores.")

    _write_csv(rows, args.out)


def _print_ranking_summary(rows: list[dict]) -> None:
    """Print Hit@5, MRR, NDCG@5 averages for rows that have ranking metrics."""
    ranked = [r for r in rows if "hit_at_k" in r]
    if not ranked:
        return

    avg_hit  = sum(r["hit_at_k"]  for r in ranked) / len(ranked)
    avg_mrr  = sum(r["mrr"]       for r in ranked) / len(ranked)
    avg_ndcg = sum(r["ndcg_at_k"] for r in ranked) / len(ranked)

    print("=" * 62)
    print(f"  Ranking Metrics  (over {len(ranked)} cases with known relevant symbols)")
    print("=" * 62)
    print(f"  Hit@5   {avg_hit:.3f}  {'#' * int(avg_hit  * 20)}")
    print(f"  MRR     {avg_mrr:.3f}  {'#' * int(avg_mrr  * 20)}")
    print(f"  NDCG@5  {avg_ndcg:.3f}  {'#' * int(avg_ndcg * 20)}")
    print()

    # Per-row detail
    print(f"  {'#':>2}  {'cat':12s}  {'Hit':>5}  {'MRR':>5}  {'NDCG':>5}  question")
    print("  " + "-" * 75)
    for i, row in enumerate(ranked, 1):
        q = row["question"][:45]
        print(f"  {i:>2}  {row['category']:12s}  "
              f"{row['hit_at_k']:>5.1f}  {row['mrr']:>5.2f}  {row['ndcg_at_k']:>5.2f}  {q}")
    print()


def _write_csv(rows: list[dict], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _meta = {"category", "question", "ground_truth", "contexts", "answer",
             "adversarial", "relevant_symbols"}
    score_cols = [k for k in (rows[0] if rows else {}) if k not in _meta]

    fieldnames = (
        ["category", "adversarial", "question", "ground_truth", "answer",
         "n_contexts", "relevant_symbols"]
        + score_cols
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["n_contexts"] = len(row.get("contexts", []))
            writer.writerow(out)

    print(f"Results written -> {path.resolve()}\n")

    # ASCII results table (RAGAS score columns only, max 4)
    ragas_score_cols = [c for c in score_cols
                        if c not in ("hit_at_k", "mrr", "ndcg_at_k")][:4]
    if ragas_score_cols:
        header_scores = "  ".join(f"{c[:8]:>8}" for c in ragas_score_cols)
        print(f"  {'#':>2}  {'category':14s}  {header_scores}  question")
        print("  " + "-" * (20 + 11 * len(ragas_score_cols) + 60))
        for i, row in enumerate(rows, 1):
            scores_str = "  ".join(
                f"{row[c]:>8.3f}" if isinstance(row.get(c), float) else f"{'—':>8}"
                for c in ragas_score_cols
            )
            q = row["question"][:55]
            print(f"  {i:>2}  {row['category']:14s}  {scores_str}  {q}")


if __name__ == "__main__":
    main()
