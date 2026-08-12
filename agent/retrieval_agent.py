"""
RetrievalAgent — candidate-first, explain-second architecture.

Flow:
  1. QueryClassifier   → deterministic intent + filters       (logged)
  2. RetrievalBroker   → SQL / BM25 / hybrid / financial      (logged per source)
  3. SQL enrichment    → sector, novelty, materiality per doc (logged)
  4. Reranker          → proxy scoring with boost breakdown   (logged)
  5. Single LLM call   → per-stock "why included / why excluded" explanation

RetrievalResponse includes:
  - answer          — the LLM explanation
  - trace           — PipelineTrace with every stage's hit list
  - candidates_used — number of docs sent to LLM
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import anthropic

from retrieval.query_classifier import QueryClassifier
from retrieval.retrieval_broker import RetrievalBroker, CandidateDoc
from retrieval.pipeline_logger import PipelineTrace


_EXPLAIN_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS    = 1500

_SYSTEM_PROMPT = """\
You are an expert Indian equity analyst specialising in EP (Episodic Pivot) momentum setups.

You receive a ranked list of stock announcements. Each entry includes:
  - Rank, symbol, company, subject, EP score (1-13), broadcast date
  - Sources that retrieved it (sql / bm25 / vector)
  - Evidence: sector tags, novelty flag, order size, catalyst type,
    government linkage, export component, retrieval boost breakdown

Your task — for each candidate write one concise paragraph:
  WHY INCLUDED: What makes this an EP candidate? Cite order size (if any),
    sector, government/export angle, catalyst type, EP score.
  CONCERN (if any): Flag if catalyst is small, repeat, or sector doesn't fit.

After all candidates, write a single "EP Watch" line: the single strongest
setup and the one-sentence reason why.

Format: plain text, no markdown tables. Maximum 500 words total.
"""

_SYSTEM_BLOCK = [
    {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
]


@dataclass
class RetrievalResponse:
    answer:          str
    intent:          str
    candidates_used: int
    trace:           PipelineTrace | None = None
    query_filters:   dict  = field(default_factory=dict)
    cache_read:      int   = 0
    cache_write:     int   = 0


def _evidence_line(c: CandidateDoc) -> str:
    """Format per-candidate evidence as a compact string for the LLM."""
    ev = c.evidence
    parts = []

    # Materiality
    val = ev.get("order_value_cr")
    size = ev.get("relative_size", "")
    if val:
        parts.append(f"order=Rs.{val:.0f}cr({size})")
    elif size and size != "unknown":
        parts.append(f"size={size}")

    # Catalyst & sector
    cat = ev.get("catalyst_type", "")
    if cat and cat != "general":
        parts.append(f"catalyst={cat}")
    sectors = ev.get("sector_tags", "")
    if sectors:
        parts.append(f"sectors=[{sectors}]")

    # Flags
    if ev.get("government_linked"):
        parts.append("GOV")
    if ev.get("export_component"):
        parts.append("EXPORT")
    if not ev.get("is_novel", True):
        parts.append("REPEAT")

    # Retrieval evidence
    bm25_r  = ev.get("bm25_rank")
    vec_r   = ev.get("vector_rank")
    src_str = "+".join(c.sources)
    ret_parts = [f"src={src_str}"]
    if bm25_r is not None:
        ret_parts.append(f"bm25#{bm25_r+1}")
    if vec_r is not None:
        ret_parts.append(f"vec#{vec_r+1}")
    ret_parts.append(f"rrf={c.rrf_score:.4f}")

    kw = ev.get("keyword_boost", 0)
    sc = ev.get("score_boost", 0)
    re = ev.get("recency_boost", 0)
    ret_parts.append(f"boosts(kw={kw:.3f},sc={sc:.3f},rec={re:.3f})")

    parts += ret_parts
    return " | ".join(parts)


def _format_candidates(candidates: list[CandidateDoc]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        subj  = c.subject[:50]
        age   = c.evidence.get("age_days", "?")
        novel = "" if c.evidence.get("is_novel", True) else "  [REPEAT FILING]"
        lines.append(
            f"\n--- Candidate #{i}: {c.symbol}  ({c.company}) ---\n"
            f"  Subject   : {subj}{novel}\n"
            f"  EP Score  : {c.score}/13\n"
            f"  Date      : {c.broadcast_dt[:10]}  ({age} days ago)\n"
            f"  Evidence  : {_evidence_line(c)}"
        )
    return "\n".join(lines)


class RetrievalAgent:
    """
    New-generation EP agent: deterministic retrieval → single LLM explanation.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        vector_store,
        fin_storage=None,
        api_key: str | None = None,
    ) -> None:
        self._classifier = QueryClassifier()
        self._broker     = RetrievalBroker(conn, vector_store, fin_storage)
        self._client     = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def ask(self, question: str, top_n: int = 10) -> RetrievalResponse:
        # ── 1. Classify + retrieve + rerank (with full trace) ────────────
        parsed              = self._classifier.classify(question)
        candidates, trace   = self._broker.retrieve_with_trace(parsed, top_n=top_n)

        if not candidates:
            trace.explanation = "No matching announcements found."
            trace.stop()
            return RetrievalResponse(
                answer         = "No matching announcements found for your query.",
                intent         = parsed.intent.value,
                candidates_used= 0,
                trace          = trace,
                query_filters  = _parsed_filters(parsed),
            )

        # ── 2. Single LLM call ───────────────────────────────────────────
        candidate_block = _format_candidates(candidates)
        user_msg = (
            f"User query: {question!r}\n\n"
            f"Routing: intent={parsed.intent.value}  themes={parsed.themes or 'none'}  "
            f"symbols={parsed.symbols or 'any'}  date={parsed.date_filter or 'any'}\n\n"
            f"Ranked candidates ({len(candidates)}):\n{candidate_block}\n\n"
            "Please provide your analysis with 'WHY INCLUDED' and any 'CONCERN' "
            "for each candidate, then the EP Watch summary."
        )

        response = self._client.messages.create(
            model      = _EXPLAIN_MODEL,
            max_tokens = _MAX_TOKENS,
            system     = _SYSTEM_BLOCK,
            messages   = [{"role": "user", "content": user_msg}],
        )

        answer = response.content[0].text if response.content else ""
        usage  = response.usage
        trace.explanation = answer
        trace.stop()

        return RetrievalResponse(
            answer          = answer,
            intent          = parsed.intent.value,
            candidates_used = len(candidates),
            trace           = trace,
            query_filters   = _parsed_filters(parsed),
            cache_read      = getattr(usage, "cache_read_input_tokens", 0),
            cache_write     = getattr(usage, "cache_creation_input_tokens", 0),
        )

    def candidates_only(self, question: str, top_n: int = 20) -> list[CandidateDoc]:
        """Ranked candidates without the LLM explanation."""
        parsed = self._classifier.classify(question)
        return self._broker.retrieve(parsed, top_n=top_n)

    def trace_only(self, question: str, top_n: int = 10) -> PipelineTrace:
        """Full pipeline trace without the LLM call. Useful for debugging."""
        parsed = self._classifier.classify(question)
        _, trace = self._broker.retrieve_with_trace(parsed, top_n=top_n)
        return trace


def _parsed_filters(parsed) -> dict:
    return {
        "intent":      parsed.intent.value,
        "symbols":     parsed.symbols,
        "date_filter": parsed.date_filter,
        "min_score":   parsed.min_score,
        "themes":      parsed.themes,
    }
