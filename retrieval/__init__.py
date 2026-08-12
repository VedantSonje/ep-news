"""
Retrieval package — deterministic candidate generation layer.

QueryClassifier   : intent detection + structured filter extraction
BM25Search        : SQLite FTS5 / BM25 keyword search
HybridSearch      : BM25 + ChromaDB vector + RRF fusion
Reranker          : proxy reranking (keyword overlap + score + recency)
RetrievalBroker   : routes query to correct strategy, applies reranker
PipelineTrace     : per-stage logging (classifier → sql/bm25/vector → fused → reranked)
"""
from retrieval.query_classifier import QueryClassifier, QueryIntent, ParsedQuery
from retrieval.bm25_search import BM25Search, BM25Hit, setup_fts
from retrieval.hybrid_search import HybridSearch, CandidateDoc
from retrieval.reranker import Reranker
from retrieval.retrieval_broker import RetrievalBroker
from retrieval.pipeline_logger import PipelineTrace, StageLog, HitSnapshot

__all__ = [
    "QueryClassifier", "QueryIntent", "ParsedQuery",
    "BM25Search", "BM25Hit", "setup_fts",
    "HybridSearch", "CandidateDoc",
    "Reranker",
    "RetrievalBroker",
    "PipelineTrace", "StageLog", "HitSnapshot",
]
