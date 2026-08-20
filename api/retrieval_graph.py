"""
LangGraph retrieval StateGraph — replaces ChatHandler.retrieve() imperative
flow with a typed state machine, giving LangSmith per-node tracing.

Graph:
  classify_params
      ↓ (route_after_classify)
  [multi_hop / breakout]  ──→ nontemporal_retrieve ──→ [needs_rerank?] ──→ rerank ──→ END
  [has temporal date]     ──→ temporal_sql ──→ [results]    ──→ END (no rerank)
                                             ──→ [empty+reflect] ──→ reflect ──→ temporal_sql (loop)
                                             ──→ [empty+guard]   ──→ END
                                             ──→ [empty]         ──→ nontemporal_retrieve
  [else]                  ──→ nontemporal_retrieve ──→ [needs_rerank?] ──→ rerank ──→ END
"""
from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

if TYPE_CHECKING:
    from api.chat_handler import ChatHandler


class RetrievalState(TypedDict):
    # ── inputs
    message:          str
    n:                int
    do_reflect:       bool
    # ── classified params (set by classify_params node)
    intent:           str
    min_cr:           float | None
    sector:           str | None
    subject:          str | None
    period_type:      str | None
    company_sym:      str | None
    from_dt:          str | None   # ISO-8601 date string or None
    to_dt:            str | None   # ISO-8601 date string or None
    is_multi_hop:     bool
    # ── working state
    results:          list[dict]
    reflection_count: int
    early_return:     bool         # temporal results → skip rerank
    needs_rerank:     bool         # nontemporal results → run rerank


def build_retrieval_graph(handler: "ChatHandler"):
    """
    Build and compile the retrieval StateGraph, binding it to a ChatHandler
    instance via closures. Call once in ChatHandler.__init__.
    """
    # Lazy import breaks the circular dependency — chat_handler imports us,
    # but by the time __init__ calls build_retrieval_graph, chat_handler is
    # fully loaded so these symbols exist.
    from api.chat_handler import (
        _detect_intent, _extract_min_cr, _extract_sector, _extract_subject,
        _extract_period_type, _extract_date_range,
        _docs_to_candidates, _candidates_to_dicts,
    )
    from retrieval.query_classifier import QueryIntent

    # ── nodes ──────────────────────────────────────────────────────────────

    def node_classify_params(state: RetrievalState) -> dict:
        """Extract all query parameters from the message. Pure, ~1 ms, no I/O."""
        msg = state["message"]

        intent      = _detect_intent(msg)
        from_dt_obj, to_dt_obj = _extract_date_range(msg)
        company_sym = handler._extract_company_symbol(msg)

        # Refine "all" → "financials" when the QueryClassifier finds financial signals
        parsed = handler._classifier.classify(msg)
        if intent == "all" and parsed.intent == QueryIntent.FINANCIAL:
            intent = "financials"

        # Multi-hop: 2+ distinct condition types and no specific company symbol
        is_multi_hop = not company_sym and handler._is_compound(msg)

        return {
            "intent":           intent,
            "min_cr":           _extract_min_cr(msg),
            "sector":           _extract_sector(msg),
            "subject":          _extract_subject(msg),
            "period_type":      _extract_period_type(msg),
            "company_sym":      company_sym,
            "from_dt":          from_dt_obj.isoformat() if from_dt_obj else None,
            "to_dt":            to_dt_obj.isoformat()   if to_dt_obj   else None,
            "is_multi_hop":     is_multi_hop,
            # Reset working state
            "results":          [],
            "reflection_count": 0,
            "early_return":     False,
            "needs_rerank":     False,
        }

    def node_temporal_sql(state: RetrievalState) -> dict:
        """
        Run temporal SQL queries using the current date-scoped params.
        Called once initially, then again after reflection with updated params.
        Returns early_return=True when results are definitive (no rerank needed).
        """
        intent      = state["intent"]
        sector      = state["sector"]
        subject     = state["subject"]
        period_type = state["period_type"]
        company_sym = state["company_sym"]
        min_cr      = state["min_cr"]

        from_dt = date.fromisoformat(state["from_dt"]) if state.get("from_dt") else None
        to_dt   = date.fromisoformat(state["to_dt"])   if state.get("to_dt")   else from_dt

        # Announcements intent → date-scoped announcement search, always exits here
        if intent == "announcements":
            rows = handler._sql_announcements_by_date(from_dt, to_dt or from_dt, subject=subject)
            return {"results": rows, "early_return": True}

        # Financials / all: try financial results table first
        fin_results: list[dict] = []
        if intent in ("financials", "all"):
            _fin_n = 50 if from_dt == (to_dt or from_dt) else 30
            fin_results = handler._sql_financials_by_date(
                from_dt, to_dt or from_dt, n=_fin_n,
                period_type=period_type, symbol=company_sym,
                sector=sector if not company_sym else None,
            )
            if fin_results:
                return {"results": fin_results, "intent": "financials", "early_return": True}

        # Order wins: try when intent is order_win, or as "all" fallback after empty financials
        if intent == "order_win" or (intent == "all" and not fin_results):
            ow_results = handler._sql_order_wins_by_date(
                from_dt, to_dt or from_dt, n=50, min_cr=min_cr,
                sector=sector if not company_sym else None,
            )
            if ow_results:
                return {"results": ow_results, "intent": "order_win", "early_return": True}

        # "all" exhausted financials + orders → announcements fallback (always exits)
        if intent == "all":
            ann = handler._sql_announcements_by_date(from_dt, to_dt or from_dt)
            return {"results": ann, "intent": "announcements", "early_return": True}

        # financials or order_win with 0 results → routing node decides: reflect or fall through
        return {"results": [], "early_return": False}

    def node_reflect(state: RetrievalState) -> dict:
        """
        Ask Ollama to suggest corrected query params after a temporal SQL miss.
        Updates sector/from_dt/to_dt/min_cr in state for the next temporal_sql run.
        """
        msg     = state["message"]
        intent  = state["intent"]
        sector  = state["sector"]
        min_cr  = state["min_cr"]
        from_dt = date.fromisoformat(state["from_dt"]) if state.get("from_dt") else None
        to_dt   = date.fromisoformat(state["to_dt"])   if state.get("to_dt")   else None

        ref = handler._reflect_params(msg, intent, sector, from_dt, to_dt, min_cr)
        new_count = state["reflection_count"] + 1

        if not ref:
            return {"reflection_count": new_count}

        updates: dict = {"reflection_count": new_count}
        if ref.get("sector"):
            updates["sector"]  = ref["sector"]
        if ref.get("from_dt"):
            updates["from_dt"] = ref["from_dt"]
        if ref.get("to_dt"):
            updates["to_dt"]   = ref["to_dt"]
        if ref.get("min_cr") is not None:
            updates["min_cr"]  = ref["min_cr"]
        return updates

    def node_nontemporal_retrieve(state: RetrievalState) -> dict:
        """
        Non-temporal retrieval: multi-hop intersection, volume breakout,
        company/sector SQL, and VectorStore hybrid (BM25 + ChromaDB) search.
        Sets needs_rerank=True for VectorStore/SQL results that need cross-encoder.
        """
        msg          = state["message"]
        intent       = state["intent"]
        sector       = state["sector"]
        subject      = state["subject"]
        company_sym  = state["company_sym"]
        min_cr       = state["min_cr"]
        n            = state["n"]
        is_multi_hop = state.get("is_multi_hop", False)

        # Convert ISO date strings back to date objects for callers that need them
        from_dt_obj = date.fromisoformat(state["from_dt"]) if state.get("from_dt") else None
        to_dt_obj   = date.fromisoformat(state["to_dt"])   if state.get("to_dt")   else None

        # Multi-hop compound query: intersect breakout ∩ order ∩ financial pools
        if is_multi_hop:
            mh_results = handler._execute_multi_hop(msg, sector, from_dt_obj, to_dt_obj, n=n)
            if mh_results:
                return {"results": mh_results, "intent": "multi_hop", "needs_rerank": False}
            # Empty multi-hop → fall through to standard non-temporal retrieval

        # Volume breakout: direct SQL into volume_breakouts table
        if intent == "volume_breakout":
            tl = set(re.split(r"\W+", msg.lower()))
            marketcap = None
            if tl & {"smallcap", "small", "smallcaps"}:    marketcap = "smallcap"
            elif tl & {"midcap", "mid", "midcaps"}:         marketcap = "midcap"
            elif tl & {"largecap", "large", "largecaps"}:   marketcap = "largecap"
            results = handler._sql_volume_breakouts(
                sector=sector, marketcap=marketcap, symbol=company_sym,
                from_dt=state.get("from_dt"),   # _sql_volume_breakouts takes ISO strings
                to_dt=state.get("to_dt"),
                n=50,
            )
            return {"results": results, "intent": "volume_breakout", "needs_rerank": False}

        # Standard non-temporal retrieval (VectorStore + direct SQL)
        results: list[dict] = []

        if intent == "order_win":
            if company_sym:
                results = handler._sql_order_wins_by_symbol(company_sym, n=n)
            if not results:
                results = handler._store.search_order_wins(msg, min_cr=min_cr, n=n)

        elif intent == "financials":
            if company_sym:
                results = handler._sql_financials_by_symbol(company_sym, n=n)
            elif sector and sector not in ("railways", "roads", "water"):
                results = handler._sql_financials_by_sector(sector, n=30)
            if not results:
                results = handler._store.search_financials(msg, n=n)

        elif intent == "announcements":
            results = handler._store.search_announcements(msg, subject=subject, n=n)

        else:  # "all"
            if company_sym:
                ann = handler._sql_announcements_by_symbol(company_sym, n=n)
                fin = handler._sql_financials_by_symbol(company_sym, n=max(3, n // 2))
                ow  = handler._sql_order_wins_by_symbol(company_sym, n=max(3, n // 2))
                combined = ann + fin + ow
                results = sorted(combined, key=lambda r: r.get("broadcast_dt", ""), reverse=True)[:n]
            else:
                fin = handler._store.search_financials(msg, n=n // 2 + 1)
                ann = handler._store.search_announcements(msg, subject=subject, n=n // 2 + 1)
                results = sorted(fin + ann, key=lambda r: r.get("_distance", 1.0))[:n]

        return {"results": results, "needs_rerank": True}

    def node_rerank(state: RetrievalState) -> dict:
        """Cross-encoder reranking via BAAI/bge-reranker-v2-m3 (loaded at startup)."""
        results = state["results"]
        if not results:
            return {}
        candidates = _docs_to_candidates(results)
        ranked, _  = handler._reranker.rerank(candidates, state["message"], top_n=state["n"])
        return {"results": _candidates_to_dicts(ranked)}

    # ── routing ────────────────────────────────────────────────────────────

    def route_after_classify(state: RetrievalState) -> str:
        # Multi-hop and volume breakout never use the temporal SQL path
        if state["is_multi_hop"] or state["intent"] == "volume_breakout":
            return "nontemporal_retrieve"
        if state.get("from_dt"):
            return "temporal_sql"
        return "nontemporal_retrieve"

    def route_after_temporal(state: RetrievalState) -> str:
        if state["results"]:
            # Temporal SQL results are already authoritative — skip rerank
            return END

        intent  = state["intent"]
        sector  = state["sector"]
        min_cr  = state["min_cr"]
        rc      = state["reflection_count"]
        do_ref  = state["do_reflect"]

        # Attempt reflection once when SQL returns 0 rows
        if do_ref and rc == 0:
            return "reflect"

        # Strict guard: explicit sector + value filter with no results →
        # return empty rather than silently stripping the filter in vector search
        if sector and min_cr and min_cr > 0 and intent == "order_win":
            return END

        # Fall through to VectorStore (clears temporal constraint in callee)
        return "nontemporal_retrieve"

    def route_after_nontemporal(state: RetrievalState) -> str:
        if state["results"] and state["needs_rerank"]:
            return "rerank"
        return END

    # ── assemble graph ─────────────────────────────────────────────────────

    g = StateGraph(RetrievalState)

    g.add_node("classify_params",      node_classify_params)
    g.add_node("temporal_sql",         node_temporal_sql)
    g.add_node("reflect",              node_reflect)
    g.add_node("nontemporal_retrieve", node_nontemporal_retrieve)
    g.add_node("rerank",               node_rerank)

    g.set_entry_point("classify_params")

    g.add_conditional_edges("classify_params", route_after_classify)
    g.add_conditional_edges("temporal_sql",    route_after_temporal)
    g.add_edge("reflect", "temporal_sql")       # loop: reflect → retry SQL
    g.add_conditional_edges("nontemporal_retrieve", route_after_nontemporal)
    g.add_edge("rerank", END)

    return g.compile()
