"""
ToolDefinitions — Claude-compatible tool schemas (JSON dicts).
ToolExecutor    — executes a tool call by name, returns string result.

Two tool types:
  query_sql              — structured query against SQLiteStorage
  semantic_search        — natural language search against ChromaDBStorage
  get_top_ep_candidates  — shortcut: top-N by score
  get_company_announcements — all filings for one symbol
"""
from __future__ import annotations

import json
from typing import Any

from storage.database_manager import DatabaseManager


# ── Tool schema definitions (passed to Claude's `tools` parameter) ────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_sql",
        "description": (
            "Execute a SELECT query against the SQLite announcements database. "
            "Use for precise filtering by score, symbol, subject, date range, or any combination. "
            "Table: announcements(id, symbol, company, subject, details, score, tags, "
            "broadcast_dt, attachment, source_file, ingested_at). "
            "Score range: 1-13 (higher = stronger catalyst). "
            "Always use LIMIT to avoid large result sets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A valid SQLite SELECT statement. "
                        "Example: SELECT symbol, company, score, tags "
                        "FROM announcements WHERE score >= 9 ORDER BY score DESC LIMIT 10"
                    ),
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Search announcements using natural language semantic similarity. "
            "Best for conceptual queries like 'companies winning government contracts', "
            "'pharma drug approvals', 'renewable energy commissioning', or 'CEO resignations'. "
            "Returns results ranked by embedding similarity distance (lower = more similar)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query describing what you are looking for.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_top_ep_candidates",
        "description": (
            "Get the highest-scoring EP (Episodic Pivot) candidates, "
            "ranked by catalyst strength. Use this first when the user asks for "
            "'best stocks', 'top picks', or 'what to watch today'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "integer",
                    "description": "Minimum catalyst score threshold (default: 8).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_company_announcements",
        "description": (
            "Get all announcements for a specific company using its NSE/BSE stock symbol. "
            "Use when the user asks about a specific company like 'what happened with HFCL' "
            "or 'show me all TATACHEM news'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE/BSE stock symbol in uppercase (e.g. TATACHEM, HFCL, AFCONS).",
                }
            },
            "required": ["symbol"],
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes a tool call by name against the DatabaseManager.
    Returns a JSON string to send back as the tool_result content.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch to the correct backend method and serialise the result."""
        try:
            if tool_name == "query_sql":
                rows = self._db.sql_query(tool_input["sql"])
                return self._fmt(rows)

            if tool_name == "semantic_search":
                rows = self._db.semantic_search(
                    query=tool_input["query"],
                    n_results=min(int(tool_input.get("n_results", 5)), 20),
                )
                return self._fmt(rows)

            if tool_name == "get_top_ep_candidates":
                rows = self._db.get_top_ep_candidates(
                    min_score=int(tool_input.get("min_score", 8)),
                    limit=int(tool_input.get("limit", 10)),
                )
                return self._fmt(rows)

            if tool_name == "get_company_announcements":
                rows = self._db.get_company_announcements(tool_input["symbol"])
                return self._fmt(rows)

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @staticmethod
    def _fmt(rows: list[dict]) -> str:
        if not rows:
            return json.dumps({"result": "No data found.", "count": 0})
        return json.dumps({"result": rows, "count": len(rows)}, default=str, indent=2)
