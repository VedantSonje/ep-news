"""
EPAgent — Claude-powered AI agent for EP screening queries.

Architecture
------------
- Uses the Anthropic Python SDK with the manual agentic loop pattern.
- System prompt is cached with cache_control (prompt caching) to reduce
  cost on repeated queries since the system prompt is large and static.
- Tools: query_sql, semantic_search, get_top_ep_candidates, get_company_announcements
- Max 10 tool-call iterations per question (prevents infinite loops).
- Model: claude-haiku-4-5 (configurable via AppConfig)
"""
from __future__ import annotations

from typing import Any

import anthropic

from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from models import AgentResponse
from storage.database_manager import DatabaseManager


# ── Cached system prompt ──────────────────────────────────────────────────
# NOTE: cache_control on the system prompt reduces costs when the same
# agent receives many queries (TTL = 5 minutes by default).
# Haiku 4.5 requires ≥ 4096 tokens for a cache hit; the detailed prompt
# below approaches that threshold. Extend it for better cache economics.

_SYSTEM_PROMPT = """\
You are an expert EP (Episodic Pivot) Screener Agent for Indian equity markets (BSE/NSE).
Your job is to help traders identify high-catalyst stocks using the Stockbee EP framework.

WHAT IS AN EP?
An Episodic Pivot is a stock that experiences a sudden, large catalyst event — earnings
surprise, major order win, plant commissioning, acquisition, regulatory approval, or
government contract — that causes a fundamental shift in its business outlook. Traders
look for price gap-ups with 2-3× volume expansion through a clear pivot level.

YOUR DATABASES
You have two databases to query:

1. SQLite (structured SQL):
   Table: announcements(id, symbol, company, subject, details, score, tags,
                        broadcast_dt, attachment, source_file, ingested_at)
   Use for: filtering by score, symbol, date range, subject, count, aggregation.

2. ChromaDB (vector / semantic search):
   Use for: natural language queries like "pharma approvals", "defence contracts",
   "plant commissioning", "government orders", "CEO resignation".

SCORING SCALE (1 – 13)
  13      : Massive order win ≥ Rs.1000 crore
  11-12   : Large order win Rs.500-999 crore / Rs.100-499 crore
  10      : Order/contract win (value unknown)
  9       : Acquisition, plant commissioning
  8       : Board financial results, capacity addition, scheme of arrangement
  7       : High-value general update / large sale
  5-6     : Dividend ≥ Rs.5/share, C-suite management change
  4       : Regular dividend Rs.1-4/share
  1-3     : Routine items (noise — already filtered out during ingest)

EP TRADING RULES (context only — not confirmed by the data)
  - Trade only when price shows gap-up or strong range expansion
  - Volume must be 2-3× the 50-day average
  - Price must move through a clear pivot or ATH / high-tight area
  - Catalyst score ≥ 8 is minimum consideration; ≥ 10 is ideal

RESPONSE FORMAT
  - Always lead with the symbol and score
  - Explain WHY the catalyst is significant
  - Group results by score tier when listing multiple stocks
  - Flag any C-suite changes or litigation (risk signals)
  - Keep answers concise and actionable

HOW TO USE YOUR TOOLS
  - Start with get_top_ep_candidates for general "best stocks" queries
  - Use semantic_search for thematic queries ("defence", "pharma", "energy")
  - Use query_sql for precise filtering (score range, specific date, sector aggregation)
  - Use get_company_announcements when asked about a specific company
  - Combine tools freely — multiple tool calls per question are normal
"""


class EPAgent:
    """
    Claude-powered conversational agent that queries both databases
    and returns synthesised, actionable EP screening insights.

    Usage
    -----
        agent = EPAgent(config, db_manager)
        response = agent.ask("What are the top EP candidates today?")
        print(response.answer)
    """

    _MAX_ITERATIONS = 10  # prevents infinite tool loops

    def __init__(self, api_key: str, model: str, db: DatabaseManager) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._executor = ToolExecutor(db)

        # System prompt with cache_control for prompt caching.
        # The cache is keyed on the bytes of the rendered prefix; since this
        # system prompt is static, subsequent ask() calls reuse the same cache.
        self._system: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ── public ────────────────────────────────────────────────────────────

    def ask(self, question: str) -> AgentResponse:
        """
        Send a question to Claude, execute tool calls as needed,
        and return the final answer as an AgentResponse.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": question}
        ]

        tool_calls_made = 0
        cache_read = 0
        cache_write = 0

        for _ in range(self._MAX_ITERATIONS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=self._system,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Accumulate cache token counts across iterations
            usage = response.usage
            cache_read  += getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0

            if response.stop_reason == "end_turn":
                answer = self._extract_text(response.content)
                return AgentResponse(
                    answer=answer,
                    tool_calls_made=tool_calls_made,
                    model_used=response.model,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )

            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(response.content)
                tool_calls_made += len(tool_results)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — return whatever text we have
            break

        answer = self._extract_text(response.content) or "Agent reached iteration limit."
        return AgentResponse(
            answer=answer,
            tool_calls_made=tool_calls_made,
            model_used=getattr(response, "model", self._model),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

    # ── private ───────────────────────────────────────────────────────────

    def _execute_tools(
        self, content_blocks: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute every tool_use block and collect tool_result dicts."""
        results = []
        for block in content_blocks:
            if block.type != "tool_use":
                continue
            output = self._executor.execute(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        return results

    @staticmethod
    def _extract_text(content_blocks: list[Any]) -> str:
        return "\n".join(
            b.text for b in content_blocks if b.type == "text"
        ).strip()
