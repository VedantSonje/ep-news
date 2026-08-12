"""
Agent package — two Claude-powered agents.

EPAgent          : tool-use loop (4 tools, up to 10 iterations)
RetrievalAgent   : deterministic retrieval + single LLM explain call (faster, cheaper)
"""
from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from agent.ep_agent import EPAgent
from agent.retrieval_agent import RetrievalAgent, RetrievalResponse

__all__ = ["TOOL_DEFINITIONS", "ToolExecutor", "EPAgent", "RetrievalAgent", "RetrievalResponse"]
