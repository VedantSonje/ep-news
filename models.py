"""
Pure data models shared across all layers (screener, storage, agent).
No logic lives here — only dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Announcement:
    """One corporate announcement row parsed from a BSE/NSE CSV."""
    symbol: str
    company: str
    subject: str
    details: str
    broadcast_dt: datetime | None
    broadcast_raw: str
    attachment: str


@dataclass
class ScoreResult:
    """Output of ScoringEngine for one announcement."""
    score: int
    tags: list[str]

    @property
    def tag_str(self) -> str:
        return ", ".join(self.tags) if self.tags else "-"


@dataclass
class ScoredAnnouncement:
    """Pairs an Announcement with its ScoreResult."""
    announcement: Announcement
    result: ScoreResult

    @property
    def score(self) -> int:
        return self.result.score

    @property
    def dt_str(self) -> str:
        dt = self.announcement.broadcast_dt
        return dt.strftime("%Y-%m-%d %H:%M") if dt else self.announcement.broadcast_raw


@dataclass
class AgentResponse:
    """Structured response from EPAgent."""
    answer: str
    tool_calls_made: int
    model_used: str
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
