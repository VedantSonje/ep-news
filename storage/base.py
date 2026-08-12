"""Abstract base class shared by all storage backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from models import ScoredAnnouncement


class BaseStorage(ABC):
    """
    Contract every storage backend must satisfy.
    Concrete implementations: SQLiteStorage, ChromaDBStorage.
    """

    @abstractmethod
    def save(self, items: list[ScoredAnnouncement], source_file: str = "") -> int:
        """Persist a batch of scored announcements. Returns count saved."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored announcements."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources (connections, file handles, etc.)."""

    def __enter__(self) -> BaseStorage:
        return self

    def __exit__(self, *_) -> None:
        self.close()
