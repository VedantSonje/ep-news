"""
Application configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; user can export vars manually


@dataclass
class AppConfig:
    """
    Central configuration object.  All paths and credentials live here.
    Create once via AppConfig.from_env() and pass into every component.
    """
    data_dir: Path
    db_path: Path
    chroma_path: Path
    model: str
    min_score: int
    anthropic_api_key: str

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config from environment variables with sensible defaults."""
        data_dir = Path(os.getenv("EP_DATA_DIR", "./data"))
        return cls(
            data_dir=data_dir,
            db_path=Path(os.getenv("EP_DB_PATH", str(data_dir / "ep_news.db"))),
            chroma_path=Path(os.getenv("EP_CHROMA_PATH", str(data_dir / "chroma"))),
            model=os.getenv("EP_MODEL", "claude-haiku-4-5"),
            min_score=int(os.getenv("EP_MIN_SCORE", "4")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
