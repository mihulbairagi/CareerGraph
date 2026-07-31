"""Centralised, validated configuration.

Everything tunable lives here so that no module ever reads ``os.environ``
directly and no secret is ever hardcoded. ``pydantic-settings`` gives us
type coercion and fail-fast validation at import time: a bad
``RETRIEVAL_TOP_K=banana`` crashes on startup with a clear message instead
of surfacing as a mysterious ``TypeError`` deep inside the retriever.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three parents up from src/careergraph/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, sourced from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Secrets
    # ------------------------------------------------------------------
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key. Wrapped in SecretStr so it never leaks into "
        "logs, tracebacks or `repr(settings)`.",
    )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = Field(default=1024, gt=0, le=8192)
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0.0 by default: for a grounded RAG system we want reproducible, "
        "extractive answers, not creative ones.",
    )
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local CPU embedding model. Free, no API round-trip, 384-dim.",
    )
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_cache_dir: Path = PROJECT_ROOT / ".cache" / "embeddings"
    embedding_cache_enabled: bool = True

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    chroma_mode: Literal["persistent", "http"] = Field(
        default="persistent",
        description="'persistent' = embedded on-disk Chroma (local dev, CI). "
        "'http' = talk to the chromadb service in docker-compose.",
    )
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "careergraph_profile"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = Field(default=5, gt=0, le=50)
    retrieval_min_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity floor. Chunks below this are dropped before "
        "they reach the LLM; if *everything* is dropped the Q&A agent refuses "
        "instead of guessing. This is the primary anti-hallucination lever.",
    )
    jd_retrieval_top_k: int = Field(
        default=12,
        gt=0,
        le=50,
        description="JD matching needs a broad view of the whole profile, not just "
        "the top few chunks, so it retrieves more than point Q&A.",
    )

    # ------------------------------------------------------------------
    # Data + chunking
    # ------------------------------------------------------------------
    profile_path: Path = PROJECT_ROOT / "data" / "profile.json"
    chunk_max_chars: int = Field(default=1200, gt=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - containers must bind all interfaces
    api_port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="'json' emits one JSON object per line for log aggregators.",
    )

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        max_chars = info.data.get("chunk_max_chars")
        if max_chars is not None and v >= max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return v

    @property
    def has_api_key(self) -> bool:
        """True when a non-empty Anthropic key is configured.

        Lets retrieval-only code paths (ingestion, tests, ``/health``) run
        without a key, while generation paths fail loudly.
        """
        return bool(self.anthropic_api_key.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached because ``Settings()`` re-reads and re-parses ``.env`` on every
    instantiation. The cache also makes it a clean FastAPI dependency and
    lets tests swap config via ``get_settings.cache_clear()``.
    """
    return Settings()
