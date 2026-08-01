"""Local sentence-transformers embeddings with a content-addressed disk cache.

Two things worth understanding here for interview purposes:

**1. Why local embeddings instead of an embedding API?**
Embeddings are computed on every ingestion *and* on every single query. Using
a hosted embedding API would put a network round-trip on the hot path of every
search (typically 100-300ms) and bill per token. ``all-MiniLM-L6-v2`` runs on
CPU in single-digit milliseconds for short text, costs nothing, and works
offline — which is also what makes the CI pipeline hermetic. The tradeoff is
lower embedding quality than a large hosted model, which is acceptable because
the corpus is one small, well-structured document rather than a million pages.

**2. Why cache by content hash rather than by chunk ID?**
Keying the cache on the chunk ID would be wrong: edit a bullet in your resume,
the ID stays ``experience::textr-ai::0``, and you'd serve a stale vector for
the new text — a silent correctness bug. Keying on ``sha256(text)`` makes the
cache *content-addressed*: the key changes exactly when the text changes, so a
hit is always correct by construction. Re-ingesting after editing one bullet
recomputes one vector and reads the rest from disk.

The model itself is also loaded lazily and only once per process — it is
~90MB and takes a couple of seconds to initialise, which we don't want to pay
at import time or on every request.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from careergraph.config import Settings
from careergraph.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class EmbeddingBackend(Protocol):
    """Minimal surface an embedder must expose.

    Declared as a Protocol so tests can inject a tiny deterministic fake
    instead of loading a real transformer — that alone takes the unit test
    suite from ~30s to well under a second, and keeps CI free of model
    downloads.
    """

    dimension: int

    def embed_documents(self, texts: list[str], hashes: list[str] | None = None) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class EmbeddingCache:
    """Content-addressed vector cache backed by one ``.npy`` file per hash.

    One file per vector (rather than a single pickled dict) keeps writes
    atomic-ish and lets a partially-complete cache still be useful: a crash
    mid-ingestion loses at most the vector being written, not the whole cache.
    """

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self._dir = directory
        self._enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, content_hash: str) -> Path:
        return self._dir / f"{content_hash}.npy"

    def get(self, content_hash: str) -> np.ndarray | None:
        if not self._enabled:
            return None
        path = self._path(content_hash)
        if not path.exists():
            self.misses += 1
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError):
            # A truncated file from an interrupted write. Treat as a miss and
            # let it be overwritten rather than crashing ingestion.
            logger.warning("Corrupt cache entry, recomputing", extra={"hash": content_hash[:12]})
            self.misses += 1
            return None
        self.hits += 1
        return vector

    def put(self, content_hash: str, vector: np.ndarray) -> None:
        if not self._enabled:
            return
        with self._lock:
            np.save(self._path(content_hash), vector)

    def clear(self) -> int:
        if not self._enabled or not self._dir.exists():
            return 0
        removed = 0
        for path in self._dir.glob("*.npy"):
            path.unlink()
            removed += 1
        return removed

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class SentenceTransformerEmbedder:
    """Embeds text with a local sentence-transformers model, cache in front."""

    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None) -> None:
        self._model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self.cache = cache or EmbeddingCache(
            settings.embedding_cache_dir, settings.embedding_cache_enabled
        )

    @property
    def model(self) -> SentenceTransformer:
        """Load the model on first use, not at import."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
            self._dimension = _read_dimension(self._model)
            logger.info("Embedding model ready", extra={"dimension": self._dimension})
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        assert self._dimension is not None
        return self._dimension

    def embed_documents(self, texts: list[str], hashes: list[str] | None = None) -> np.ndarray:
        """Embed a batch, computing only the vectors not already cached.

        ``hashes`` normally comes from ``Chunk.content_hash`` so the caller and
        the vector store agree on identity; it is derived here if omitted.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if hashes is None:
            from careergraph.models import Chunk

            hashes = [Chunk.hash_text(t) for t in texts]
        if len(hashes) != len(texts):
            raise ValueError("texts and hashes must be the same length")

        vectors: list[np.ndarray | None] = [self.cache.get(h) for h in hashes]
        pending = [i for i, v in enumerate(vectors) if v is None]

        if pending:
            logger.info(
                "Computing embeddings",
                extra={
                    "to_compute": len(pending),
                    "from_cache": len(texts) - len(pending),
                    "total": len(texts),
                },
            )
            computed = self.model.encode(
                [texts[i] for i in pending],
                batch_size=self._batch_size,
                # Unit-normalised vectors mean the dot product *is* cosine
                # similarity, so Chroma's cosine distance stays in [0, 2] and
                # `1 - distance` is a well-behaved similarity in [-1, 1].
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            for slot, vector in zip(pending, computed, strict=True):
                vectors[slot] = vector
                self.cache.put(hashes[slot], vector)
        else:
            logger.info("All embeddings served from cache", extra={"total": len(texts)})

        return np.vstack([v for v in vectors if v is not None])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        Deliberately *not* cached: queries are open-ended and mostly unique, so
        a query cache would grow without bound for a near-zero hit rate. Chunk
        text is the bounded, repeatedly-embedded input worth caching.
        """
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)[0]


def build_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings)


def _read_dimension(model: SentenceTransformer) -> int:
    """Read the embedding width across sentence-transformers versions.

    v5 renamed ``get_sentence_embedding_dimension`` to ``get_embedding_dimension``
    and deprecated the old name. Probing both keeps the project working on
    either version without pinning an upper bound on the dependency.
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            return int(getter())
    raise RuntimeError("Could not determine embedding dimension from the model")
