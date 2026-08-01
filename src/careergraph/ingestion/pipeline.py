"""Ingestion pipeline: ``profile.json`` -> chunks -> embeddings -> ChromaDB.

The pipeline is *incremental by default*. It diffs the content hashes already
in the collection against the hashes of the freshly-chunked profile and only
touches what actually changed:

* new or edited chunk  -> embed (cache miss) and upsert
* unchanged chunk      -> skipped entirely, no embedding, no write
* chunk that vanished  -> deleted, so removing a job from your resume actually
  removes it from the index instead of leaving a ghost that keeps getting
  retrieved

That last case is the one naive pipelines get wrong: they upsert the new state
and never delete, so the vector store slowly fills with content the source
document no longer contains — and the LLM cites it as fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from careergraph.config import Settings
from careergraph.ingestion.chunker import ProfileChunker
from careergraph.ingestion.embedder import SentenceTransformerEmbedder, build_embedder
from careergraph.ingestion.vector_store import VectorStore, build_vector_store
from careergraph.logging_config import get_logger
from careergraph.models import Chunk

logger = get_logger(__name__)


@dataclass
class IngestionReport:
    """What the pipeline actually did. Returned so callers can assert on it."""

    total_chunks: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    embeddings_computed: int = 0
    embeddings_cached: int = 0
    sections: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.total_chunks} chunks "
            f"({self.added} added, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.deleted} deleted); "
            f"embeddings: {self.embeddings_computed} computed, "
            f"{self.embeddings_cached} from cache"
        )


def load_profile(path: Path) -> dict[str, Any]:
    """Read and validate the profile document."""
    if not path.exists():
        raise FileNotFoundError(
            f"Profile not found at {path}. Copy data/profile.json and fill it in."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # A trailing comma in hand-edited JSON is the single most common
        # failure here, so point at the exact line rather than dumping a stack.
        raise ValueError(f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return data


class IngestionPipeline:
    """Orchestrates chunking, embedding and persistence."""

    def __init__(
        self,
        settings: Settings,
        embedder: SentenceTransformerEmbedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._settings = settings
        self._chunker = ProfileChunker(settings)
        self._embedder = embedder or build_embedder(settings)
        self._store = store or build_vector_store(settings)

    def run(self, profile_path: Path | None = None, rebuild: bool = False) -> IngestionReport:
        path = profile_path or self._settings.profile_path
        profile = load_profile(path)
        chunks = self._chunker.chunk(profile)

        if rebuild:
            logger.info("Rebuild requested; dropping existing collection")
            self._store.reset()

        existing = {} if rebuild else self._store.existing_hashes()
        report = IngestionReport(total_chunks=len(chunks))
        for chunk in chunks:
            report.sections[chunk.section.value] = report.sections.get(chunk.section.value, 0) + 1

        to_write: list[Chunk] = []
        for chunk in chunks:
            previous = existing.get(chunk.id)
            if previous is None:
                report.added += 1
                to_write.append(chunk)
            elif previous != chunk.content_hash:
                report.updated += 1
                to_write.append(chunk)
            else:
                report.unchanged += 1

        # Anything in the store that the profile no longer produces is stale.
        stale = [cid for cid in existing if cid not in {c.id for c in chunks}]
        if stale:
            self._store.delete_ids(stale)
            report.deleted = len(stale)

        if to_write:
            before = self._embedder.cache.stats
            vectors = self._embedder.embed_documents(
                [c.text for c in to_write], [c.content_hash for c in to_write]
            )
            after = self._embedder.cache.stats
            report.embeddings_cached = after["hits"] - before["hits"]
            report.embeddings_computed = len(to_write) - report.embeddings_cached
            self._store.upsert(to_write, vectors)
        else:
            logger.info("Nothing to write; profile is already fully ingested")

        logger.info("Ingestion complete", extra={"report": report.summary()})
        return report
