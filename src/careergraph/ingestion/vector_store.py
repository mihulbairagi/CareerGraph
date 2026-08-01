"""ChromaDB wrapper for storing and searching profile chunks.

Two deployment shapes, one interface:

* ``CHROMA_MODE=persistent`` — embedded Chroma writing to ``data/chroma``.
  Zero infrastructure, which is what you want for local dev, for CI, and for
  anyone who clones the repo and wants it working in one command.
* ``CHROMA_MODE=http`` — talks to the ``chromadb`` service in
  ``docker-compose.yml``. This is the shape that actually scales: the API
  container becomes stateless and can be replicated, with vector state living
  in one service behind a volume.

The same class serves both, so switching is a single env var and no code
change — that separation is the whole point of putting a wrapper here rather
than sprinkling ``chromadb.Client()`` calls through the agents.

We also pass embeddings in explicitly rather than letting Chroma call an
embedding function for us. That keeps a single embedding code path (the cached
one in ``embedder.py``) for both ingestion and query, which is what guarantees
documents and queries land in the same vector space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from careergraph.config import Settings
from careergraph.logging_config import get_logger
from careergraph.models import Chunk, RetrievedChunk, Section

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)

# Chroma needs to be told which distance metric to build the index with.
# Cosine is correct for normalised sentence-transformer embeddings; the
# default (L2) would give subtly worse ranking for the same vectors.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class VectorStore:
    """Persistence + semantic search over profile chunks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._collection: Any | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            import chromadb
            from chromadb.config import DEFAULT_TENANT
            from chromadb.config import Settings as ChromaSettings

            if self._settings.chroma_mode == "http":
                logger.info(
                    "Connecting to Chroma over HTTP",
                    extra={
                        "host": self._settings.chroma_host,
                        "port": self._settings.chroma_port,
                    },
                )
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                    tenant=DEFAULT_TENANT,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            else:
                path = self._settings.chroma_persist_dir
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Opening persistent Chroma store", extra={"path": str(path)})
                self._client = chromadb.PersistentClient(
                    path=str(path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._settings.chroma_collection,
                metadata=_COLLECTION_METADATA,
            )
        return self._collection

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or replace chunks by ID.

        ``upsert`` rather than ``add`` so re-running ingestion is idempotent —
        running it twice must not produce duplicate chunks that then crowd out
        each other's neighbours in the top-k.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}")

        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[c.to_metadata() for c in chunks],
        )
        logger.info("Upserted chunks", extra={"count": len(chunks)})

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)
        logger.info("Deleted stale chunks", extra={"count": len(ids)})

    def reset(self) -> None:
        """Drop the collection entirely. Used by ``ingest --rebuild``."""
        try:
            self.client.delete_collection(self._settings.chroma_collection)
            logger.info("Dropped collection", extra={"name": self._settings.chroma_collection})
        except Exception as exc:  # collection may simply not exist yet
            logger.debug("Collection drop skipped", extra={"reason": str(exc)})
        self._collection = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def existing_hashes(self) -> dict[str, str]:
        """Map of ``chunk_id -> content_hash`` for everything already stored.

        This is what lets ingestion diff old against new and skip unchanged
        chunks entirely, instead of blindly rewriting the whole collection.
        """
        result = self.collection.get(include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        return {
            chunk_id: (meta or {}).get("content_hash", "")
            for chunk_id, meta in zip(ids, metadatas, strict=False)
        }

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            logger.warning("Could not count collection", extra={"error": str(exc)})
            return 0

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        section: Section | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks, most similar first."""
        available = self.count()
        if available == 0:
            logger.warning("Search against an empty collection; run ingestion first")
            return []

        where = {"section": section.value} if section else None
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            # Asking for more than exists makes some Chroma versions error out.
            n_results=min(top_k, available),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_retrieved(result)

    @staticmethod
    def _to_retrieved(result: dict[str, Any]) -> list[RetrievedChunk]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            meta = meta or {}
            chunk = Chunk(
                id=chunk_id,
                text=text or "",
                section=Section(meta.get("section", Section.ABOUT.value)),
                title=meta.get("title", chunk_id),
                content_hash=meta.get("content_hash", ""),
            )
            # Chroma returns cosine *distance* in [0, 2]; similarity is 1 - d.
            # Clamped because floating-point error can nudge it a hair outside
            # the range, which would fail the Pydantic bounds on the model.
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))
        return retrieved


def build_vector_store(settings: Settings) -> VectorStore:
    return VectorStore(settings)
