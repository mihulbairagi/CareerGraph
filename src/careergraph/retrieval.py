"""Retrieval: query -> ranked, score-filtered chunks.

Thin by design, but it owns one decision that matters more than anything else
in this system: **the relevance floor**.

Vector search always returns *something*. Ask "what is his experience with
Kubernetes?" against a profile with no Kubernetes in it, and Chroma will
cheerfully hand back the five least-irrelevant chunks — probably the Docker
line and some backend work. Hand those to an LLM under a "answer the question"
instruction and it will synthesise a plausible Kubernetes story out of Docker
adjacency. That is exactly the hallucination the brief asks us to prevent.

So retrieval drops anything below ``RETRIEVAL_MIN_SCORE`` and is allowed to
return an empty list. An empty list is a *valid, meaningful result* that the
Q&A agent turns into an explicit refusal. Filtering here rather than inside
the agent keeps the rule in one place and makes it tunable from ``.env``.
"""

from __future__ import annotations

from dataclasses import dataclass

from careergraph.config import Settings
from careergraph.ingestion.embedder import EmbeddingBackend
from careergraph.ingestion.vector_store import VectorStore
from careergraph.logging_config import get_logger
from careergraph.models import RetrievedChunk, Section

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Chunks that survived the score filter, plus what was thrown away.

    ``rejected`` is kept rather than discarded because it is the difference
    between "I found nothing at all" and "I found things but none were
    relevant enough" — useful in logs and in the eval harness when tuning the
    threshold.
    """

    chunks: list[RetrievedChunk]
    rejected: list[RetrievedChunk]
    query: str

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    def to_context(self, max_chars: int = 8000) -> str:
        """Render chunks as a numbered, labelled context block for the prompt.

        Numbering is what makes citation possible: the prompt instructs the
        model to cite ``[1]``, ``[2]``, and those indices map back to
        ``self.chunks`` so we can attach real provenance to the answer instead
        of trusting the model to repeat source names correctly.
        """
        blocks: list[str] = []
        budget = max_chars
        for i, item in enumerate(self.chunks, start=1):
            block = (
                f"[{i}] (section: {item.chunk.section.value} | source: {item.chunk.title})\n"
                f"{item.chunk.text}"
            )
            if len(block) > budget:
                break
            blocks.append(block)
            budget -= len(block)
        return "\n\n".join(blocks)


class Retriever:
    """Embeds a query and returns the relevant chunks above the score floor."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingBackend,
        store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        section: Section | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self._settings.retrieval_top_k
        threshold = self._settings.retrieval_min_score if min_score is None else min_score

        query_vector = self._embedder.embed_query(query)
        candidates = self._store.search(query_vector, top_k=top_k, section=section)

        kept = [c for c in candidates if c.score >= threshold]
        rejected = [c for c in candidates if c.score < threshold]

        logger.info(
            "Retrieved chunks",
            extra={
                "query_chars": len(query),
                "candidates": len(candidates),
                "kept": len(kept),
                "rejected": len(rejected),
                "top_score": round(candidates[0].score, 4) if candidates else None,
                "threshold": threshold,
            },
        )
        if not kept and candidates:
            # The interesting case when tuning: we had results, they just
            # weren't good enough. Log it so threshold changes are informed.
            logger.info(
                "All candidates below relevance floor; agent will refuse",
                extra={"best_rejected_score": round(candidates[0].score, 4)},
            )

        return RetrievalResult(chunks=kept, rejected=rejected, query=query)

    def retrieve_profile_overview(self, query: str) -> RetrievalResult:
        """Broad retrieval for JD matching.

        Point Q&A wants precision — a few highly relevant chunks. JD matching
        wants *recall*: to say "he is missing Kubernetes" you must be confident
        Kubernetes appears nowhere in the profile, which means seeing as much
        of it as possible. So this uses a larger top_k and a lower floor, and
        accepts more marginal chunks in exchange for not inventing gaps that
        aren't real.
        """
        return self.retrieve(
            query,
            top_k=self._settings.jd_retrieval_top_k,
            # Half the normal floor: a weak match is still evidence of presence,
            # and a false "missing skill" is worse than a marginal chunk.
            min_score=self._settings.retrieval_min_score * 0.5,
        )
