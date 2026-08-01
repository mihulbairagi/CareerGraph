"""Composition root: builds the object graph once, in one place.

Every heavyweight component in this system is a singleton per process — the
embedding model is ~90MB of weights, the Chroma client holds a file handle or
an HTTP pool, and the compiled LangGraph is immutable once built. Constructing
any of them per request would be a serious performance bug.

Centralising construction here (rather than letting modules instantiate their
own collaborators) is what makes the whole system testable: agents receive
their dependencies, so a test can build a ``CareerGraphApp`` with a fake
embedder and a scripted LLM and exercise the real routing logic offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from careergraph.agents.jd_matcher import JDMatcherAgent
from careergraph.agents.qa_agent import ResumeQAAgent
from careergraph.agents.router import QueryRouter
from careergraph.config import Settings, get_settings
from careergraph.ingestion.embedder import EmbeddingBackend, build_embedder
from careergraph.ingestion.vector_store import VectorStore, build_vector_store
from careergraph.llm import LLMClient, build_llm_client
from careergraph.logging_config import get_logger
from careergraph.retrieval import Retriever

logger = get_logger(__name__)


@dataclass
class CareerGraphApp:
    """Fully wired application. Everything the API layer needs."""

    settings: Settings
    embedder: EmbeddingBackend
    store: VectorStore
    retriever: Retriever
    llm: LLMClient
    qa_agent: ResumeQAAgent
    jd_agent: JDMatcherAgent
    router: QueryRouter

    def health(self) -> dict[str, object]:
        """Snapshot for ``GET /health``.

        Reports readiness rather than just liveness: an API that returns 200
        while its vector store is empty looks healthy and answers every
        question with a refusal, which is a miserable failure mode to debug.
        """
        chunk_count = self.store.count()
        return {
            "status": "ok" if chunk_count > 0 else "degraded",
            "chunks_indexed": chunk_count,
            "vector_store": self.settings.chroma_mode,
            "collection": self.settings.chroma_collection,
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.anthropic_model,
            "llm_configured": self.settings.has_api_key,
            "detail": (
                None
                if chunk_count > 0
                else "Vector store is empty. Run `careergraph-ingest` to index the profile."
            ),
        }


def build_app(
    settings: Settings | None = None,
    embedder: EmbeddingBackend | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
) -> CareerGraphApp:
    """Assemble the object graph. Overrides exist purely for tests."""
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = store or build_vector_store(settings)
    llm = llm or build_llm_client(settings)

    retriever = Retriever(settings, embedder, store)
    qa_agent = ResumeQAAgent(settings, retriever, llm)
    jd_agent = JDMatcherAgent(settings, retriever, llm)
    router = QueryRouter(settings, qa_agent, jd_agent, llm)

    logger.info(
        "Application wired",
        extra={
            "chroma_mode": settings.chroma_mode,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.anthropic_model,
            "llm_configured": settings.has_api_key,
        },
    )
    return CareerGraphApp(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        llm=llm,
        qa_agent=qa_agent,
        jd_agent=jd_agent,
        router=router,
    )


@lru_cache(maxsize=1)
def get_app() -> CareerGraphApp:
    """Process-wide singleton, used as the FastAPI dependency."""
    return build_app()
