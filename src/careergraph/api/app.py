"""FastAPI application.

Structure notes:

* **Lifespan, not ``@app.on_event``.** The object graph (embedding model,
  Chroma client, compiled LangGraph) is built once at startup and torn down
  cleanly. Building it per request would reload ~90MB of model weights every
  time; building it at import time would make the module un-importable without
  a working Chroma directory, which breaks tooling and tests.

* **Exceptions map to HTTP semantics.** A refusal is *not* an error — it is a
  200 with ``grounded: false``, because the system correctly determined it
  could not answer. Real errors are separated by who is at fault: bad input is
  422, an unavailable model is 503, an unindexed store is 409. Returning 500
  for everything throws away all of that.

* **The composition root is a dependency.** Endpoints receive the wired app
  via ``Depends``, so tests can override it with fakes and never touch the
  network.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from careergraph import __version__
from careergraph.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    MatchRequest,
    MatchResponse,
    QueryRequest,
    RoutePreviewResponse,
    RouteResponse,
    SearchResponse,
    SearchResultItem,
)
from careergraph.config import get_settings
from careergraph.dependencies import CareerGraphApp, get_app
from careergraph.llm import LLMError
from careergraph.logging_config import configure_logging, get_logger
from careergraph.models import Intent

logger = get_logger(__name__)

DESCRIPTION = """
Multi-agent RAG over a single professional profile.

**Two agents behind one router:**

* **Resume Q&A** — answers questions strictly from indexed profile content.
  When the profile does not contain the answer it says so rather than guessing.
* **JD Matcher** — scores a pasted job description against the profile and
  reports the specific skills and ATS keywords that are missing.

**Grounding.** Retrieved chunks below a cosine-similarity floor are discarded.
If nothing survives, the language model is never called at all — the API
returns a refusal with `grounded: false` and HTTP 200. A refusal is a correct
outcome, not a failure.

**Getting started.** Run `careergraph-ingest` before querying; `GET /health`
reports `degraded` until the profile is indexed.
"""

TAGS_METADATA = [
    {"name": "query", "description": "Router-dispatched entrypoint. Start here."},
    {"name": "agents", "description": "Call a specific agent directly, bypassing the router."},
    {"name": "diagnostics", "description": "Health, raw retrieval, and graph introspection."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - signature fixed by FastAPI
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info("Starting CareerGraph API", extra={"version": __version__})

    wired = get_app()
    health = wired.health()
    if health["status"] != "ok":
        # Warn loudly but still start: a running API that reports 'degraded' is
        # far easier to diagnose than a container that crash-loops at boot.
        logger.warning("Starting with an empty vector store", extra={"detail": health["detail"]})

    yield

    logger.info("Shutting down CareerGraph API")
    get_app.cache_clear()


app = FastAPI(
    title="CareerGraph",
    version=__version__,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

# Open CORS because this is a public, read-only, unauthenticated portfolio API
# meant to be called from a browser frontend. It exposes no mutating routes and
# no private data beyond the profile it is built to serve.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

AppDep = Annotated[CareerGraphApp, Depends(get_app)]


# ---------------------------------------------------------------------------
# Middleware + error handling
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log method, path, status and latency for every request."""
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    logger.info(
        "Request handled",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return response


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    """503, not 500: the model is an upstream dependency and the caller may retry."""
    logger.error("LLM error", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(detail=str(exc), error_type="llm_error").model_dump(),
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("Invalid request", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(detail=str(exc), error_type="invalid_request").model_dump(),
    )


def _require_index(wired: CareerGraphApp) -> None:
    """409 when the store is empty.

    Without this, an unindexed deployment answers every question with a
    plausible-looking refusal, and the operator concludes the model is broken
    rather than that they forgot to run ingestion.
    """
    if wired.store.count() == 0:
        raise _IndexEmptyError()


class _IndexEmptyError(Exception):
    pass


@app.exception_handler(_IndexEmptyError)
async def handle_index_empty(request: Request, exc: _IndexEmptyError) -> JSONResponse:  # noqa: ARG001 - signature fixed by FastAPI
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(
            detail="No profile is indexed. Run `careergraph-ingest` first.",
            error_type="index_empty",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/query",
    response_model=RouteResponse,
    tags=["query"],
    summary="Ask anything — the router picks the agent",
    responses={
        409: {"model": ErrorResponse, "description": "Profile not yet ingested"},
        503: {"model": ErrorResponse, "description": "Language model unavailable"},
    },
)
async def query(request: QueryRequest, wired: AppDep) -> RouteResponse:
    """Classify the input and dispatch to the right agent.

    Send a question and you get an answer; paste a job description and you get
    a match report. Callers do not need to know which is which — check the
    `intent` field on the response to see what ran.
    """
    _require_index(wired)
    state = wired.router.route(request.query)
    intent = state["intent"]

    return RouteResponse(
        intent=intent,
        classification_method=state.get("classification_method", "heuristic"),
        answer=(
            AskResponse(**state["answer"].model_dump())
            if intent is Intent.RESUME_QA and state.get("answer")
            else None
        ),
        match=(
            MatchResponse.from_domain(state["match"])
            if intent is Intent.JD_MATCH and state.get("match")
            else None
        ),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["agents"],
    summary="Ask a question about the profile",
    responses={
        409: {"model": ErrorResponse, "description": "Profile not yet ingested"},
        503: {"model": ErrorResponse, "description": "Language model unavailable"},
    },
)
async def ask(request: AskRequest, wired: AppDep) -> AskResponse:
    """Answer a question strictly from indexed profile content.

    Returns HTTP 200 with `grounded: false` when the profile does not contain
    the answer. That is the designed behaviour — the alternative is a
    confident, fabricated answer.
    """
    _require_index(wired)
    return AskResponse(**wired.qa_agent.answer(request.question).model_dump())


@app.post(
    "/match",
    response_model=MatchResponse,
    tags=["agents"],
    summary="Score a job description against the profile",
    responses={
        409: {"model": ErrorResponse, "description": "Profile not yet ingested"},
        503: {"model": ErrorResponse, "description": "Language model unavailable"},
    },
)
async def match(request: MatchRequest, wired: AppDep) -> MatchResponse:
    """Compare a job description to the profile and report the gaps.

    Returns a 0-100 fit score, the requirements that are met (each with
    supporting evidence), the ones that are not, missing ATS keywords, and
    concrete recommendations.
    """
    _require_index(wired)
    return MatchResponse.from_domain(wired.jd_agent.match(request.job_description))


@app.get("/health", response_model=HealthResponse, tags=["diagnostics"], summary="Readiness check")
async def health(wired: AppDep) -> HealthResponse:
    """Report readiness, including whether the profile is actually indexed."""
    return HealthResponse(**wired.health())


@app.post(
    "/route-preview",
    response_model=RoutePreviewResponse,
    tags=["diagnostics"],
    summary="See how a query would be routed, without running an agent",
)
async def route_preview(request: QueryRequest, wired: AppDep) -> RoutePreviewResponse:
    """Classify intent only. Useful for debugging routing without spending tokens."""
    from careergraph.agents.router import heuristic_intent

    intent = heuristic_intent(request.query)
    method = "heuristic" if intent is not None else "llm"
    if intent is None:
        intent = wired.router.classify(request.query)

    preview = request.query[:120] + ("..." if len(request.query) > 120 else "")
    return RoutePreviewResponse(query_preview=preview, intent=intent, classification_method=method)


@app.get(
    "/search",
    response_model=SearchResponse,
    tags=["diagnostics"],
    summary="Raw semantic search, no LLM",
)
async def search(
    wired: AppDep,
    q: Annotated[str, Query(min_length=2, description="Search text.")],
    top_k: Annotated[int, Query(ge=1, le=25, description="Max chunks to return.")] = 5,
) -> SearchResponse:
    """Return the chunks retrieval would feed the LLM, with their scores.

    The fastest way to answer "was that a retrieval problem or a generation
    problem?" when an answer looks wrong.
    """
    _require_index(wired)
    result = wired.retriever.retrieve(q, top_k=top_k)
    return SearchResponse(
        query=q,
        threshold=wired.settings.retrieval_min_score,
        rejected_count=len(result.rejected),
        results=[
            SearchResultItem(
                chunk_id=item.chunk.id,
                section=item.chunk.section.value,
                title=item.chunk.title,
                score=round(item.score, 4),
                text=item.chunk.text,
            )
            for item in result.chunks
        ],
    )


@app.get(
    "/graph",
    tags=["diagnostics"],
    summary="Router graph as Mermaid",
    response_class=Response,
)
async def graph(wired: AppDep) -> Response:
    """Render the compiled LangGraph as Mermaid source.

    Generated from the actual compiled graph, so it cannot drift out of sync
    with the code the way a hand-drawn diagram does.
    """
    return Response(content=wired.router.mermaid(), media_type="text/plain")
