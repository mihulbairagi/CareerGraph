"""HTTP request/response schemas.

Kept separate from ``careergraph.models`` on purpose. The domain models are
internal and should be free to change; these are a public contract. Reusing
domain objects as API schemas is the shortcut that later makes every internal
rename a breaking API change.

Every model carries a ``json_schema_extra`` example. That is the difference
between Swagger docs that are genuinely usable — click "Try it out" and a real
request is already filled in — and the default "additionalProp1: string"
noise that tells a reader nothing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from careergraph.models import Citation, Intent, JDMatch, MatchedSkill, SkillGap

# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

_MIN_QUESTION_CHARS = 3
_MAX_QUESTION_CHARS = 2_000
_MIN_JD_CHARS = 50
_MAX_JD_CHARS = 20_000


def _require_non_blank(value: str) -> str:
    """Reject whitespace-only input.

    ``min_length`` alone would happily accept "        ", which then embeds to
    a meaningless vector and produces a confusing refusal instead of a clear
    422 telling the caller what they did wrong.
    """
    if not value.strip():
        raise ValueError("must not be empty or whitespace only")
    return value.strip()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """A natural-language question about the candidate."""

    question: str = Field(
        min_length=_MIN_QUESTION_CHARS,
        max_length=_MAX_QUESTION_CHARS,
        description="Question to answer strictly from the indexed profile.",
    )

    _v = field_validator("question")(_require_non_blank)

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What did the candidate build during the TEXTR AI internship?"}
        }
    }


class MatchRequest(BaseModel):
    """A job description to compare against the profile."""

    job_description: str = Field(
        min_length=_MIN_JD_CHARS,
        max_length=_MAX_JD_CHARS,
        description=(
            "Full text of the job posting. Treated strictly as data: any "
            "instructions embedded in it are ignored by the agent."
        ),
    )

    _v = field_validator("job_description")(_require_non_blank)

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_description": (
                    "Backend Engineer (New Grad)\n\n"
                    "Responsibilities:\n"
                    "- Build and operate REST APIs serving production traffic\n"
                    "- Work with the ML team to deploy models behind low-latency endpoints\n\n"
                    "Requirements:\n"
                    "- Strong Python and SQL\n"
                    "- Experience with FastAPI or a similar framework\n"
                    "- Familiarity with Docker and CI/CD\n"
                    "- Experience with Kubernetes and AWS\n\n"
                    "Preferred:\n"
                    "- Exposure to vector databases or RAG systems\n"
                    "- Go or Rust"
                )
            }
        }
    }


class QueryRequest(BaseModel):
    """Free-form input for the router to classify and dispatch."""

    query: str = Field(
        min_length=_MIN_QUESTION_CHARS,
        max_length=_MAX_JD_CHARS,
        description=(
            "Either a question or a pasted job description. The router decides "
            "which agent handles it, so callers do not need to know."
        ),
    )

    _v = field_validator("query")(_require_non_blank)

    model_config = {
        "json_schema_extra": {"example": {"query": "Which programming languages does he know?"}}
    }


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class AskResponse(BaseModel):
    """A grounded answer, or an explicit refusal."""

    answer: str = Field(description="The answer text, or a refusal message.")
    grounded: bool = Field(
        description=(
            "True when answered from retrieved profile content. False means the "
            "agent declined because the profile lacks the information — this is "
            "correct behaviour, not an error, and still returns HTTP 200."
        )
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Chunks used to produce the answer. Always empty on a refusal.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": (
                        "At TEXTR AI he worked on the document understanding pipeline, "
                        "building an OCR post-processing pipeline in Python that cut "
                        "character error rate by 18%, and fine-tuning a DistilBERT "
                        "classifier for document routing that reached 94% F1 [1]."
                    ),
                    "grounded": True,
                    "citations": [
                        {
                            "chunk_id": "experience::textr-ai::0",
                            "section": "experience",
                            "title": "TEXTR AI - Machine Learning Intern",
                            "score": 0.6721,
                            "excerpt": "He worked at TEXTR AI as Machine Learning Intern...",
                        }
                    ],
                },
                {
                    "answer": (
                        "I don't have information about that in the profile. I can only "
                        "answer questions grounded in the candidate's documented "
                        "experience, projects, skills, education and achievements."
                    ),
                    "grounded": False,
                    "citations": [],
                },
            ]
        }
    }


class MatchResponse(BaseModel):
    """Fit assessment for a job description."""

    match_score: int = Field(ge=0, le=100, description="Overall fit, 0-100.")
    verdict: str
    matched_skills: list[MatchedSkill] = Field(
        default_factory=list, description="Requirements met, each with profile evidence."
    )
    missing_skills: list[SkillGap] = Field(
        default_factory=list, description="Requirements with no supporting evidence."
    )
    missing_keywords: list[str] = Field(
        default_factory=list, description="Literal ATS terms from the JD absent from the profile."
    )
    recommendations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, match: JDMatch) -> MatchResponse:
        return cls(**match.model_dump())

    model_config = {
        "json_schema_extra": {
            "example": {
                "match_score": 68,
                "verdict": (
                    "Strong on Python, FastAPI and ML deployment; missing the "
                    "cloud and orchestration requirements."
                ),
                "matched_skills": [
                    {
                        "skill": "Python",
                        "evidence": "Primary language at TEXTR AI and in the CalAI project.",
                    },
                    {
                        "skill": "FastAPI",
                        "evidence": "Served CalAI model inference via FastAPI at ~310ms median.",
                    },
                ],
                "missing_skills": [
                    {
                        "skill": "Kubernetes",
                        "importance": "required",
                        "evidence_gap": "Docker appears in the profile, but no orchestration experience.",
                    },
                    {
                        "skill": "AWS",
                        "importance": "required",
                        "evidence_gap": "No cloud platform experience is documented.",
                    },
                ],
                "missing_keywords": ["Kubernetes", "AWS", "CI/CD", "Go", "Rust"],
                "recommendations": [
                    "Deploy an existing project to EKS or ECS to evidence both AWS and Kubernetes.",
                    "Add a CI/CD pipeline to a public repo to cover the CI/CD keyword.",
                ],
                "citations": [],
            }
        }
    }


class RouteResponse(BaseModel):
    """Router output: which agent ran, plus that agent's result.

    Exactly one of ``answer`` / ``match`` is populated, determined by
    ``intent``. Modelling both as optional on one response — rather than
    returning a bare union — keeps the OpenAPI schema legible and lets clients
    branch on ``intent`` instead of sniffing which key exists.
    """

    intent: Intent = Field(description="Which agent the router selected.")
    classification_method: Literal["heuristic", "llm"] = Field(
        description=(
            "How the intent was decided. 'heuristic' means no LLM call was "
            "needed — the common, free, instant path."
        )
    )
    answer: AskResponse | None = Field(
        default=None, description="Populated when intent is 'resume_qa'."
    )
    match: MatchResponse | None = Field(
        default=None, description="Populated when intent is 'jd_match'."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intent": "resume_qa",
                "classification_method": "heuristic",
                "answer": {
                    "answer": "He knows Python, C++, JavaScript, TypeScript and SQL [1].",
                    "grounded": True,
                    "citations": [],
                },
                "match": None,
            }
        }
    }


class RoutePreviewResponse(BaseModel):
    """Intent classification without running an agent."""

    query_preview: str
    intent: Intent
    classification_method: Literal["heuristic", "llm"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_preview": "Which programming languages does he know?",
                "intent": "resume_qa",
                "classification_method": "heuristic",
            }
        }
    }


class HealthResponse(BaseModel):
    """Readiness, not just liveness."""

    status: Literal["ok", "degraded"] = Field(
        description=(
            "'degraded' means the service is running but the vector store is "
            "empty, so every question would be refused. Reported explicitly "
            "because a 200 from an unindexed service is a painful thing to debug."
        )
    )
    chunks_indexed: int
    vector_store: str
    collection: str
    embedding_model: str
    llm_model: str
    llm_configured: bool = Field(
        description="False when ANTHROPIC_API_KEY is unset; retrieval still works."
    )
    detail: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "chunks_indexed": 13,
                "vector_store": "persistent",
                "collection": "careergraph_profile",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_model": "claude-sonnet-4-6",
                "llm_configured": True,
                "detail": None,
            }
        }
    }


class SearchResultItem(BaseModel):
    """One retrieved chunk, exposed for debugging retrieval quality."""

    chunk_id: str
    section: str
    title: str
    score: float
    text: str


class SearchResponse(BaseModel):
    """Raw retrieval output, with no LLM in the loop.

    Deliberately exposed: when an answer looks wrong, the first question is
    always "was it bad retrieval or bad generation?". Without this endpoint
    that is guesswork.
    """

    query: str
    threshold: float = Field(description="Similarity floor applied to this search.")
    results: list[SearchResultItem]
    rejected_count: int = Field(
        description="Candidates found but dropped for scoring below the threshold."
    )


class ErrorResponse(BaseModel):
    """Uniform error body for non-2xx responses."""

    detail: str
    error_type: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Language model request failed: rate limit exceeded",
                "error_type": "llm_error",
            }
        }
    }
