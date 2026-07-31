"""Core domain types shared across ingestion, retrieval and the agents.

These are deliberately kept separate from the HTTP schemas in
``careergraph.api.schemas``. The API layer is a *contract with the outside
world* and should be free to evolve (renamed fields, added examples,
versioning) without dragging the internal model along with it. Coupling the
two is the classic mistake that makes an API impossible to change later.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field


class Section(StrEnum):
    """Top-level sections of the profile.

    Carried as chunk metadata so retrieval can be filtered by section
    (e.g. "only look at internships") and so citations can say *where* an
    answer came from.
    """

    ABOUT = "about"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    SKILLS = "skills"
    EDUCATION = "education"
    ACHIEVEMENTS = "achievements"


class Intent(StrEnum):
    """What the router decided a query is asking for."""

    RESUME_QA = "resume_qa"
    JD_MATCH = "jd_match"


class Chunk(BaseModel):
    """One retrievable unit of profile text plus its provenance."""

    id: str = Field(description="Stable, content-independent ID, e.g. 'experience::textr-ai::0'.")
    text: str = Field(description="The embedded and LLM-visible text.")
    section: Section
    title: str = Field(description="Human-readable source label, e.g. 'TEXTR AI - ML Intern'.")
    content_hash: str = Field(
        description="SHA-256 of the text. Drives the embedding cache and lets "
        "re-ingestion skip chunks whose content did not change."
    )

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, str]:
        """Chroma metadata values must be scalars, so flatten to plain strings."""
        return {
            "section": self.section.value,
            "title": self.title,
            "content_hash": self.content_hash,
        }


class RetrievedChunk(BaseModel):
    """A chunk returned by semantic search, with its similarity score."""

    chunk: Chunk
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity in [-1, 1]; higher is more relevant. "
        "Derived from Chroma's cosine *distance* as `1 - distance`.",
    )

    @property
    def citation(self) -> str:
        return f"{self.chunk.section.value}/{self.chunk.title}"


class Citation(BaseModel):
    """A source pointer attached to a generated answer."""

    chunk_id: str
    section: Section
    title: str
    score: float
    excerpt: str = Field(description="First ~240 chars of the chunk, for eyeballing.")

    @classmethod
    def from_retrieved(cls, retrieved: RetrievedChunk, excerpt_chars: int = 240) -> Citation:
        text = retrieved.chunk.text
        excerpt = text if len(text) <= excerpt_chars else text[:excerpt_chars].rstrip() + "..."
        return cls(
            chunk_id=retrieved.chunk.id,
            section=retrieved.chunk.section,
            title=retrieved.chunk.title,
            score=round(retrieved.score, 4),
            excerpt=excerpt,
        )


class Answer(BaseModel):
    """Output of the Resume Q&A agent."""

    answer: str
    grounded: bool = Field(
        description="False when the agent declined to answer because the profile "
        "did not contain the information. A refusal is a *success*, not an error."
    )
    citations: list[Citation] = Field(default_factory=list)


class SkillGap(BaseModel):
    """One requirement from a JD that the profile does not evidence."""

    skill: str
    importance: str = Field(description="'required' or 'preferred', as stated by the JD.")
    evidence_gap: str = Field(description="Why the profile does not satisfy this.")


class MatchedSkill(BaseModel):
    """One JD requirement the profile does evidence, with the receipt."""

    skill: str
    evidence: str = Field(description="Where in the profile this is demonstrated.")


class JDMatch(BaseModel):
    """Output of the JD-Matcher agent."""

    match_score: int = Field(ge=0, le=100)
    verdict: str = Field(description="One-line human summary of the fit.")
    matched_skills: list[MatchedSkill] = Field(default_factory=list)
    missing_skills: list[SkillGap] = Field(default_factory=list)
    missing_keywords: list[str] = Field(
        default_factory=list,
        description="Literal ATS keywords in the JD that are absent from the profile.",
    )
    recommendations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
