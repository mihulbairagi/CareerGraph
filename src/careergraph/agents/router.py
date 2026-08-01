"""LangGraph router: classify intent, dispatch to the right agent.

The graph is deliberately small — classify, branch, run one agent, done:

    START -> classify -> (resume_qa | jd_match) -> END

The value is not graph complexity, it is that the control flow is *declared*
rather than buried in if/else inside a request handler. Nodes are pure
functions over a typed state dict, so each is unit-testable in isolation, the
execution path is inspectable, and adding a third agent (or a re-ranking step,
or a human-approval interrupt) is a node plus an edge rather than a rewrite of
the dispatch logic.

**Classification is heuristic first, LLM second — and that ordering is the
main design decision here.** Telling a question from a pasted job description
is not a hard problem: JDs are long, contain "Responsibilities"/"Requirements"
headings, and don't end in a question mark. Spending an LLM call, ~500ms and
real money on every request to determine something a length check answers
correctly the vast majority of the time is the wrong trade. The LLM
tie-breaker exists only for the genuinely ambiguous middle band, so the common
path stays free and instant while the hard cases stay accurate.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from careergraph.agents.jd_matcher import JDMatcherAgent
from careergraph.agents.qa_agent import ResumeQAAgent
from careergraph.config import Settings
from careergraph.llm import LLMClient, LLMError
from careergraph.logging_config import get_logger
from careergraph.models import Answer, Intent, JDMatch

logger = get_logger(__name__)

# Section headings that essentially only appear in job postings.
_JD_MARKERS = (
    "responsibilities",
    "requirements",
    "qualifications",
    "what you'll do",
    "what you will do",
    "who you are",
    "about the role",
    "job description",
    "we are looking for",
    "we're looking for",
    "years of experience",
    "minimum qualifications",
    "preferred qualifications",
    "nice to have",
    "must have",
    "benefits",
    "equal opportunity",
    "apply now",
    "job title",
    "employment type",
)

# Openers that mark a natural-language question about the candidate.
_QUESTION_MARKERS = (
    "what ",
    "which ",
    "who ",
    "where ",
    "when ",
    "why ",
    "how ",
    "does ",
    "did ",
    "do ",
    "is ",
    "are ",
    "was ",
    "were ",
    "can ",
    "has ",
    "have ",
    "tell me",
    "describe ",
    "explain ",
    "list ",
    "summarize",
    "summarise",
)

# Below this, a JD is implausible — nobody pastes a 200-char job posting.
# Above the upper bound, a "question" that long is almost certainly a JD.
_SHORT_INPUT_CHARS = 200
_LONG_INPUT_CHARS = 900

CLASSIFIER_SYSTEM_PROMPT = """You classify a single piece of user input into exactly one category.

resume_qa  - a question about a candidate's background, skills, experience, projects, \
education or achievements.
jd_match   - a job description, job posting, or role specification to be compared \
against the candidate.

Answer with exactly one word: resume_qa or jd_match. No punctuation or explanation."""


class RouterState(TypedDict, total=False):
    """State passed between graph nodes.

    ``total=False`` because nodes populate it progressively: ``classify`` adds
    ``intent``, then exactly one agent node adds its result.
    """

    query: str
    intent: Intent
    classification_method: str
    answer: Answer | None
    match: JDMatch | None


def heuristic_intent(text: str) -> Intent | None:
    """Classify without an LLM. Returns ``None`` when genuinely ambiguous.

    Signals, in priority order:
      * Short input ending in '?' -> a question. Unambiguous.
      * Several JD section headings -> a posting, whatever its length.
      * Long input -> a posting; real questions are short.
      * Short input with a question opener -> a question.
    """
    stripped = text.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    length = len(stripped)
    marker_hits = sum(1 for m in _JD_MARKERS if m in lowered)

    # A short input ending in a question mark is a question. Checked first so
    # that a question *about* a JD keyword ("what are his requirements?")
    # doesn't get miscounted as a posting.
    if stripped.endswith("?") and length < _SHORT_INPUT_CHARS:
        return Intent.RESUME_QA

    # Two or more section headings is a strong, length-independent signal.
    if marker_hits >= 2:
        return Intent.JD_MATCH

    if length > _LONG_INPUT_CHARS:
        return Intent.JD_MATCH

    if length < _SHORT_INPUT_CHARS:
        if lowered.startswith(_QUESTION_MARKERS) or stripped.endswith("?"):
            return Intent.RESUME_QA
        # One JD marker in a short string, e.g. "Requirements: Python, 3 years".
        if marker_hits >= 1:
            return Intent.JD_MATCH
        return Intent.RESUME_QA

    # 200-900 chars with at most one marker: a long question and a short JD
    # look alike here. Hand it to the LLM.
    if marker_hits >= 1:
        return Intent.JD_MATCH
    if stripped.endswith("?") and len(re.findall(r"\n", stripped)) <= 2:
        return Intent.RESUME_QA
    return None


class QueryRouter:
    """Builds and runs the LangGraph routing graph."""

    def __init__(
        self,
        settings: Settings,
        qa_agent: ResumeQAAgent,
        jd_agent: JDMatcherAgent,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._qa_agent = qa_agent
        self._jd_agent = jd_agent
        self._llm = llm
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _classify_node(self, state: RouterState) -> dict[str, Any]:
        query = state["query"]

        intent = heuristic_intent(query)
        method = "heuristic"
        if intent is None:
            intent = self._classify_with_llm(query)
            method = "llm"

        logger.info(
            "Routed query",
            extra={"intent": intent.value, "method": method, "query_chars": len(query)},
        )
        return {"intent": intent, "classification_method": method}

    def _qa_node(self, state: RouterState) -> dict[str, Any]:
        return {"answer": self._qa_agent.answer(state["query"])}

    def _jd_node(self, state: RouterState) -> dict[str, Any]:
        return {"match": self._jd_agent.match(state["query"])}

    @staticmethod
    def _route_edge(state: RouterState) -> str:
        return state["intent"].value

    def _classify_with_llm(self, query: str) -> Intent:
        try:
            # Truncated: the first 2000 chars are more than enough to tell a
            # question from a posting, and it caps cost on a huge paste.
            raw = self._llm.complete(
                system=CLASSIFIER_SYSTEM_PROMPT,
                user=query[:2000],
                max_tokens=8,
            )
        except LLMError as exc:
            # Never fail a request because *classification* failed. Fall back
            # on length, which is the single most predictive feature.
            logger.warning(
                "LLM classification failed; using length fallback", extra={"error": str(exc)}
            )
            return Intent.JD_MATCH if len(query) > _LONG_INPUT_CHARS else Intent.RESUME_QA

        normalised = raw.strip().lower()
        if "jd_match" in normalised:
            return Intent.JD_MATCH
        if "resume_qa" in normalised:
            return Intent.RESUME_QA
        logger.warning("Unrecognised classifier output", extra={"raw": raw[:80]})
        return Intent.RESUME_QA

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(RouterState)
        graph.add_node("classify", self._classify_node)
        graph.add_node(Intent.RESUME_QA.value, self._qa_node)
        graph.add_node(Intent.JD_MATCH.value, self._jd_node)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_edge,
            {
                Intent.RESUME_QA.value: Intent.RESUME_QA.value,
                Intent.JD_MATCH.value: Intent.JD_MATCH.value,
            },
        )
        graph.add_edge(Intent.RESUME_QA.value, END)
        graph.add_edge(Intent.JD_MATCH.value, END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route(self, query: str) -> RouterState:
        """Classify and dispatch. Returns the populated final state."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self._graph.invoke({"query": query})

    def classify(self, query: str) -> Intent:
        """Intent only, no agent execution. Powers ``GET /route-preview``."""
        return heuristic_intent(query) or self._classify_with_llm(query)

    def mermaid(self) -> str:
        """Render the compiled graph as Mermaid, for the README."""
        return self._graph.get_graph().draw_mermaid()
