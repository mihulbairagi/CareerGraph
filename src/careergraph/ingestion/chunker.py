"""Turn ``profile.json`` into retrievable chunks.

Design note — **why not a generic recursive text splitter?**

The off-the-shelf approach is to serialise the whole profile to text and cut
it every N characters. That is the wrong tool here. A resume is not prose;
it is a small, highly structured document where the natural semantic
boundaries are already explicit (one job, one project, one skill category).
Splitting on character count would slice a job's bullet list in half, so a
query like *"what did you build at TEXTR AI?"* could retrieve the second half
of the bullets without the company name attached — the retrieved chunk would
be unattributable and the LLM would be forced to guess.

So we chunk **structurally**: one chunk per semantic record, each one
self-describing (it repeats the company/project name and dates inside the
text). Character-based splitting only kicks in as a fallback for records that
are genuinely too long to embed well, and even then the header is repeated on
every piece so provenance survives the cut.
"""

from __future__ import annotations

import re
from typing import Any

from careergraph.config import Settings
from careergraph.logging_config import get_logger
from careergraph.models import Chunk, Section

logger = get_logger(__name__)

# Placeholder markers left in the scaffolded profile.json. Ingesting these
# would poison retrieval with instructional text like "FILL_ME — bullet with
# a metric", which the LLM would happily quote back as fact.
_PLACEHOLDER_RE = re.compile(r"FILL_ME|^TODO\b", re.IGNORECASE)


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def _clean(value: Any) -> str:
    """Normalise a scalar to trimmed text, dropping unfilled placeholders."""
    if value is None or _is_placeholder(value):
        return ""
    return str(value).strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [c for c in (_clean(v) for v in values) if c]


def _date_range(start: str, end: str) -> str:
    start, end = _clean(start), _clean(end)
    if start and end:
        return f"{start} to {end}"
    return start or end or ""


def _join_block(parts: list[str]) -> str:
    """Join non-empty lines into a single chunk body."""
    return "\n".join(p for p in parts if p).strip()


class ProfileChunker:
    """Converts the parsed profile document into a list of :class:`Chunk`."""

    def __init__(self, settings: Settings) -> None:
        self._max_chars = settings.chunk_max_chars
        self._overlap = settings.chunk_overlap_chars

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def chunk(self, profile: dict[str, Any]) -> list[Chunk]:
        person = _clean(profile.get("name")) or "The candidate"

        chunks: list[Chunk] = []
        chunks += self._about(profile, person)
        chunks += self._experience(profile, person)
        chunks += self._projects(profile, person)
        chunks += self._skills(profile, person)
        chunks += self._education(profile, person)
        chunks += self._achievements(profile, person)

        logger.info(
            "Chunked profile",
            extra={
                "chunk_count": len(chunks),
                "sections": sorted({c.section.value for c in chunks}),
            },
        )
        if not chunks:
            raise ValueError(
                "Profile produced zero chunks. Every field still contains FILL_ME "
                "placeholders — edit data/profile.json before ingesting."
            )
        return chunks

    # ------------------------------------------------------------------
    # Per-section builders
    # ------------------------------------------------------------------
    def _about(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        paragraphs = _clean_list(profile.get("about"))
        headline = _clean(profile.get("headline"))
        location = _clean(profile.get("location"))

        # Links and preferences are short and always queried together with the
        # bio ("how do I contact him?", "is he open to relocating?"), so they
        # ride along in the About chunk rather than becoming their own.
        links = {k: _clean(v) for k, v in (profile.get("links") or {}).items()}
        link_line = ", ".join(f"{k}: {v}" for k, v in links.items() if v)

        prefs = profile.get("preferences") or {}
        pref_lines = []
        if roles := _clean_list(prefs.get("target_roles")):
            pref_lines.append(f"Target roles: {', '.join(roles)}.")
        if (relocate := prefs.get("open_to_relocate")) is not None:
            pref_lines.append(f"Open to relocation: {'yes' if relocate else 'no'}.")
        if availability := _clean(prefs.get("availability")):
            pref_lines.append(f"Availability: {availability}.")

        body = _join_block(
            [
                f"{person} — {headline}" if headline else person,
                f"Location: {location}." if location else "",
                *paragraphs,
                *pref_lines,
                f"Links — {link_line}." if link_line else "",
            ]
        )
        if not body or body == person:
            return []
        return self._emit(Section.ABOUT, "profile", f"About {person}", body)

    def _experience(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, job in enumerate(profile.get("experience") or []):
            company = _clean(job.get("company"))
            role = _clean(job.get("role"))
            if not company:
                continue

            dates = _date_range(job.get("start_date"), job.get("end_date"))
            title = f"{company} — {role}" if role else company
            header = f"{person} worked at {company}"
            if role:
                header += f" as {role}"
            if emp_type := _clean(job.get("employment_type")):
                header += f" ({emp_type})"
            if dates:
                header += f", {dates}"
            if loc := _clean(job.get("location")):
                header += f", based in {loc}"

            highlights = _clean_list(job.get("highlights"))
            tech = _clean_list(job.get("tech"))

            body = _join_block(
                [
                    header + ".",
                    _clean(job.get("summary")),
                    ("Key contributions:\n" + "\n".join(f"- {h}" for h in highlights))
                    if highlights
                    else "",
                    f"Technologies used at {company}: {', '.join(tech)}." if tech else "",
                ]
            )
            # A header with no substance behind it is noise, not signal.
            if not (highlights or tech or _clean(job.get("summary"))):
                logger.warning(
                    "Skipping experience entry with no content",
                    extra={"company": company},
                )
                continue
            chunks += self._emit(Section.EXPERIENCE, _slug(company) or f"job-{i}", title, body)
        return chunks

    def _projects(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, proj in enumerate(profile.get("projects") or []):
            name = _clean(proj.get("name"))
            if not name:
                continue

            dates = _date_range(proj.get("start_date"), proj.get("end_date"))
            tagline = _clean(proj.get("tagline"))
            highlights = _clean_list(proj.get("highlights"))
            tech = _clean_list(proj.get("tech"))
            summary = _clean(proj.get("summary"))

            if not (summary or highlights or tech):
                logger.warning("Skipping project with no content", extra={"project": name})
                continue

            header = f"{name} is a project built by {person}"
            if tagline:
                header += f": {tagline}"
            if dates:
                header += f" ({dates})"

            links = [
                f"Repository: {_clean(proj.get('repo'))}" if _clean(proj.get("repo")) else "",
                f"Demo: {_clean(proj.get('demo'))}" if _clean(proj.get("demo")) else "",
            ]

            body = _join_block(
                [
                    header.rstrip(".") + ".",
                    summary,
                    ("Highlights:\n" + "\n".join(f"- {h}" for h in highlights))
                    if highlights
                    else "",
                    f"Tech stack for {name}: {', '.join(tech)}." if tech else "",
                    *links,
                ]
            )
            chunks += self._emit(Section.PROJECTS, _slug(name) or f"project-{i}", name, body)
        return chunks

    def _skills(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        groups = profile.get("skills") or []
        chunks: list[Chunk] = []
        all_items: list[str] = []

        for i, group in enumerate(groups):
            category = _clean(group.get("category"))
            items = _clean_list(group.get("items"))
            if not items:
                continue
            all_items += items
            body = f"{person}'s skills in {category}: {', '.join(items)}."
            chunks += self._emit(
                Section.SKILLS, _slug(category) or f"skills-{i}", f"Skills — {category}", body
            )

        # One flat roll-up chunk in addition to the per-category ones. The JD
        # matcher asks "which of these 20 requirements does he meet?" — a single
        # chunk listing every skill answers that in one retrieval hit, whereas
        # per-category chunks alone could push some categories below top_k.
        if all_items:
            body = (
                f"Complete list of {person}'s technical skills and technologies: "
                f"{', '.join(dict.fromkeys(all_items))}."
            )
            chunks += self._emit(Section.SKILLS, "all", "Skills — full list", body)
        return chunks

    def _education(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, edu in enumerate(profile.get("education") or []):
            institution = _clean(edu.get("institution"))
            if not institution:
                continue

            degree = _clean(edu.get("degree"))
            field = _clean(edu.get("field"))
            years = _date_range(edu.get("start_year"), edu.get("end_year"))

            qualification = " in ".join(p for p in (degree, field) if p)
            header = f"{person} studied at {institution}"
            if qualification:
                header += f", earning a {qualification}"
            if years:
                header += f" ({years})"

            coursework = _clean_list(edu.get("coursework"))
            body = _join_block(
                [
                    header + ".",
                    f"Academic performance: {_clean(edu.get('score'))}."
                    if _clean(edu.get("score"))
                    else "",
                    f"Relevant coursework: {', '.join(coursework)}." if coursework else "",
                    _clean(edu.get("notes")),
                ]
            )
            title = f"{institution} — {qualification}" if qualification else institution
            chunks += self._emit(
                Section.EDUCATION, _slug(institution) or f"education-{i}", title, body
            )
        return chunks

    def _achievements(self, profile: dict[str, Any], person: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        entries = list(profile.get("achievements") or []) + list(
            profile.get("certifications") or []
        )
        for i, item in enumerate(entries):
            title = _clean(item.get("title"))
            description = _clean(item.get("description"))
            if not title:
                continue
            body = _join_block(
                [
                    f"{person} achieved: {title}"
                    + (f" ({_clean(item.get('date'))})" if _clean(item.get("date")) else "")
                    + ".",
                    description,
                    f"Link: {_clean(item.get('link'))}" if _clean(item.get("link")) else "",
                ]
            )
            chunks += self._emit(
                Section.ACHIEVEMENTS, _slug(title) or f"achievement-{i}", title, body
            )
        return chunks

    # ------------------------------------------------------------------
    # Emission + overflow splitting
    # ------------------------------------------------------------------
    def _emit(self, section: Section, key: str, title: str, body: str) -> list[Chunk]:
        """Build one chunk, splitting with overlap only if the body is oversized."""
        body = body.strip()
        if not body:
            return []

        if len(body) <= self._max_chars:
            return [self._build(section, key, 0, title, body)]

        pieces = _split_with_overlap(body, self._max_chars, self._overlap)
        logger.debug(
            "Record exceeded chunk_max_chars; split",
            extra={"title": title, "parts": len(pieces)},
        )
        # Repeat the title on continuation parts: a mid-record fragment must
        # still be attributable on its own, since retrieval may return part 3
        # without parts 1 and 2.
        return [
            self._build(
                section,
                key,
                idx,
                title,
                piece if idx == 0 else f"(continued) {title}\n{piece}",
            )
            for idx, piece in enumerate(pieces)
        ]

    def _build(self, section: Section, key: str, idx: int, title: str, text: str) -> Chunk:
        return Chunk(
            id=f"{section.value}::{key}::{idx}",
            text=text,
            section=section,
            title=title,
            content_hash=Chunk.hash_text(text),
        )


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph/line boundaries, carrying ``overlap`` chars forward.

    The overlap exists so a sentence straddling a boundary is fully present in
    at least one chunk; without it, retrieval can return a fragment whose
    subject was cut off in the previous piece.
    """
    lines = text.split("\n")
    pieces: list[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{line}" if tail else line
        else:
            # A single line longer than max_chars: hard-cut it.
            for i in range(0, len(line), max_chars):
                pieces.append(line[i : i + max_chars])
            current = ""

    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def _slug(value: str) -> str:
    """Lowercase, hyphenated, ASCII-safe identifier fragment."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
