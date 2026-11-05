"""Render generic retrieval hits without fabricating provenance."""

from collections.abc import Sequence

from rag_core.generation.errors import GenerationValidationError
from rag_core.retrieval.models import KnowledgeHit


def _character_range(hit: KnowledgeHit) -> tuple[int, int] | None:
    """Return the enclosing real character range when available."""

    offsets = tuple(
        (span.char_start, span.char_end)
        for span in hit.source_spans
        if (
            span.char_start is not None
            and span.char_end is not None
        )
    )

    if not offsets:
        return None

    return (
        min(start for start, _ in offsets),
        max(end for _, end in offsets),
    )


def build_context_sections(
    hits: tuple[KnowledgeHit, ...],
) -> tuple[str, ...]:
    """Build one stable source-labelled section per retrieval hit."""

    if not hits:
        raise GenerationValidationError(
            "Cannot build context without retrieval results"
        )

    sections: list[str] = []

    for source_number, hit in enumerate(hits, start=1):
        lines = [
            f"[S{source_number}]",
            (
                f"Source: {hit.source_id}, "
                f"version {hit.source_version}"
            ),
            f"Location: {' > '.join(hit.structure_path)}",
        ]

        if hit.page_start is not None and hit.page_end is not None:
            lines.append(
                f"Pages: {hit.page_start}-{hit.page_end}"
            )
        else:
            character_range = _character_range(hit)

            if character_range is not None:
                lines.append(
                    "Characters: "
                    f"{character_range[0]}-{character_range[1]}"
                )

        lines.extend((
            "Content:",
            hit.text,
        ))
        sections.append("\n".join(lines))

    return tuple(sections)


def render_context(sections: Sequence[str]) -> str:
    """Join non-empty context sections exactly once."""

    if not sections or any(
        not section.strip()
        for section in sections
    ):
        raise GenerationValidationError(
            "Grounded context sections must not be empty"
        )

    return "\n\n".join(sections)
