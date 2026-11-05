"""Validate source-label linkage in generated answers."""

import re

from rag_core.generation.errors import CitationValidationError

_CONTEXT_CITATION_PATTERN = re.compile(
    r"(?m)^\[(S[1-9]\d*)\]$"
)
_ANSWER_CITATION_PATTERN = re.compile(
    r"\[(S[1-9]\d*)\]"
)


def _extract_unique_labels(
    pattern: re.Pattern[str],
    text: str,
) -> tuple[str, ...]:
    """Extract labels while preserving first-use order."""

    return tuple(dict.fromkeys(pattern.findall(text)))


def validate_answer_citations(
    *,
    answer: str,
    context: str,
) -> tuple[str, ...]:
    """Require at least one known label in a non-empty answer."""

    normalized_answer = answer.strip()
    normalized_context = context.strip()

    if not normalized_answer:
        raise CitationValidationError(
            "Cannot validate an empty answer"
        )

    if not normalized_context:
        raise CitationValidationError(
            "Cannot validate citations without context"
        )

    available_labels = _extract_unique_labels(
        _CONTEXT_CITATION_PATTERN,
        normalized_context,
    )

    if not available_labels:
        raise CitationValidationError(
            "Context does not contain source labels"
        )

    used_labels = _extract_unique_labels(
        _ANSWER_CITATION_PATTERN,
        normalized_answer,
    )

    if not used_labels:
        raise CitationValidationError(
            "Generated answer does not contain citations"
        )

    unknown_labels = tuple(
        label
        for label in used_labels
        if label not in available_labels
    )

    if unknown_labels:
        raise CitationValidationError(
            "Generated answer contains unknown citations: "
            f"{unknown_labels}"
        )

    return used_labels
