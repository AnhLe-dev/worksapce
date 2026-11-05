"""Public API for provider-independent grounded generation."""

from rag_core.generation.citations import validate_answer_citations
from rag_core.generation.context import (
    build_context_sections,
    render_context,
)
from rag_core.generation.errors import (
    CitationValidationError,
    GenerationError,
    GenerationValidationError,
)
from rag_core.generation.models import (
    AnswerStatus,
    GenerationPrompt,
    GroundedAnswer,
    GroundedCitation,
    GroundedExecution,
    GroundedExecutionTrace,
)
from rag_core.generation.ports import AnswerGenerator
from rag_core.generation.prompt import build_grounded_prompt
from rag_core.generation.service import GroundedRagService

__all__ = (
    "AnswerGenerator",
    "AnswerStatus",
    "CitationValidationError",
    "GenerationError",
    "GenerationPrompt",
    "GenerationValidationError",
    "GroundedAnswer",
    "GroundedCitation",
    "GroundedExecution",
    "GroundedExecutionTrace",
    "GroundedRagService",
    "build_context_sections",
    "build_grounded_prompt",
    "render_context",
    "validate_answer_citations",
)
