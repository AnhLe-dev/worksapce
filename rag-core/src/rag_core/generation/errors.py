"""Typed failures for grounded answer generation."""


class GenerationError(RuntimeError):
    """Base class for expected generation failures."""


class GenerationValidationError(GenerationError):
    """Raised when generation input or provider output is invalid."""


class CitationValidationError(GenerationError):
    """Raised when answer citations do not map to supplied context."""
