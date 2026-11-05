"""Typed failures for provider-independent retrieval."""


class RetrievalError(RuntimeError):
    """Base class for expected retrieval failures."""


class RetrievalValidationError(RetrievalError):
    """Raised when a request or query vector is invalid."""


class RetrievalIsolationError(RetrievalError):
    """Raised when returned evidence escapes the requested scope."""


class UnsupportedPayloadSchemaError(RetrievalError):
    """Raised when indexed payloads use an unknown schema version."""
