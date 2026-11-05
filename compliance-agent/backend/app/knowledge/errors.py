"""Typed failures for knowledge lifecycle and authority boundaries."""


class KnowledgeError(RuntimeError):
    """Base class for expected knowledge failures."""


class KnowledgeBaseNotFoundError(KnowledgeError, LookupError):
    """Raised when a KB is not available in the trusted scope."""


class KnowledgeDocumentConflictError(KnowledgeError):
    """Raised when a source version points to different bytes."""


class KnowledgeSnapshotNotReadyError(KnowledgeError):
    """Raised when a KB contains no fully verified source snapshot."""


class KnowledgeIsolationError(KnowledgeError):
    """Raised when a document, snapshot, membership, or hit escapes scope."""


class KnowledgeIndexingConflictError(KnowledgeError):
    """Raised when an index result does not match the prepared snapshot."""


class KnowledgeReferenceNotFoundError(KnowledgeError, LookupError):
    """Raised when a scoped full reference unit does not exist."""
