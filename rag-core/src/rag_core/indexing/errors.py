"""Typed failures for provider-independent document indexing."""


class IndexingError(RuntimeError):
    """Base class for expected indexing failures."""


class IndexingValidationError(IndexingError):
    """Raised before persistence when indexing input is invalid."""


class SourceVersionConflictError(IndexingError):
    """Raised when one source version already has another checksum."""


class IndexSnapshotConflictError(IndexingError):
    """Raised when stored chunk IDs are not part of the new snapshot."""


class IndexVerificationError(IndexingError):
    """Raised when stored vector data cannot prove index integrity."""