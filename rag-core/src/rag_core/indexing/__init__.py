"""Public API for provider-independent document indexing."""

from rag_core.indexing.errors import (
    IndexingError,
    IndexingValidationError,
    IndexSnapshotConflictError,
    IndexVerificationError,
    SourceVersionConflictError,
)
from rag_core.indexing.models import (
    IndexingResult,
    IndexingStatus,
    IndexSnapshotInspection,
    IndexSnapshotStatus,
)
from rag_core.indexing.payloads import (
    INDEX_PAYLOAD_SCHEMA_VERSION,
    build_index_payload,
)
from rag_core.indexing.ports import (
    EmbeddingProvider,
    IndexableChunk,
    VectorIndex,
)
from rag_core.indexing.service import (
    DocumentIndexingService,
    build_embedding_text,
)

__all__ = (
    "INDEX_PAYLOAD_SCHEMA_VERSION",
    "DocumentIndexingService",
    "EmbeddingProvider",
    "IndexSnapshotConflictError",
    "IndexSnapshotInspection",
    "IndexSnapshotStatus",
    "IndexVerificationError",
    "IndexableChunk",
    "IndexingError",
    "IndexingResult",
    "IndexingStatus",
    "IndexingValidationError",
    "SourceVersionConflictError",
    "VectorIndex",
    "build_embedding_text",
    "build_index_payload",
)