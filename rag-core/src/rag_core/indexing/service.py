"""Application service for provider-independent dense indexing."""

from math import isfinite

from rag_core.indexing.errors import IndexingValidationError
from rag_core.indexing.models import IndexingResult
from rag_core.indexing.ports import (
    EmbeddingProvider,
    IndexableChunk,
    VectorIndex,
)


def build_embedding_text(
    chunk: IndexableChunk,
) -> str:
    """Build deterministic embedding input from structure and content."""

    structure_context = " > ".join(
        chunk.structure_path
    )

    return (
        f"{structure_context}\n"
        f"{chunk.text}"
    ).strip()


class DocumentIndexingService:
    """Validate, embed, and persist one immutable source snapshot."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        expected_dimensions: int,
    ) -> None:
        if expected_dimensions <= 0:
            raise IndexingValidationError(
                "Expected vector dimensions must be greater than zero"
            )

        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._expected_dimensions = expected_dimensions

    @staticmethod
    def _validate_snapshot(
        chunks: tuple[IndexableChunk, ...],
    ) -> None:
        """Require one non-empty source identity and unique chunk IDs."""

        if not chunks:
            raise IndexingValidationError(
                "Cannot index an empty chunk collection"
            )

        for chunk in chunks:
            identity_values = (
                chunk.chunk_id,
                chunk.source_id,
                chunk.source_version,
                chunk.source_checksum_sha256,
                chunk.access_scope,
            )

            if any(not value.strip() for value in identity_values):
                raise IndexingValidationError(
                    "Chunk identity fields must not be empty"
                )

            checksum = chunk.source_checksum_sha256

            if len(checksum) != 64 or any(
                character not in "0123456789abcdef"
                for character in checksum
            ):
                raise IndexingValidationError(
                    "Chunk checksum must be a lowercase SHA-256 value"
                )

            if not chunk.text.strip():
                raise IndexingValidationError(
                    "Chunk text must not be empty"
                )

        identities = {
            (
                chunk.source_id,
                chunk.source_version,
                chunk.source_checksum_sha256,
                chunk.access_scope,
            )
            for chunk in chunks
        }

        if len(identities) != 1:
            raise IndexingValidationError(
                "Indexing request must contain one source snapshot"
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise IndexingValidationError(
                "Indexing request contains duplicate chunk IDs"
            )

    def index(
        self,
        chunks: tuple[IndexableChunk, ...],
    ) -> IndexingResult:
        """Generate validated vectors and persist one source snapshot."""

        self._validate_snapshot(chunks)

        texts = tuple(
            build_embedding_text(chunk)
            for chunk in chunks
        )

        raw_vectors = self._embedding_provider.embed_documents(
            texts
        )

        if len(raw_vectors) != len(chunks):
            raise IndexingValidationError(
                "Embedding vector count does not match chunk count"
            )

        try:
            vectors = tuple(
                tuple(float(value) for value in vector)
                for vector in raw_vectors
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise IndexingValidationError(
                "Embedding provider returned a non-numeric vector"
            ) from exc

        for vector_index, vector in enumerate(vectors):
            if len(vector) != self._expected_dimensions:
                raise IndexingValidationError(
                    "Embedding vector has an unexpected dimension: "
                    f"index={vector_index}, "
                    f"expected={self._expected_dimensions}, "
                    f"actual={len(vector)}"
                )

            if not all(isfinite(value) for value in vector):
                raise IndexingValidationError(
                    "Embedding vector contains a non-finite value: "
                    f"index={vector_index}"
                )

        return self._vector_index.upsert(
            chunks=chunks,
            vectors=vectors,
        )