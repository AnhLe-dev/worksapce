"""Provider-independent models for document indexing."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IndexingStatus(StrEnum):
    """Describe whether an immutable source snapshot changed the index."""

    INDEXED = "indexed"
    UNCHANGED = "unchanged"


class IndexSnapshotStatus(StrEnum):
    """Describe a read-only comparison with one expected snapshot."""

    EMPTY = "empty"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CONFLICT = "conflict"


class IndexingResult(BaseModel):
    """Record one verified vector-index operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    collection_name: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_scope: str = Field(min_length=1)
    status: IndexingStatus
    chunk_count: int = Field(gt=0)
    uploaded_count: int = Field(ge=0)
    stored_count: int = Field(ge=0)
    vector_dimensions: int = Field(gt=0)


class IndexSnapshotInspection(BaseModel):
    """Report stored lineage without mutating the vector index."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    collection_name: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_scope: str = Field(min_length=1)
    status: IndexSnapshotStatus
    expected_count: int = Field(gt=0)
    stored_count: int = Field(ge=0)
    missing_chunk_ids: tuple[str, ...] = ()
    unexpected_chunk_ids: tuple[str, ...] = ()
    stored_checksums: tuple[str, ...] = ()
