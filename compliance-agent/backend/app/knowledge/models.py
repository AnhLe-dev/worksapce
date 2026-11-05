"""Domain models for knowledge documents, bases, and source snapshots."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from rag_core.ingestion.models import DocumentChunk, SourceSpan
from rag_core.retrieval.models import KnowledgeHit, RetrievalScope


class KnowledgeDocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class KnowledgeBaseStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class SnapshotStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class MembershipStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    REMOVED = "removed"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KnowledgeDocument(_FrozenModel):
    knowledge_document_id: UUID
    role: Literal["knowledge"] = "knowledge"
    stored_object_id: UUID
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=512)
    access_scope: str = Field(min_length=1, max_length=128)
    status: KnowledgeDocumentStatus
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "source_id",
        "source_version",
        "filename",
        "access_scope",
    )
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Knowledge identity fields must not be blank")

        return normalized

    @model_validator(mode="after")
    def validate_failure(self) -> "KnowledgeDocument":
        if self.status is KnowledgeDocumentStatus.FAILED:
            if not self.error_code:
                raise ValueError("FAILED knowledge document needs error code")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("Only FAILED documents may contain error details")

        return self


class KnowledgeBase(_FrozenModel):
    knowledge_base_id: UUID
    name: str = Field(min_length=1, max_length=256)
    access_scope: str = Field(min_length=1, max_length=128)
    status: KnowledgeBaseStatus
    applicability_metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("name", "access_scope")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Knowledge base identity must not be blank")

        return normalized


class KnowledgeSourceSnapshot(_FrozenModel):
    snapshot_id: UUID
    knowledge_document_id: UUID
    access_scope: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    index_profile: str = Field(min_length=1)
    status: SnapshotStatus
    collection_name: str | None = Field(default=None, min_length=1)
    chunk_count: int | None = Field(default=None, gt=0)
    vector_dimensions: int | None = Field(default=None, gt=0)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "access_scope",
        "source_id",
        "source_version",
        "index_profile",
    )
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Snapshot identity must not be blank")

        return normalized

    @model_validator(mode="after")
    def validate_state(self) -> "KnowledgeSourceSnapshot":
        verification = (
            self.collection_name,
            self.chunk_count,
            self.vector_dimensions,
        )

        if self.status is SnapshotStatus.READY and not all(verification):
            raise ValueError("READY snapshot requires index verification")

        if self.status is SnapshotStatus.FAILED and not self.error_code:
            raise ValueError("FAILED snapshot requires an error code")

        if self.status is not SnapshotStatus.FAILED and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("Only FAILED snapshots may contain errors")

        return self

    @property
    def retrieval_scope(self) -> RetrievalScope:
        """Convert one READY snapshot to the rag-core retrieval contract."""

        if self.status is not SnapshotStatus.READY:
            raise ValueError("Only READY snapshots can be retrieved")

        return RetrievalScope(
            access_scope=self.access_scope,
            source_id=self.source_id,
            source_version=self.source_version,
            source_checksum_sha256=self.source_checksum_sha256,
            index_profile=self.index_profile,
        )


class KnowledgeBaseMembership(_FrozenModel):
    membership_id: UUID
    knowledge_base_id: UUID
    snapshot_id: UUID
    access_scope: str = Field(min_length=1)
    membership_version: int = Field(ge=1)
    status: MembershipStatus
    created_at: datetime
    updated_at: datetime


class KnowledgeReferenceUnit(_FrozenModel):
    reference_unit_id: UUID
    snapshot_id: UUID
    order_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    structure_path: tuple[str, ...] = Field(min_length=1)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_pages(self) -> "KnowledgeReferenceUnit":
        pages = tuple(span.page_number for span in self.source_spans)
        all_paged = all(page is not None for page in pages)
        all_unpaged = all(page is None for page in pages)

        if not (all_paged or all_unpaged):
            raise ValueError(
                "Reference unit cannot mix paged and unpaged spans"
            )

        if all_unpaged:
            if self.page_start is not None or self.page_end is not None:
                raise ValueError("Unpaged reference cannot declare pages")
            return self

        concrete_pages = tuple(page for page in pages if page is not None)

        if (
            self.page_start != min(concrete_pages)
            or self.page_end != max(concrete_pages)
        ):
            raise ValueError("Reference page range must match source spans")

        return self


class ResolvedKnowledgeBase(_FrozenModel):
    knowledge_base: KnowledgeBase
    snapshots: tuple[KnowledgeSourceSnapshot, ...] = Field(min_length=1)
    scopes: tuple[RetrievalScope, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_snapshot_scopes(self) -> "ResolvedKnowledgeBase":
        expected_scopes = tuple(
            snapshot.retrieval_scope for snapshot in self.snapshots
        )

        if self.scopes != expected_scopes:
            raise ValueError("Resolved scopes must match READY snapshots")

        return self


class PreparedKnowledgeSnapshot(_FrozenModel):
    snapshot: KnowledgeSourceSnapshot
    chunks: tuple[DocumentChunk, ...] = Field(min_length=1)
    reference_units: tuple[KnowledgeReferenceUnit, ...] = Field(min_length=1)


class KnowledgeIndexingOutcome(_FrozenModel):
    snapshot: KnowledgeSourceSnapshot
    reused_existing: bool


class KnowledgeRegistration(_FrozenModel):
    document: KnowledgeDocument
    snapshot: KnowledgeSourceSnapshot
    membership: KnowledgeBaseMembership
    reused_snapshot: bool


class KnowledgeRetrievalRequest(_FrozenModel):
    knowledge_base_id: UUID
    access_scope: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(gt=0, le=100)

    @field_validator("access_scope", "query")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Retrieval fields must not be blank")

        return normalized


class KnowledgeRetrievalResult(_FrozenModel):
    knowledge_base_id: UUID
    snapshots: tuple[KnowledgeSourceSnapshot, ...] = Field(min_length=1)
    scopes: tuple[RetrievalScope, ...] = Field(min_length=1)
    hits: tuple[KnowledgeHit, ...]

    @model_validator(mode="after")
    def validate_result_scopes(self) -> "KnowledgeRetrievalResult":
        if self.scopes != tuple(
            snapshot.retrieval_scope for snapshot in self.snapshots
        ):
            raise ValueError("Retrieval scopes must match resolved snapshots")

        return self
