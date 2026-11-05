"""Release A knowledge-base API schemas."""

from uuid import UUID

from pydantic import Field, field_validator

from app.api.schemas.common import ApiSchema
from app.api.schemas.jobs import JobResponse
from app.knowledge.models import (
    KnowledgeBaseStatus,
    KnowledgeDocumentStatus,
    MembershipStatus,
    SnapshotStatus,
)


class KnowledgeBaseCreateRequest(ApiSchema):
    name: str = Field(min_length=1, max_length=256)
    applicability_metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Knowledge base name must not be blank")

        return normalized


class KnowledgeBaseResponse(ApiSchema):
    knowledge_base_id: UUID
    name: str
    status: KnowledgeBaseStatus
    applicability_metadata: dict[str, str]


class KnowledgeMemberResponse(ApiSchema):
    membership_id: UUID
    membership_status: MembershipStatus
    membership_version: int
    snapshot_id: UUID
    snapshot_status: SnapshotStatus
    source_id: str
    source_version: str
    source_checksum_sha256: str
    index_profile: str


class KnowledgeBaseDetailResponse(KnowledgeBaseResponse):
    members: tuple[KnowledgeMemberResponse, ...]


class KnowledgeDocumentResponse(ApiSchema):
    knowledge_document_id: UUID
    role: str
    source_id: str
    source_version: str
    filename: str
    status: KnowledgeDocumentStatus


class KnowledgeUploadResponse(ApiSchema):
    document: KnowledgeDocumentResponse
    member: KnowledgeMemberResponse
    job: JobResponse | None
