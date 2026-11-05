"""Release A contract API schemas."""

from uuid import UUID

from pydantic import Field
from rag_core.ingestion.models import SourceSpan

from app.api.schemas.common import ApiSchema
from app.contracts.models import ContractStatus


class ContractResponse(ApiSchema):
    contract_id: UUID
    role: str
    stored_object_id: UUID
    filename: str
    source_version: str
    status: ContractStatus
    processing_revision: int
    page_count: int | None


class ProcessingJobRequest(ApiSchema):
    expected_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=255)


class ReviewUnitResponse(ApiSchema):
    review_unit_id: UUID
    processing_revision: int
    order_index: int
    text: str
    label: str | None
    source_spans: tuple[SourceSpan, ...]
    page_start: int | None
    page_end: int | None
