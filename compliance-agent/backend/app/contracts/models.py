"""Generic contract and review-unit domain models."""

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
from rag_core.ingestion.models import SourceSpan


class ContractStatus(StrEnum):
    """Represent the generic contract-processing lifecycle."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContractDocument(_FrozenModel):
    """Represent a contract independently from its physical blob."""

    contract_id: UUID
    role: Literal["contract"] = "contract"
    stored_object_id: UUID
    filename: str = Field(min_length=1, max_length=512)
    access_scope: str = Field(min_length=1, max_length=128)
    source_version: str = Field(min_length=1, max_length=128)
    processing_revision: int = Field(default=0, ge=0)
    status: ContractStatus
    page_count: int | None = Field(default=None, ge=1)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("filename", "access_scope", "source_version")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Contract identity fields must not be blank")

        return normalized

    @model_validator(mode="after")
    def validate_status_details(self) -> "ContractDocument":
        if self.status is ContractStatus.FAILED and not self.error_code:
            raise ValueError("FAILED contract requires an error code")

        if self.status is not ContractStatus.FAILED and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError(
                "Only a FAILED contract may contain error details"
            )

        return self


class ReviewUnit(_FrozenModel):
    """Represent reviewable contract text with only real source spans."""

    review_unit_id: UUID
    contract_id: UUID
    processing_revision: int = Field(ge=1)
    order_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    label: str | None = Field(default=None, min_length=1, max_length=512)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "ReviewUnit":
        pages = tuple(
            span.page_number
            for span in self.source_spans
            if span.page_number is not None
        )

        if not pages:
            if self.page_start is not None or self.page_end is not None:
                raise ValueError("Unpaged review units cannot declare pages")
            return self

        if len(pages) != len(self.source_spans):
            raise ValueError("Review units cannot mix paged and unpaged spans")

        if self.page_start != min(pages) or self.page_end != max(pages):
            raise ValueError("Review unit page range must match its spans")

        return self


class ContractProcessingResult(_FrozenModel):
    """Summarize one successful deterministic processing revision."""

    contract: ContractDocument
    review_units: tuple[ReviewUnit, ...] = Field(min_length=1)
