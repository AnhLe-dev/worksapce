"""Durable technical job contracts shared by API, relay, and workers."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class JobKind(StrEnum):
    CONTRACT_PROCESSING = "contract_processing"
    CONTRACT_REVIEW = "contract_review"
    KNOWLEDGE_INDEXING = "knowledge_indexing"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRY = "retry"


class JobStage(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    INDEXING = "indexing"
    STORING_RESULT = "storing_result"
    RETRY_WAIT = "retry_wait"
    COMPLETED = "completed"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TechnicalJob(_FrozenModel):
    job_id: UUID
    kind: JobKind
    status: JobStatus
    stage: JobStage
    access_scope: str = Field(min_length=1)
    contract_id: UUID | None = None
    snapshot_id: UUID | None = None
    request_payload: dict[str, object]
    result_payload: dict[str, object] | None = None
    result_schema_version: int = Field(default=1, ge=1)
    idempotency_key: str = Field(min_length=1, max_length=255)
    dispatch_generation: int = Field(ge=0)
    attempt_id: UUID | None = None
    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("access_scope", "idempotency_key")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Job identity fields must not be blank")

        return normalized

    @model_validator(mode="after")
    def validate_target_and_state(self) -> "TechnicalJob":
        if self.kind in {
            JobKind.CONTRACT_PROCESSING,
            JobKind.CONTRACT_REVIEW,
        }:
            if self.contract_id is None or self.snapshot_id is not None:
                raise ValueError("Contract job requires only contract_id")
        elif self.snapshot_id is None or self.contract_id is not None:
            raise ValueError("Knowledge job requires only snapshot_id")

        if self.status is JobStatus.RUNNING and (
            self.attempt_id is None or self.lease_expires_at is None
        ):
            raise ValueError("RUNNING job requires an owned lease")

        if self.status is JobStatus.SUCCEEDED and (
            self.result_payload is None or self.finished_at is None
        ):
            raise ValueError(
                "SUCCEEDED job requires result and finish time"
            )

        if self.status is JobStatus.FAILED and (
            not self.error_code or self.finished_at is None
        ):
            raise ValueError("FAILED job requires error and finish time")

        return self


class JobClaim(_FrozenModel):
    job_id: UUID
    kind: JobKind
    access_scope: str = Field(min_length=1)
    contract_id: UUID | None = None
    snapshot_id: UUID | None = None
    request_payload: dict[str, object]
    attempt_id: UUID
    attempt_count: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    lease_expires_at: datetime


class JobCreationResult(_FrozenModel):
    job: TechnicalJob
    created: bool
