"""Canonical technical job HTTP schema."""

from datetime import datetime
from uuid import UUID

from app.api.schemas.common import ApiSchema
from app.jobs.models import JobKind, JobStage, JobStatus


class JobResponse(ApiSchema):
    job_id: UUID
    kind: JobKind
    status: JobStatus
    stage: JobStage
    contract_id: UUID | None
    snapshot_id: UUID | None
    attempt_count: int
    max_attempts: int
    result_payload: dict[str, object] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
