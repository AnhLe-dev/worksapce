"""Release A technical job status endpoint."""

from uuid import UUID

from fastapi import APIRouter

from app.api.dependencies import RuntimeDependency, SessionDependency
from app.api.errors import ApiError
from app.api.schemas.jobs import JobResponse
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService, JobTargetNotFoundError

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: UUID,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> JobResponse:
    service = JobService(
        jobs=JobRepository(session),
        outbox=OutboxRepository(session),
        lease_seconds=runtime.settings.job_lease_seconds,
        max_attempts=runtime.settings.job_max_attempts,
    )

    try:
        job = service.get_job(
            job_id,
            runtime.settings.system_access_scope,
        )
    except JobTargetNotFoundError as error:
        raise ApiError(404, "job_not_found", str(error)) from error

    return JobResponse.model_validate(job)
