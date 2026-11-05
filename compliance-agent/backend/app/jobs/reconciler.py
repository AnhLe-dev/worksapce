"""Recover expired outbox claims and stale worker leases."""

from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.jobs.models import TechnicalJob
from app.jobs.repository import OutboxRepository
from app.jobs.service import JobService


class RecoveredJobFailureHandler(Protocol):
    def __call__(
        self,
        job: TechnicalJob,
        error_code: str,
        error_message: str,
    ) -> None:
        """Persist target failure after attempts are exhausted."""

        ...


class ReconciliationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    released_outbox_events: int = Field(ge=0)
    retried_jobs: int = Field(ge=0)
    failed_jobs: int = Field(ge=0)


class JobReconciler:
    def __init__(
        self,
        *,
        jobs: JobService,
        outbox: OutboxRepository,
        failure_handler: RecoveredJobFailureHandler | None = None,
    ) -> None:
        self._jobs = jobs
        self._outbox = outbox
        self._failure_handler = failure_handler

    def reconcile(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> ReconciliationReport:
        current = now or datetime.now(UTC)
        released = self._outbox.release_expired_claims(current)
        recovery = self._jobs.recover_expired(
            now=current,
            limit=limit,
        )

        if self._failure_handler is not None:
            for job in recovery.failed_jobs:
                self._failure_handler(
                    job,
                    job.error_code or "job_attempts_exhausted",
                    job.error_message or "Job attempts were exhausted",
                )

        return ReconciliationReport(
            released_outbox_events=released,
            retried_jobs=recovery.retried_jobs,
            failed_jobs=len(recovery.failed_jobs),
        )
