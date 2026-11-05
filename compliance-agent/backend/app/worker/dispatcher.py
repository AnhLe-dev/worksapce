"""Idempotent worker dispatch driven only by a durable job ID."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from celery import Celery
from kombu.exceptions import OperationalError

from app.jobs.models import (
    JobClaim,
    JobKind,
    JobStage,
    JobStatus,
    TechnicalJob,
)
from app.jobs.outbox_dispatcher import MessagePublishError
from app.jobs.service import JobService


class JobHandler(Protocol):
    def __call__(self, claim: JobClaim) -> dict[str, object]:
        """Execute one already-authorized DB-backed job claim."""


class JobFailureHandler(Protocol):
    def __call__(
        self,
        claim: JobClaim,
        error_code: str,
        error_message: str,
    ) -> None:
        """Persist target-domain failure in the job transaction."""


class RetryableJobError(RuntimeError):
    """Ask the worker dispatcher to create a durable retry event."""


class PermanentJobError(RuntimeError):
    """Represent a validated non-retryable handler failure."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class CeleryPublisher:
    """Publish the outbox envelope without copying trusted DB payload."""

    def __init__(self, celery_app: Celery) -> None:
        self._celery_app = celery_app

    def publish(
        self,
        *,
        topic: str,
        payload: dict[str, object],
        message_id: str,
    ) -> None:
        try:
            self._celery_app.send_task(
                topic,
                kwargs=payload,
                task_id=message_id,
            )
        except (OperationalError, ConnectionError, TimeoutError) as error:
            raise MessagePublishError(
                "Celery broker publish failed"
            ) from error


class WorkerDispatcher:
    def __init__(
        self,
        *,
        jobs: JobService,
        handlers: dict[JobKind, JobHandler],
        failure_handler: JobFailureHandler | None = None,
    ) -> None:
        missing = set(JobKind).difference(handlers)

        if missing:
            names = ", ".join(sorted(kind.value for kind in missing))
            raise ValueError(f"Missing job handlers: {names}")

        self._jobs = jobs
        self._handlers = dict(handlers)
        self._failure_handler = failure_handler

    def _record_target_failure(
        self,
        claim: JobClaim,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        if self._failure_handler is not None:
            self._failure_handler(claim, error_code, error_message)

    def execute(
        self,
        job_id: UUID,
        *,
        now: datetime | None = None,
    ) -> TechnicalJob | None:
        """Convenience path for tests using one caller-owned transaction."""

        claim = self.claim(job_id, now=now)

        if claim is None:
            return None

        return self.execute_claim(claim, now=now)

    def claim(
        self,
        job_id: UUID,
        *,
        now: datetime | None = None,
    ) -> JobClaim | None:
        """Claim a job; production callers must commit before execution."""

        return self._jobs.claim(job_id, now=now)

    def execute_claim(
        self,
        claim: JobClaim,
        *,
        now: datetime | None = None,
    ) -> TechnicalJob:
        """Execute an already committed lease in a second transaction."""

        self._jobs.heartbeat(
            job_id=claim.job_id,
            attempt_id=claim.attempt_id,
            stage=(
                JobStage.INDEXING
                if claim.kind is JobKind.KNOWLEDGE_INDEXING
                else JobStage.PROCESSING
            ),
            now=now,
        )

        try:
            result = self._handlers[claim.kind](claim)
        except RetryableJobError as error:
            job = self._jobs.retry(
                job_id=claim.job_id,
                attempt_id=claim.attempt_id,
                error_code="dependency_unavailable",
                error_message=str(error) or "Dependency unavailable",
                delay_seconds=min(2 ** claim.attempt_count, 60),
                now=now,
            )

            if job.status is JobStatus.FAILED:
                self._record_target_failure(
                    claim,
                    error_code=job.error_code or "job_attempts_exhausted",
                    error_message=(
                        job.error_message or "Job attempts were exhausted"
                    ),
                )

            return job
        except PermanentJobError as error:
            self._record_target_failure(
                claim,
                error_code=error.code,
                error_message=error.message,
            )
            return self._jobs.fail(
                job_id=claim.job_id,
                attempt_id=claim.attempt_id,
                error_code=error.code,
                error_message=error.message,
                now=now,
            )

        self._jobs.heartbeat(
            job_id=claim.job_id,
            attempt_id=claim.attempt_id,
            stage=JobStage.STORING_RESULT,
            now=now,
        )
        return self._jobs.succeed(
            job_id=claim.job_id,
            attempt_id=claim.attempt_id,
            result_payload=result,
            now=now,
        )
