"""Transactional job creation and lease-owned state transitions."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.jobs.db_models import OutboxEventRecord, TechnicalJobRecord
from app.jobs.models import (
    JobClaim,
    JobCreationResult,
    JobKind,
    JobStage,
    JobStatus,
    TechnicalJob,
)
from app.jobs.outbox_models import OutboxStatus
from app.jobs.repository import (
    JobRepository,
    OutboxRepository,
    to_job,
)


class JobError(RuntimeError):
    """Base class for safe job lifecycle failures."""


class JobTargetNotFoundError(JobError, LookupError):
    pass


class JobLeaseActiveError(JobError):
    pass


class JobAttemptOwnershipError(JobError):
    pass


class JobAttemptsExhaustedError(JobError):
    pass


@dataclass(frozen=True, slots=True)
class JobRecoveryResult:
    retried_jobs: int
    failed_jobs: tuple[TechnicalJob, ...]


class JobService:
    """Keep trusted payload in DB and broker messages minimal."""

    def __init__(
        self,
        *,
        jobs: JobRepository,
        outbox: OutboxRepository,
        lease_seconds: int,
        max_attempts: int,
    ) -> None:
        if lease_seconds < 1 or max_attempts < 1:
            raise ValueError("Lease and maximum attempts must be positive")

        self._jobs = jobs
        self._outbox = outbox
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)

    def _add_dispatch_event(
        self,
        job: TechnicalJobRecord,
        *,
        available_at: datetime,
    ) -> None:
        self._outbox.add(
            OutboxEventRecord(
                job_id=job.job_id,
                event_type="job.dispatch",
                topic="jobs.dispatch",
                payload={"job_id": str(job.job_id)},
                deduplication_key=(
                    f"job:{job.job_id}:dispatch:{job.dispatch_generation}"
                ),
                status=OutboxStatus.PENDING,
                available_at=available_at,
            )
        )

    def _create(
        self,
        *,
        kind: JobKind,
        access_scope: str,
        idempotency_key: str,
        request_payload: dict[str, object],
        contract_id: UUID | None,
        snapshot_id: UUID | None,
        now: datetime,
    ) -> JobCreationResult:
        scope = access_scope.strip()
        key = idempotency_key.strip()

        if not scope or not key:
            raise ValueError("Scope and idempotency key are required")

        existing = self._jobs.get_by_idempotency(
            kind=kind,
            access_scope=scope,
            idempotency_key=key,
        )

        if existing is not None:
            return JobCreationResult(job=to_job(existing), created=False)

        if kind in {
            JobKind.CONTRACT_PROCESSING,
            JobKind.CONTRACT_REVIEW,
        }:
            if contract_id is None or not self._jobs.contract_exists(
                contract_id,
                scope,
            ):
                raise JobTargetNotFoundError("Contract was not found")
        elif snapshot_id is None or not self._jobs.snapshot_exists(
            snapshot_id,
            scope,
        ):
            raise JobTargetNotFoundError("Knowledge snapshot was not found")

        record, inserted = self._jobs.add_job_idempotently(
            TechnicalJobRecord(
                kind=kind,
                status=JobStatus.QUEUED,
                stage=JobStage.QUEUED,
                access_scope=scope,
                contract_id=contract_id,
                snapshot_id=snapshot_id,
                request_payload=request_payload,
                idempotency_key=key,
                dispatch_generation=0,
                attempt_count=0,
                max_attempts=self._max_attempts,
            )
        )

        if not inserted:
            return JobCreationResult(job=to_job(record), created=False)

        self._add_dispatch_event(record, available_at=now)
        return JobCreationResult(job=to_job(record), created=True)

    def create_contract_processing(
        self,
        *,
        contract_id: UUID,
        access_scope: str,
        expected_revision: int,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> JobCreationResult:
        if expected_revision < 0:
            raise ValueError("Expected revision must not be negative")

        return self._create(
            kind=JobKind.CONTRACT_PROCESSING,
            access_scope=access_scope,
            idempotency_key=idempotency_key,
            request_payload={"expected_revision": expected_revision},
            contract_id=contract_id,
            snapshot_id=None,
            now=now or self._now(),
        )

    def create_knowledge_indexing(
        self,
        *,
        snapshot_id: UUID,
        access_scope: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> JobCreationResult:
        return self._create(
            kind=JobKind.KNOWLEDGE_INDEXING,
            access_scope=access_scope,
            idempotency_key=idempotency_key,
            request_payload={},
            contract_id=None,
            snapshot_id=snapshot_id,
            now=now or self._now(),
        )

    def create_contract_review(
        self,
        *,
        contract_id: UUID,
        access_scope: str,
        knowledge_base_id: UUID,
        expected_processing_revision: int,
        review_unit_ids: tuple[UUID, ...],
        question: str | None,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> JobCreationResult:
        """Persist only client intent; the worker re-resolves authority."""

        if expected_processing_revision < 1:
            raise ValueError(
                "Expected processing revision must be positive"
            )

        if not review_unit_ids or len(review_unit_ids) > 100:
            raise ValueError(
                "Review job requires between 1 and 100 review units"
            )

        if len(set(review_unit_ids)) != len(review_unit_ids):
            raise ValueError("Review-unit IDs must be unique")

        normalized_question = question.strip() if question else None

        return self._create(
            kind=JobKind.CONTRACT_REVIEW,
            access_scope=access_scope,
            idempotency_key=idempotency_key,
            request_payload={
                "knowledge_base_id": str(knowledge_base_id),
                "expected_processing_revision": (
                    expected_processing_revision
                ),
                "review_unit_ids": [
                    str(review_unit_id)
                    for review_unit_id in review_unit_ids
                ],
                "question": normalized_question,
            },
            contract_id=contract_id,
            snapshot_id=None,
            now=now or self._now(),
        )

    def get_job(self, job_id: UUID, access_scope: str) -> TechnicalJob:
        record = self._jobs.get_record(
            job_id=job_id,
            access_scope=access_scope,
        )

        if record is None:
            raise JobTargetNotFoundError("Job was not found")

        return to_job(record)

    def claim(
        self,
        job_id: UUID,
        *,
        now: datetime | None = None,
    ) -> JobClaim | None:
        current = now or self._now()
        record = self._jobs.get_record(job_id=job_id, for_update=True)

        if record is None:
            raise JobTargetNotFoundError("Job was not found")

        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            return None

        if (
            record.status is JobStatus.RUNNING
            and record.lease_expires_at is not None
            and self._utc(record.lease_expires_at) > self._utc(current)
        ):
            raise JobLeaseActiveError("Job is owned by another live attempt")

        if record.attempt_count >= record.max_attempts:
            record.status = JobStatus.FAILED
            record.stage = JobStage.COMPLETED
            record.error_code = "job_attempts_exhausted"
            record.error_message = "Job exhausted every configured attempt"
            record.finished_at = current
            record.lease_expires_at = None
            self._jobs.flush()
            raise JobAttemptsExhaustedError(record.error_message)

        attempt_id = uuid4()
        lease_expires_at = current + timedelta(seconds=self._lease_seconds)
        record.status = JobStatus.RUNNING
        record.stage = JobStage.CLAIMED
        record.attempt_id = attempt_id
        record.attempt_count += 1
        record.lease_expires_at = lease_expires_at
        record.heartbeat_at = current
        record.started_at = record.started_at or current
        record.finished_at = None
        record.error_code = None
        record.error_message = None
        self._jobs.flush()
        return JobClaim(
            job_id=record.job_id,
            kind=record.kind,
            access_scope=record.access_scope,
            contract_id=record.contract_id,
            snapshot_id=record.snapshot_id,
            request_payload=dict(record.request_payload),
            attempt_id=attempt_id,
            attempt_count=record.attempt_count,
            max_attempts=record.max_attempts,
            lease_expires_at=lease_expires_at,
        )

    def _owned(
        self,
        *,
        job_id: UUID,
        attempt_id: UUID,
        now: datetime,
    ) -> TechnicalJobRecord:
        record = self._jobs.get_record(job_id=job_id, for_update=True)

        if (
            record is None
            or record.status is not JobStatus.RUNNING
            or record.attempt_id != attempt_id
            or record.lease_expires_at is None
            or self._utc(record.lease_expires_at) <= self._utc(now)
        ):
            raise JobAttemptOwnershipError(
                "Worker attempt does not own a live job lease"
            )

        return record

    def heartbeat(
        self,
        *,
        job_id: UUID,
        attempt_id: UUID,
        stage: JobStage,
        now: datetime | None = None,
    ) -> datetime:
        current = now or self._now()
        record = self._owned(
            job_id=job_id,
            attempt_id=attempt_id,
            now=current,
        )
        record.stage = stage
        record.heartbeat_at = current
        record.lease_expires_at = current + timedelta(
            seconds=self._lease_seconds
        )
        self._jobs.flush()
        return record.lease_expires_at

    def succeed(
        self,
        *,
        job_id: UUID,
        attempt_id: UUID,
        result_payload: dict[str, object],
        now: datetime | None = None,
    ) -> TechnicalJob:
        current = now or self._now()
        record = self._owned(
            job_id=job_id,
            attempt_id=attempt_id,
            now=current,
        )
        record.status = JobStatus.SUCCEEDED
        record.stage = JobStage.COMPLETED
        record.result_payload = result_payload
        record.finished_at = current
        record.lease_expires_at = None
        record.heartbeat_at = current
        self._jobs.flush()
        return to_job(record)

    def fail(
        self,
        *,
        job_id: UUID,
        attempt_id: UUID,
        error_code: str,
        error_message: str,
        now: datetime | None = None,
    ) -> TechnicalJob:
        current = now or self._now()
        record = self._owned(
            job_id=job_id,
            attempt_id=attempt_id,
            now=current,
        )
        record.status = JobStatus.FAILED
        record.stage = JobStage.COMPLETED
        record.error_code = error_code
        record.error_message = error_message
        record.finished_at = current
        record.lease_expires_at = None
        self._jobs.flush()
        return to_job(record)

    def retry(
        self,
        *,
        job_id: UUID,
        attempt_id: UUID,
        error_code: str,
        error_message: str,
        delay_seconds: int,
        now: datetime | None = None,
    ) -> TechnicalJob:
        if delay_seconds < 0:
            raise ValueError("Retry delay must not be negative")

        current = now or self._now()
        record = self._owned(
            job_id=job_id,
            attempt_id=attempt_id,
            now=current,
        )

        if record.attempt_count >= record.max_attempts:
            return self.fail(
                job_id=job_id,
                attempt_id=attempt_id,
                error_code="job_attempts_exhausted",
                error_message="Job exhausted every configured attempt",
                now=current,
            )

        record.status = JobStatus.RETRY
        record.stage = JobStage.RETRY_WAIT
        record.error_code = error_code
        record.error_message = error_message
        record.attempt_id = None
        record.lease_expires_at = None
        record.dispatch_generation += 1
        self._add_dispatch_event(
            record,
            available_at=current + timedelta(seconds=delay_seconds),
        )
        self._jobs.flush()
        return to_job(record)

    def recover_expired(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> JobRecoveryResult:
        """Requeue expired attempts or fail them after exhaustion."""

        if limit < 1 or limit > 1000:
            raise ValueError("Recovery limit must be between 1 and 1000")

        current = now or self._now()
        retried = 0
        failed_jobs: list[TechnicalJob] = []

        for record in self._jobs.list_expired_running(current, limit=limit):
            if record.attempt_count >= record.max_attempts:
                record.status = JobStatus.FAILED
                record.stage = JobStage.COMPLETED
                record.error_code = "job_attempts_exhausted"
                record.error_message = (
                    "Job lease expired after every configured attempt"
                )
                record.finished_at = current
                record.lease_expires_at = None
                failed_jobs.append(to_job(record))
                continue

            record.status = JobStatus.RETRY
            record.stage = JobStage.RETRY_WAIT
            record.error_code = "job_lease_expired"
            record.error_message = "Worker lease expired before completion"
            record.attempt_id = None
            record.lease_expires_at = None
            record.dispatch_generation += 1
            self._add_dispatch_event(record, available_at=current)
            retried += 1

        self._jobs.flush()
        return JobRecoveryResult(
            retried_jobs=retried,
            failed_jobs=tuple(failed_jobs),
        )
