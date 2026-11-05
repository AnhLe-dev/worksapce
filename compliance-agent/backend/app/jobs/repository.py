"""Persistence operations for technical jobs and outbox events."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.db_models import ContractDocumentRecord
from app.jobs.db_models import OutboxEventRecord, TechnicalJobRecord
from app.jobs.models import JobKind, JobStatus, TechnicalJob
from app.jobs.outbox_models import OutboxEvent, OutboxStatus
from app.knowledge.db_models import KnowledgeSourceSnapshotRecord


def to_job(record: TechnicalJobRecord) -> TechnicalJob:
    return TechnicalJob(
        job_id=record.job_id,
        kind=record.kind,
        status=record.status,
        stage=record.stage,
        access_scope=record.access_scope,
        contract_id=record.contract_id,
        snapshot_id=record.snapshot_id,
        request_payload=dict(record.request_payload),
        result_payload=(
            None
            if record.result_payload is None
            else dict(record.result_payload)
        ),
        result_schema_version=record.result_schema_version,
        idempotency_key=record.idempotency_key,
        dispatch_generation=record.dispatch_generation,
        attempt_id=record.attempt_id,
        attempt_count=record.attempt_count,
        max_attempts=record.max_attempts,
        lease_expires_at=record.lease_expires_at,
        heartbeat_at=record.heartbeat_at,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


def to_outbox(record: OutboxEventRecord) -> OutboxEvent:
    return OutboxEvent(
        event_id=record.event_id,
        job_id=record.job_id,
        event_type=record.event_type,
        topic=record.topic,
        payload=dict(record.payload),
        deduplication_key=record.deduplication_key,
        status=record.status,
        publish_attempts=record.publish_attempts,
        available_at=record.available_at,
        claimed_by=record.claimed_by,
        claim_expires_at=record.claim_expires_at,
        published_at=record.published_at,
        last_error=record.last_error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _lock(statement: Select, for_update: bool) -> Select:
        return statement.with_for_update() if for_update else statement

    def get_record(
        self,
        *,
        job_id: UUID,
        access_scope: str | None = None,
        for_update: bool = False,
    ) -> TechnicalJobRecord | None:
        statement = select(TechnicalJobRecord).where(
            TechnicalJobRecord.job_id == job_id
        )

        if access_scope is not None:
            statement = statement.where(
                TechnicalJobRecord.access_scope == access_scope
            )

        return self._session.scalar(self._lock(statement, for_update))

    def get_by_idempotency(
        self,
        *,
        kind: JobKind,
        access_scope: str,
        idempotency_key: str,
    ) -> TechnicalJobRecord | None:
        return self._session.scalar(
            select(TechnicalJobRecord).where(
                TechnicalJobRecord.kind == kind,
                TechnicalJobRecord.access_scope == access_scope,
                TechnicalJobRecord.idempotency_key == idempotency_key,
            )
        )

    def contract_exists(self, contract_id: UUID, access_scope: str) -> bool:
        return (
            self._session.scalar(
                select(ContractDocumentRecord.contract_id).where(
                    ContractDocumentRecord.contract_id == contract_id,
                    ContractDocumentRecord.access_scope == access_scope,
                )
            )
            is not None
        )

    def snapshot_exists(self, snapshot_id: UUID, access_scope: str) -> bool:
        return (
            self._session.scalar(
                select(KnowledgeSourceSnapshotRecord.snapshot_id).where(
                    KnowledgeSourceSnapshotRecord.snapshot_id == snapshot_id,
                    KnowledgeSourceSnapshotRecord.access_scope == access_scope,
                )
            )
            is not None
        )

    def add_job(self, record: TechnicalJobRecord) -> TechnicalJobRecord:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return record

    def add_job_idempotently(
        self,
        record: TechnicalJobRecord,
    ) -> tuple[TechnicalJobRecord, bool]:
        """Insert under a savepoint or return the idempotency winner."""

        try:
            with self._session.begin_nested():
                inserted = self.add_job(record)
        except IntegrityError:
            existing = self.get_by_idempotency(
                kind=record.kind,
                access_scope=record.access_scope,
                idempotency_key=record.idempotency_key,
            )

            if existing is None:
                raise

            return existing, False

        return inserted, True

    def flush(self) -> None:
        self._session.flush()

    def list_expired_running(
        self,
        now: datetime,
        *,
        limit: int,
    ) -> tuple[TechnicalJobRecord, ...]:
        records = self._session.scalars(
            select(TechnicalJobRecord)
            .where(
                TechnicalJobRecord.status == JobStatus.RUNNING,
                TechnicalJobRecord.lease_expires_at.is_not(None),
                TechnicalJobRecord.lease_expires_at <= now,
            )
            .order_by(TechnicalJobRecord.lease_expires_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
        return tuple(records)


class OutboxRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, record: OutboxEventRecord) -> OutboxEventRecord:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return record

    def get_record(self, event_id: UUID) -> OutboxEventRecord | None:
        return self._session.get(OutboxEventRecord, event_id)

    def flush(self) -> None:
        self._session.flush()

    def mark_published(
        self,
        record: OutboxEventRecord,
        *,
        published_at: datetime,
    ) -> None:
        record.status = OutboxStatus.PUBLISHED
        record.published_at = published_at
        record.claimed_by = None
        record.claim_expires_at = None
        record.last_error = None
        self._session.flush()

    def release_after_failure(
        self,
        record: OutboxEventRecord,
        *,
        available_at: datetime,
        error_message: str,
    ) -> None:
        record.status = OutboxStatus.PENDING
        record.available_at = available_at
        record.claimed_by = None
        record.claim_expires_at = None
        record.last_error = error_message
        self._session.flush()

    def claim_pending(
        self,
        *,
        now: datetime,
        claimed_by: str,
        claim_expires_at: datetime,
        limit: int,
    ) -> tuple[OutboxEventRecord, ...]:
        records = self._session.scalars(
            select(OutboxEventRecord)
            .where(
                OutboxEventRecord.status == OutboxStatus.PENDING,
                OutboxEventRecord.available_at <= now,
            )
            .order_by(OutboxEventRecord.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()

        for record in records:
            record.status = OutboxStatus.CLAIMED
            record.claimed_by = claimed_by
            record.claim_expires_at = claim_expires_at
            record.publish_attempts += 1

        self._session.flush()
        return tuple(records)

    def release_expired_claims(self, now: datetime) -> int:
        records = self._session.scalars(
            select(OutboxEventRecord).where(
                OutboxEventRecord.status == OutboxStatus.CLAIMED,
                OutboxEventRecord.claim_expires_at.is_not(None),
                OutboxEventRecord.claim_expires_at <= now,
            )
        ).all()

        for record in records:
            record.status = OutboxStatus.PENDING
            record.claimed_by = None
            record.claim_expires_at = None

        self._session.flush()
        return len(records)

    def list_unpublished_for_job(
        self,
        job_id: UUID,
    ) -> tuple[OutboxEvent, ...]:
        records = self._session.scalars(
            select(OutboxEventRecord).where(
                OutboxEventRecord.job_id == job_id,
                or_(
                    OutboxEventRecord.status == OutboxStatus.PENDING,
                    OutboxEventRecord.status == OutboxStatus.CLAIMED,
                ),
            )
        ).all()
        return tuple(to_outbox(record) for record in records)
