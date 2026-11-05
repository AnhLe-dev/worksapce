"""SQLAlchemy records for technical jobs and their dispatch outbox."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.documents.db_models import enum_values
from app.jobs.models import JobKind, JobStage, JobStatus
from app.jobs.outbox_models import OutboxStatus


class TechnicalJobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["contract_id", "access_scope"],
            [
                "contract_documents.contract_id",
                "contract_documents.access_scope",
            ],
            ondelete="CASCADE",
            name="fk_job_contract_scope",
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "access_scope"],
            [
                "knowledge_source_snapshots.snapshot_id",
                "knowledge_source_snapshots.access_scope",
            ],
            ondelete="CASCADE",
            name="fk_job_snapshot_scope",
        ),
        CheckConstraint(
            "(contract_id IS NOT NULL AND snapshot_id IS NULL) OR "
            "(contract_id IS NULL AND snapshot_id IS NOT NULL)",
            name="exactly_one_target",
        ),
        CheckConstraint(
            "(kind IN ('contract_processing', 'contract_review') "
            "AND contract_id IS NOT NULL) OR "
            "(kind = 'knowledge_indexing' AND snapshot_id IS NOT NULL)",
            name="kind_matches_target",
        ),
        CheckConstraint("attempt_count >= 0", name="attempt_non_negative"),
        CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
        CheckConstraint(
            "dispatch_generation >= 0",
            name="dispatch_generation_non_negative",
        ),
        UniqueConstraint(
            "kind",
            "access_scope",
            "idempotency_key",
            name="uq_job_idempotency",
        ),
        Index(
            "uq_jobs_active_contract",
            "contract_id",
            unique=True,
            postgresql_where=text(
                "contract_id IS NOT NULL AND "
                "status IN ('queued', 'running', 'retry')"
            ),
            sqlite_where=text(
                "contract_id IS NOT NULL AND "
                "status IN ('queued', 'running', 'retry')"
            ),
        ),
        Index(
            "uq_jobs_active_snapshot",
            "snapshot_id",
            unique=True,
            postgresql_where=text(
                "snapshot_id IS NOT NULL AND "
                "status IN ('queued', 'running', 'retry')"
            ),
            sqlite_where=text(
                "snapshot_id IS NOT NULL AND "
                "status IN ('queued', 'running', 'retry')"
            ),
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    kind: Mapped[JobKind] = mapped_column(
        SqlEnum(
            JobKind,
            name="job_kind",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
    )
    status: Mapped[JobStatus] = mapped_column(
        SqlEnum(
            JobStatus,
            name="job_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )
    stage: Mapped[JobStage] = mapped_column(
        SqlEnum(
            JobStage,
            name="job_stage",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=JobStage.QUEUED,
    )
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    contract_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
    )
    snapshot_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
    )
    request_payload: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
    )
    result_payload: Mapped[dict[str, object] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    result_schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    dispatch_generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    attempt_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class OutboxEventRecord(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (
        CheckConstraint(
            "publish_attempts >= 0",
            name="publish_attempts_non_negative",
        ),
    )

    event_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    job_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    deduplication_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
    )
    status: Mapped[OutboxStatus] = mapped_column(
        SqlEnum(
            OutboxStatus,
            name="outbox_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=OutboxStatus.PENDING,
        index=True,
    )
    publish_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    claimed_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
