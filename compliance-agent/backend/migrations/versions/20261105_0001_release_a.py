"""Create Release A document, knowledge, job, and outbox state.

Revision ID: 20261105_0001
Revises:
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261105_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def upgrade() -> None:
    op.create_table(
        "stored_objects",
        sa.Column("stored_object_id", sa.Uuid(), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=False),
        sa.Column("checksum_sha256", sa.String(64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column(
            "document_format",
            sa.Enum("pdf", "txt", name="document_format", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "verified",
                "missing",
                name="stored_object_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        _timestamps()[0],
        sa.CheckConstraint(
            "length(checksum_sha256) = 64",
            name=op.f("ck_stored_objects_checksum_sha256_length"),
        ),
        sa.CheckConstraint(
            "byte_size > 0",
            name=op.f("ck_stored_objects_byte_size_positive"),
        ),
        sa.PrimaryKeyConstraint(
            "stored_object_id",
            name=op.f("pk_stored_objects"),
        ),
        sa.UniqueConstraint(
            "checksum_sha256",
            name=op.f("uq_stored_objects_checksum_sha256"),
        ),
        sa.UniqueConstraint(
            "object_key",
            name=op.f("uq_stored_objects_object_key"),
        ),
    )

    op.create_table(
        "contract_documents",
        sa.Column("contract_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(16), server_default="contract", nullable=False),
        sa.Column("stored_object_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column("source_version", sa.String(128), nullable=False),
        sa.Column("processing_revision", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "processing",
                "ready",
                "partial",
                "failed",
                name="contract_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "role = 'contract'",
            name=op.f("ck_contract_documents_role_contract"),
        ),
        sa.CheckConstraint(
            "processing_revision >= 0",
            name=op.f(
                "ck_contract_documents_processing_revision_non_negative"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.stored_object_id"],
            name=op.f(
                "fk_contract_documents_stored_object_id_stored_objects"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "contract_id",
            name=op.f("pk_contract_documents"),
        ),
        sa.UniqueConstraint(
            "access_scope",
            "stored_object_id",
            "source_version",
            name="uq_contract_scope_object_version",
        ),
        sa.UniqueConstraint(
            "contract_id",
            "access_scope",
            name="uq_contract_id_scope",
        ),
    )
    op.create_index(
        op.f("ix_contract_documents_access_scope"),
        "contract_documents",
        ["access_scope"],
    )
    op.create_index(
        op.f("ix_contract_documents_status"),
        "contract_documents",
        ["status"],
    )
    op.create_index(
        op.f("ix_contract_documents_stored_object_id"),
        "contract_documents",
        ["stored_object_id"],
    )

    op.create_table(
        "knowledge_bases",
        sa.Column("knowledge_base_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "archived",
                name="knowledge_base_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("applicability_metadata", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint(
            "knowledge_base_id",
            name=op.f("pk_knowledge_bases"),
        ),
        sa.UniqueConstraint(
            "knowledge_base_id",
            "access_scope",
            name="uq_knowledge_base_scope",
        ),
        sa.UniqueConstraint(
            "access_scope",
            "name",
            name="uq_knowledge_base_scope_name",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_bases_access_scope"),
        "knowledge_bases",
        ["access_scope"],
    )

    op.create_table(
        "knowledge_documents",
        sa.Column("knowledge_document_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(16), server_default="knowledge", nullable=False),
        sa.Column("stored_object_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.String(256), nullable=False),
        sa.Column("source_version", sa.String(128), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "indexing",
                "ready",
                "failed",
                name="knowledge_document_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "role = 'knowledge'",
            name=op.f("ck_knowledge_documents_role_knowledge"),
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.stored_object_id"],
            name=op.f(
                "fk_knowledge_documents_stored_object_id_stored_objects"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "knowledge_document_id",
            name=op.f("pk_knowledge_documents"),
        ),
        sa.UniqueConstraint(
            "access_scope",
            "source_id",
            "source_version",
            name="uq_knowledge_source_version",
        ),
        sa.UniqueConstraint(
            "knowledge_document_id",
            "access_scope",
            name="uq_knowledge_document_scope",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_documents_access_scope"),
        "knowledge_documents",
        ["access_scope"],
    )
    op.create_index(
        op.f("ix_knowledge_documents_stored_object_id"),
        "knowledge_documents",
        ["stored_object_id"],
    )

    op.create_table(
        "review_units",
        sa.Column("review_unit_id", sa.Uuid(), nullable=False),
        sa.Column("contract_id", sa.Uuid(), nullable=False),
        sa.Column("processing_revision", sa.Integer(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("label", sa.String(512), nullable=True),
        sa.Column("source_spans", sa.JSON(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "processing_revision >= 1",
            name=op.f("ck_review_units_processing_revision_positive"),
        ),
        sa.CheckConstraint(
            "order_index >= 0",
            name=op.f("ck_review_units_order_index_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["contract_documents.contract_id"],
            name=op.f("fk_review_units_contract_id_contract_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "review_unit_id",
            name=op.f("pk_review_units"),
        ),
        sa.UniqueConstraint(
            "contract_id",
            "processing_revision",
            "order_index",
            name="uq_review_unit_contract_revision_order",
        ),
    )
    op.create_index(
        op.f("ix_review_units_contract_id"),
        "review_units",
        ["contract_id"],
    )

    op.create_table(
        "knowledge_source_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_document_id", sa.Uuid(), nullable=False),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column("source_id", sa.String(256), nullable=False),
        sa.Column("source_version", sa.String(128), nullable=False),
        sa.Column("source_checksum_sha256", sa.String(64), nullable=False),
        sa.Column("index_profile", sa.String(512), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "indexing",
                "ready",
                "failed",
                name="knowledge_snapshot_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("collection_name", sa.String(256), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("vector_dimensions", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["knowledge_document_id", "access_scope"],
            [
                "knowledge_documents.knowledge_document_id",
                "knowledge_documents.access_scope",
            ],
            name="fk_snapshot_document_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_id",
            name=op.f("pk_knowledge_source_snapshots"),
        ),
        sa.UniqueConstraint(
            "snapshot_id",
            "access_scope",
            name="uq_snapshot_scope",
        ),
        sa.UniqueConstraint(
            "access_scope",
            "source_id",
            "source_version",
            "index_profile",
            name="uq_snapshot_source_version_profile",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_source_snapshots_knowledge_document_id"),
        "knowledge_source_snapshots",
        ["knowledge_document_id"],
    )

    op.create_table(
        "knowledge_base_memberships",
        sa.Column("membership_id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_base_id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column("membership_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "ready",
                "removed",
                name="knowledge_membership_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        *_timestamps(),
        sa.CheckConstraint(
            "membership_version >= 1",
            name=op.f(
                "ck_knowledge_base_memberships_membership_version_positive"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id", "access_scope"],
            [
                "knowledge_bases.knowledge_base_id",
                "knowledge_bases.access_scope",
            ],
            name="fk_membership_base_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "access_scope"],
            [
                "knowledge_source_snapshots.snapshot_id",
                "knowledge_source_snapshots.access_scope",
            ],
            name="fk_membership_snapshot_scope",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "membership_id",
            name=op.f("pk_knowledge_base_memberships"),
        ),
        sa.UniqueConstraint(
            "knowledge_base_id",
            "snapshot_id",
            "membership_version",
            name="uq_membership_version",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_base_memberships_knowledge_base_id"),
        "knowledge_base_memberships",
        ["knowledge_base_id"],
    )
    op.create_index(
        op.f("ix_knowledge_base_memberships_snapshot_id"),
        "knowledge_base_memberships",
        ["snapshot_id"],
    )

    op.create_table(
        "knowledge_reference_units",
        sa.Column("reference_unit_id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("structure_path", sa.JSON(), nullable=False),
        sa.Column("source_spans", sa.JSON(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "order_index >= 0",
            name=op.f(
                "ck_knowledge_reference_units_order_index_non_negative"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["knowledge_source_snapshots.snapshot_id"],
            name=op.f(
                "fk_knowledge_reference_units_snapshot_id_"
                "knowledge_source_snapshots"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "reference_unit_id",
            name=op.f("pk_knowledge_reference_units"),
        ),
        sa.UniqueConstraint(
            "snapshot_id",
            "order_index",
            name="uq_reference_snapshot_order",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_reference_units_snapshot_id"),
        "knowledge_reference_units",
        ["snapshot_id"],
    )

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "contract_processing",
                "knowledge_indexing",
                name="job_kind",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "queued",
                "running",
                "succeeded",
                "failed",
                "retry",
                name="job_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "stage",
            sa.Enum(
                "queued",
                "claimed",
                "processing",
                "indexing",
                "storing_result",
                "retry_wait",
                "completed",
                name="job_stage",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("access_scope", sa.String(128), nullable=False),
        sa.Column("contract_id", sa.Uuid(), nullable=True),
        sa.Column("snapshot_id", sa.Uuid(), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("result_schema_version", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("dispatch_generation", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamps(),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(contract_id IS NOT NULL AND snapshot_id IS NULL) OR "
            "(contract_id IS NULL AND snapshot_id IS NOT NULL)",
            name=op.f("ck_jobs_exactly_one_target"),
        ),
        sa.CheckConstraint(
            "(kind = 'contract_processing' AND contract_id IS NOT NULL) OR "
            "(kind = 'knowledge_indexing' AND snapshot_id IS NOT NULL)",
            name=op.f("ck_jobs_kind_matches_target"),
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_jobs_attempt_non_negative"),
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name=op.f("ck_jobs_max_attempts_positive"),
        ),
        sa.CheckConstraint(
            "dispatch_generation >= 0",
            name=op.f("ck_jobs_dispatch_generation_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["contract_id", "access_scope"],
            ["contract_documents.contract_id", "contract_documents.access_scope"],
            name="fk_job_contract_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "access_scope"],
            [
                "knowledge_source_snapshots.snapshot_id",
                "knowledge_source_snapshots.access_scope",
            ],
            name="fk_job_snapshot_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", name=op.f("pk_jobs")),
        sa.UniqueConstraint(
            "kind",
            "access_scope",
            "idempotency_key",
            name="uq_job_idempotency",
        ),
    )
    for column in ("access_scope", "contract_id", "snapshot_id", "status"):
        op.create_index(op.f(f"ix_jobs_{column}"), "jobs", [column])
    op.create_index(
        "uq_jobs_active_contract",
        "jobs",
        ["contract_id"],
        unique=True,
        postgresql_where=sa.text(
            "contract_id IS NOT NULL AND "
            "status IN ('queued', 'running', 'retry')"
        ),
    )
    op.create_index(
        "uq_jobs_active_snapshot",
        "jobs",
        ["snapshot_id"],
        unique=True,
        postgresql_where=sa.text(
            "snapshot_id IS NOT NULL AND "
            "status IN ('queued', 'running', 'retry')"
        ),
    )

    op.create_table(
        "outbox_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(128), nullable=False),
        sa.Column("topic", sa.String(255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("deduplication_key", sa.String(512), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "claimed",
                "published",
                name="outbox_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("publish_attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_by", sa.String(255), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "publish_attempts >= 0",
            name=op.f("ck_outbox_events_publish_attempts_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.job_id"],
            name=op.f("fk_outbox_events_job_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "event_id",
            name=op.f("pk_outbox_events"),
        ),
        sa.UniqueConstraint(
            "deduplication_key",
            name=op.f("uq_outbox_events_deduplication_key"),
        ),
    )
    op.create_index(
        op.f("ix_outbox_events_job_id"),
        "outbox_events",
        ["job_id"],
    )
    op.create_index(
        op.f("ix_outbox_events_status"),
        "outbox_events",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("jobs")
    op.drop_table("knowledge_reference_units")
    op.drop_table("knowledge_base_memberships")
    op.drop_table("knowledge_source_snapshots")
    op.drop_table("review_units")
    op.drop_table("knowledge_documents")
    op.drop_table("knowledge_bases")
    op.drop_table("contract_documents")
    op.drop_table("stored_objects")
