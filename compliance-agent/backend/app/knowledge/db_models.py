"""SQLAlchemy schema for PostgreSQL-owned knowledge lifecycle state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.documents.db_models import enum_values
from app.knowledge.models import (
    KnowledgeBaseStatus,
    KnowledgeDocumentStatus,
    MembershipStatus,
    SnapshotStatus,
)


class KnowledgeDocumentRecord(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        CheckConstraint("role = 'knowledge'", name="role_knowledge"),
        UniqueConstraint(
            "access_scope",
            "source_id",
            "source_version",
            name="uq_knowledge_source_version",
        ),
        UniqueConstraint(
            "knowledge_document_id",
            "access_scope",
            name="uq_knowledge_document_scope",
        ),
    )

    knowledge_document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="knowledge",
        server_default="knowledge",
    )
    stored_object_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stored_objects.stored_object_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_version: Mapped[str] = mapped_column(String(128), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    status: Mapped[KnowledgeDocumentStatus] = mapped_column(
        SqlEnum(
            KnowledgeDocumentStatus,
            name="knowledge_document_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=KnowledgeDocumentStatus.UPLOADED,
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


class KnowledgeBaseRecord(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "access_scope",
            name="uq_knowledge_base_scope",
        ),
        UniqueConstraint(
            "access_scope",
            "name",
            name="uq_knowledge_base_scope_name",
        ),
    )

    knowledge_base_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    status: Mapped[KnowledgeBaseStatus] = mapped_column(
        SqlEnum(
            KnowledgeBaseStatus,
            name="knowledge_base_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=KnowledgeBaseStatus.ACTIVE,
    )
    applicability_metadata: Mapped[dict[str, str]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
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


class KnowledgeSourceSnapshotRecord(Base):
    __tablename__ = "knowledge_source_snapshots"
    __table_args__ = (
        ForeignKeyConstraint(
            ["knowledge_document_id", "access_scope"],
            [
                "knowledge_documents.knowledge_document_id",
                "knowledge_documents.access_scope",
            ],
            ondelete="CASCADE",
            name="fk_snapshot_document_scope",
        ),
        UniqueConstraint(
            "snapshot_id",
            "access_scope",
            name="uq_snapshot_scope",
        ),
        UniqueConstraint(
            "access_scope",
            "source_id",
            "source_version",
            "index_profile",
            name="uq_snapshot_source_version_profile",
        ),
    )

    snapshot_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    knowledge_document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    index_profile: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[SnapshotStatus] = mapped_column(
        SqlEnum(
            SnapshotStatus,
            name="knowledge_snapshot_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=SnapshotStatus.PENDING,
    )
    collection_name: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vector_dimensions: Mapped[int | None] = mapped_column(
        Integer,
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


class KnowledgeBaseMembershipRecord(Base):
    __tablename__ = "knowledge_base_memberships"
    __table_args__ = (
        ForeignKeyConstraint(
            ["knowledge_base_id", "access_scope"],
            [
                "knowledge_bases.knowledge_base_id",
                "knowledge_bases.access_scope",
            ],
            ondelete="CASCADE",
            name="fk_membership_base_scope",
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "access_scope"],
            [
                "knowledge_source_snapshots.snapshot_id",
                "knowledge_source_snapshots.access_scope",
            ],
            ondelete="RESTRICT",
            name="fk_membership_snapshot_scope",
        ),
        UniqueConstraint(
            "knowledge_base_id",
            "snapshot_id",
            "membership_version",
            name="uq_membership_version",
        ),
        CheckConstraint(
            "membership_version >= 1",
            name="membership_version_positive",
        ),
    )

    membership_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    membership_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    status: Mapped[MembershipStatus] = mapped_column(
        SqlEnum(
            MembershipStatus,
            name="knowledge_membership_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=MembershipStatus.PENDING,
    )
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


class KnowledgeReferenceUnitRecord(Base):
    __tablename__ = "knowledge_reference_units"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "order_index",
            name="uq_reference_snapshot_order",
        ),
        CheckConstraint("order_index >= 0", name="order_index_non_negative"),
    )

    reference_unit_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "knowledge_source_snapshots.snapshot_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    structure_path: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_spans: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        nullable=False,
    )
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
