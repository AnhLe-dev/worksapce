"""SQLAlchemy records for generic contracts and review units."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.contracts.models import ContractStatus
from app.db.base import Base
from app.documents.db_models import StoredObjectRecord, enum_values


class ContractDocumentRecord(Base):
    """Persist contract identity separately from stored blob identity."""

    __tablename__ = "contract_documents"
    __table_args__ = (
        CheckConstraint("role = 'contract'", name="role_contract"),
        CheckConstraint(
            "processing_revision >= 0",
            name="processing_revision_non_negative",
        ),
        UniqueConstraint(
            "access_scope",
            "stored_object_id",
            "source_version",
            name="uq_contract_scope_object_version",
        ),
        UniqueConstraint(
            "contract_id",
            "access_scope",
            name="uq_contract_id_scope",
        ),
    )

    contract_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="contract",
        server_default="contract",
    )
    stored_object_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stored_objects.stored_object_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    access_scope: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    source_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    processing_revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    status: Mapped[ContractStatus] = mapped_column(
        SqlEnum(
            ContractStatus,
            name="contract_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=ContractStatus.UPLOADED,
        index=True,
    )
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
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

    stored_object: Mapped[StoredObjectRecord] = relationship()
    review_units: Mapped[list[ReviewUnitRecord]] = relationship(
        back_populates="contract",
        cascade="all, delete-orphan",
        order_by="ReviewUnitRecord.order_index",
    )


class ReviewUnitRecord(Base):
    """Persist generic review text and serialized exact source spans."""

    __tablename__ = "review_units"
    __table_args__ = (
        CheckConstraint(
            "processing_revision >= 1",
            name="processing_revision_positive",
        ),
        CheckConstraint("order_index >= 0", name="order_index_non_negative"),
        UniqueConstraint(
            "contract_id",
            "processing_revision",
            "order_index",
            name="uq_review_unit_contract_revision_order",
        ),
    )

    review_unit_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    contract_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("contract_documents.contract_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    processing_revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_spans: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        nullable=False,
    )
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    contract: Mapped[ContractDocumentRecord] = relationship(
        back_populates="review_units"
    )
