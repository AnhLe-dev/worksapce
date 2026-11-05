"""SQLAlchemy records for physical object metadata."""

from datetime import datetime
from uuid import UUID, uuid4

from rag_core.ingestion.models import DocumentFormat
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Integer,
    String,
    Uuid,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.documents.models import StoredObjectStatus


def enum_values(enum_class: type) -> list[str]:
    """Persist stable enum values instead of Python member names."""

    return [member.value for member in enum_class]


class StoredObjectRecord(Base):
    """Persist blob identity without role, owner, or access scope."""

    __tablename__ = "stored_objects"
    __table_args__ = (
        CheckConstraint(
            "length(checksum_sha256) = 64",
            name="checksum_sha256_length",
        ),
        CheckConstraint(
            "byte_size > 0",
            name="byte_size_positive",
        ),
    )

    stored_object_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    object_key: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )
    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    document_format: Mapped[DocumentFormat] = mapped_column(
        SqlEnum(
            DocumentFormat,
            name="document_format",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
    )
    status: Mapped[StoredObjectStatus] = mapped_column(
        SqlEnum(
            StoredObjectStatus,
            name="stored_object_status",
            values_callable=enum_values,
            native_enum=False,
        ),
        nullable=False,
        default=StoredObjectStatus.VERIFIED,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
