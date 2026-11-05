"""Value objects for binary storage identity."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from rag_core.ingestion.models import DocumentFormat


class StoredObjectStatus(StrEnum):
    """Describe whether a blob was verified in object storage."""

    VERIFIED = "verified"
    MISSING = "missing"


class StoredObject(BaseModel):
    """Represent one physical blob, never a business document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stored_object_id: UUID
    object_key: str = Field(min_length=1)
    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    byte_size: int = Field(gt=0)
    media_type: str = Field(min_length=1)
    document_format: DocumentFormat
    status: StoredObjectStatus
    created_at: datetime


class StoredObjectIngestionResult(BaseModel):
    """Return the blob identity and whether storage was reused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stored_object: StoredObject
    reused_existing: bool
