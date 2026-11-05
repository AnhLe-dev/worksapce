"""Contracts for durable job-dispatch outbox events."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OutboxStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PUBLISHED = "published"


class OutboxEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID
    job_id: UUID
    event_type: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    payload: dict[str, object]
    deduplication_key: str = Field(min_length=1)
    status: OutboxStatus
    publish_attempts: int = Field(ge=0)
    available_at: datetime
    claimed_by: str | None = None
    claim_expires_at: datetime | None = None
    published_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_payload_and_state(self) -> "OutboxEvent":
        if self.payload != {"job_id": str(self.job_id)}:
            raise ValueError("Outbox payload may contain only job_id")

        if self.status is OutboxStatus.CLAIMED and (
            not self.claimed_by or self.claim_expires_at is None
        ):
            raise ValueError("CLAIMED event requires lease ownership")

        if self.status is OutboxStatus.PUBLISHED and self.published_at is None:
            raise ValueError("PUBLISHED event requires published_at")

        return self
