"""Claim and publish pending outbox events with retry-safe state."""

from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.jobs.repository import OutboxRepository


class MessagePublisher(Protocol):
    def publish(
        self,
        *,
        topic: str,
        payload: dict[str, object],
        message_id: str,
    ) -> None:
        """Publish one idempotently identified broker message."""


class MessagePublishError(RuntimeError):
    """Represent a provider failure that the relay may safely retry."""


class DispatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claimed: int = Field(ge=0)
    published: int = Field(ge=0)
    failed: int = Field(ge=0)


class OutboxDispatcher:
    def __init__(
        self,
        *,
        repository: OutboxRepository,
        publisher: MessagePublisher,
        relay_id: str,
        lease_seconds: int,
        batch_size: int,
    ) -> None:
        normalized_relay_id = relay_id.strip()

        if not normalized_relay_id:
            raise ValueError("Relay ID must not be empty")

        if lease_seconds < 1 or batch_size < 1:
            raise ValueError("Relay lease and batch size must be positive")

        self._repository = repository
        self._publisher = publisher
        self._relay_id = normalized_relay_id
        self._lease_seconds = lease_seconds
        self._batch_size = batch_size

    def dispatch(self, *, now: datetime | None = None) -> DispatchReport:
        current = now or datetime.now(UTC)
        records = self._repository.claim_pending(
            now=current,
            claimed_by=self._relay_id,
            claim_expires_at=current
            + timedelta(seconds=self._lease_seconds),
            limit=self._batch_size,
        )
        published = 0
        failed = 0

        for record in records:
            try:
                self._publisher.publish(
                    topic=record.topic,
                    payload=dict(record.payload),
                    message_id=record.deduplication_key,
                )
            except MessagePublishError as error:
                delay = min(2 ** record.publish_attempts, 60)
                self._repository.release_after_failure(
                    record,
                    available_at=current + timedelta(seconds=delay),
                    error_message=type(error).__name__,
                )
                failed += 1
            else:
                self._repository.mark_published(
                    record,
                    published_at=current,
                )
                published += 1

        return DispatchReport(
            claimed=len(records),
            published=published,
            failed=failed,
        )
