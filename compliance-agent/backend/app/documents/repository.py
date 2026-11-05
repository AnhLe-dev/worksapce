"""Persistence operations for stored binary objects."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.documents.db_models import StoredObjectRecord
from app.documents.models import StoredObject


def to_stored_object(record: StoredObjectRecord) -> StoredObject:
    """Convert a persistent record to an immutable value object."""

    return StoredObject(
        stored_object_id=record.stored_object_id,
        object_key=record.object_key,
        checksum_sha256=record.checksum_sha256,
        byte_size=record.byte_size,
        media_type=record.media_type,
        document_format=record.document_format,
        status=record.status,
        created_at=record.created_at,
    )


class StoredObjectRepository:
    """Read and stage physical object metadata in one transaction."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_record_by_id(
        self,
        stored_object_id: UUID,
    ) -> StoredObjectRecord | None:
        """Load one blob record by database identity."""

        return self._session.get(StoredObjectRecord, stored_object_id)

    def get_by_checksum(self, checksum_sha256: str) -> StoredObject | None:
        """Resolve an already verified physical blob by content hash."""

        record = self._session.scalar(
            select(StoredObjectRecord).where(
                StoredObjectRecord.checksum_sha256 == checksum_sha256
            )
        )
        return None if record is None else to_stored_object(record)

    def add(self, record: StoredObjectRecord) -> StoredObject:
        """Stage a new blob record and materialize generated fields."""

        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_stored_object(record)

    def add_if_checksum_absent(
        self,
        record: StoredObjectRecord,
    ) -> tuple[StoredObject, bool]:
        """Insert under a savepoint or return the concurrent winner."""

        try:
            with self._session.begin_nested():
                stored_object = self.add(record)
        except IntegrityError:
            existing = self.get_by_checksum(record.checksum_sha256)

            if existing is None:
                raise

            return existing, False

        return stored_object, True
