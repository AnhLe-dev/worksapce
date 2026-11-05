"""Application service for validating and deduplicating physical blobs."""

from pathlib import Path
from uuid import uuid4

from rag_core.ingestion.loading import (
    calculate_file_sha256,
    detect_document_format,
    validate_document_path,
)
from rag_core.ingestion.models import DocumentFormat

from app.documents.db_models import StoredObjectRecord
from app.documents.models import (
    StoredObjectIngestionResult,
    StoredObjectStatus,
)
from app.documents.repository import StoredObjectRepository
from app.documents.storage import ObjectStorage

MEDIA_TYPES = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.TXT: "text/plain",
}


class UploadTooLargeError(ValueError):
    """Raised before storage when a file exceeds the configured limit."""


class UploadChecksumMismatchError(ValueError):
    """Raised when a caller-supplied digest does not match the file."""


class StoredObjectIngestionService:
    """Validate one file, reuse its blob, or persist a verified new blob."""

    def __init__(
        self,
        *,
        repository: StoredObjectRepository,
        storage: ObjectStorage,
        max_upload_bytes: int,
        object_prefix: str = "sources",
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("Maximum upload bytes must be positive")

        normalized_prefix = object_prefix.strip().strip("/")

        if not normalized_prefix:
            raise ValueError("Object prefix must not be empty")

        self._repository = repository
        self._storage = storage
        self._max_upload_bytes = max_upload_bytes
        self._object_prefix = normalized_prefix

    def ingest(
        self,
        file_path: Path,
        *,
        expected_checksum: str | None = None,
    ) -> StoredObjectIngestionResult:
        """Persist exactly one verified blob without assigning a role."""

        resolved_path = validate_document_path(file_path)
        byte_size = resolved_path.stat().st_size

        if byte_size > self._max_upload_bytes:
            raise UploadTooLargeError(
                "Uploaded document exceeds the configured byte limit"
            )

        document_format = detect_document_format(resolved_path)
        checksum = calculate_file_sha256(resolved_path)

        if expected_checksum is not None and checksum != expected_checksum:
            raise UploadChecksumMismatchError(
                "Uploaded document checksum does not match"
            )

        existing = self._repository.get_by_checksum(checksum)

        if existing is not None:
            return StoredObjectIngestionResult(
                stored_object=existing,
                reused_existing=True,
            )

        stored_object_id = uuid4()
        object_key = (
            f"{self._object_prefix}/{stored_object_id}/{checksum}"
            f"{resolved_path.suffix.casefold()}"
        )
        media_type = MEDIA_TYPES[document_format]
        self._storage.put_file(
            object_key=object_key,
            file_path=resolved_path,
            media_type=media_type,
            expected_checksum=checksum,
        )
        record = StoredObjectRecord(
            stored_object_id=stored_object_id,
            object_key=object_key,
            checksum_sha256=checksum,
            byte_size=byte_size,
            media_type=media_type,
            document_format=document_format,
            status=StoredObjectStatus.VERIFIED,
        )

        try:
            stored_object, inserted = (
                self._repository.add_if_checksum_absent(record)
            )
        except Exception:
            self._storage.delete_object(object_key)
            raise

        if not inserted:
            self._storage.delete_object(object_key)

        return StoredObjectIngestionResult(
            stored_object=stored_object,
            reused_existing=not inserted,
        )
