"""Tests for role-free binary storage and deduplication."""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.documents.db_models import StoredObjectRecord
from app.documents.ingestion_service import (
    StoredObjectIngestionService,
    UploadChecksumMismatchError,
)
from app.documents.repository import StoredObjectRepository
from tests.fakes import FakeObjectStorage


def test_stored_object_schema_has_no_role_or_scope() -> None:
    columns = StoredObjectRecord.__table__.columns

    assert "role" not in columns
    assert "access_scope" not in columns


def test_same_bytes_reuse_one_verified_blob(
    session: Session,
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("A generic refund policy.", encoding="utf-8")
    storage = FakeObjectStorage()
    service = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=storage,
        max_upload_bytes=1024,
    )

    first = service.ingest(path)
    second = service.ingest(path)

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert second.stored_object.stored_object_id == (
        first.stored_object.stored_object_id
    )
    assert len(storage.put_calls) == 1


def test_checksum_mismatch_has_no_storage_side_effect(
    session: Session,
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Policy text", encoding="utf-8")
    storage = FakeObjectStorage()
    service = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=storage,
        max_upload_bytes=1024,
    )

    with pytest.raises(UploadChecksumMismatchError):
        service.ingest(path, expected_checksum="0" * 64)

    assert storage.put_calls == []
