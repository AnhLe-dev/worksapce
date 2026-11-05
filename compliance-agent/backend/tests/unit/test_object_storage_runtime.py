"""Offline tests for object-storage runtime initialization."""

import pytest

from app.core.config import Settings
from app.documents import storage as storage_module
from app.documents.storage import (
    ObjectStorageUnavailableError,
    ensure_object_storage_bucket,
)


class FakeMinioClient:
    """Model only the provider calls used by bucket initialization."""

    def __init__(self, *, bucket_exists: bool) -> None:
        self.bucket_exists_value = bucket_exists
        self.created_buckets: list[str] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        del bucket_name
        return self.bucket_exists_value

    def make_bucket(self, bucket_name: str) -> None:
        self.created_buckets.append(bucket_name)
        self.bucket_exists_value = True


class UnavailableMinioClient:
    """Raise the same connectivity category as the real SDK boundary."""

    def bucket_exists(self, bucket_name: str) -> bool:
        del bucket_name
        raise OSError("provider unavailable")


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        minio_endpoint="minio:9000",
        minio_access_key="test-access-key",
        minio_secret_key="test-secret-key",
        minio_bucket="test-documents",
    )


def install_client(
    monkeypatch: pytest.MonkeyPatch,
    client: FakeMinioClient | UnavailableMinioClient,
) -> None:
    monkeypatch.setattr(
        storage_module,
        "_create_minio_client",
        lambda settings: client,
    )


def test_initializer_creates_missing_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMinioClient(bucket_exists=False)
    install_client(monkeypatch, client)

    ensure_object_storage_bucket(make_settings())

    assert client.created_buckets == ["test-documents"]


def test_initializer_keeps_existing_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMinioClient(bucket_exists=True)
    install_client(monkeypatch, client)

    ensure_object_storage_bucket(make_settings())

    assert client.created_buckets == []


def test_initializer_translates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, UnavailableMinioClient())

    with pytest.raises(
        ObjectStorageUnavailableError,
        match="could not be initialized",
    ):
        ensure_object_storage_bucket(make_settings())
