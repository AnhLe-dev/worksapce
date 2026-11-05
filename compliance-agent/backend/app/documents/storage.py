"""Binary storage port and MinIO adapter."""

from pathlib import Path
from re import fullmatch
from typing import Protocol

from minio import Minio
from minio.error import MinioException
from pydantic import SecretStr
from rag_core.ingestion.loading import calculate_file_sha256
from urllib3 import PoolManager, Retry, Timeout
from urllib3.exceptions import HTTPError

from app.core.config import Settings


class ObjectStorage(Protocol):
    """Define checksum-verifying blob operations used by the backend."""

    def verify_bucket(self) -> None:
        """Raise when the configured bucket is unavailable."""

    def put_file(
        self,
        *,
        object_key: str,
        file_path: Path,
        media_type: str,
        expected_checksum: str,
    ) -> None:
        """Store and verify one immutable local file."""

    def download_file(
        self,
        *,
        object_key: str,
        destination_path: Path,
        expected_checksum: str,
    ) -> Path:
        """Download one object and verify its checksum."""

    def delete_object(self, object_key: str) -> None:
        """Delete a newly written orphan during compensation."""


class ObjectStorageUnavailableError(RuntimeError):
    """Raised when readiness cannot reach the configured object bucket."""


class StoredObjectIntegrityError(RuntimeError):
    """Raised when downloaded bytes do not match durable blob identity."""


def _require_sha256(value: str) -> str:
    if not fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected checksum must be a SHA-256 hex digest")
    return value


def require_secret(value: SecretStr | None, variable_name: str) -> str:
    """Return a configured secret without displaying its value."""

    if value is None or not value.get_secret_value():
        raise ValueError(f"{variable_name} is not configured")
    return value.get_secret_value()


def _create_minio_client(settings: Settings) -> Minio:
    """Create the S3-compatible client at the provider boundary."""

    if settings.minio_endpoint is None:
        raise ValueError("MINIO_ENDPOINT is not configured")

    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=require_secret(
            settings.minio_access_key,
            "MINIO_ACCESS_KEY",
        ),
        secret_key=require_secret(
            settings.minio_secret_key,
            "MINIO_SECRET_KEY",
        ),
        secure=settings.minio_secure,
        http_client=PoolManager(
            timeout=Timeout(
                connect=settings.dependency_timeout_seconds,
                read=settings.dependency_timeout_seconds,
            ),
            retries=Retry(total=0),
        ),
    )


class MinioObjectStorage:
    """Store immutable verified blobs in one private MinIO bucket."""

    def __init__(self, *, client: Minio, bucket_name: str) -> None:
        self._client = client
        self._bucket_name = bucket_name

    def verify_bucket(self) -> None:
        try:
            exists = self._client.bucket_exists(self._bucket_name)
        except (
            ConnectionError,
            HTTPError,
            MinioException,
            OSError,
            TimeoutError,
        ) as error:
            raise ObjectStorageUnavailableError(
                "Configured object bucket is unavailable"
            ) from error

        if not exists:
            raise ObjectStorageUnavailableError(
                "Configured object bucket does not exist"
            )

    def put_file(
        self,
        *,
        object_key: str,
        file_path: Path,
        media_type: str,
        expected_checksum: str,
    ) -> None:
        normalized_key = object_key.strip()
        _require_sha256(expected_checksum)

        if not normalized_key:
            raise ValueError("Object key must not be empty")

        resolved_path = file_path.resolve()
        actual_checksum = calculate_file_sha256(resolved_path)

        if actual_checksum != expected_checksum:
            raise ValueError("File checksum does not match expected checksum")

        self._client.fput_object(
            bucket_name=self._bucket_name,
            object_name=normalized_key,
            file_path=str(resolved_path),
            content_type=media_type,
            metadata={"sha256": actual_checksum},
        )
        stored = self._client.stat_object(
            bucket_name=self._bucket_name,
            object_name=normalized_key,
        )
        stored_checksum = (
            stored.metadata.get("x-amz-meta-sha256")
            or stored.metadata.get("sha256")
        )

        if (
            stored_checksum != actual_checksum
            or stored.size != resolved_path.stat().st_size
        ):
            self.delete_object(normalized_key)
            raise RuntimeError("Stored object verification failed")

    def download_file(
        self,
        *,
        object_key: str,
        destination_path: Path,
        expected_checksum: str,
    ) -> Path:
        normalized_key = object_key.strip()
        _require_sha256(expected_checksum)

        if not normalized_key:
            raise ValueError("Object key must not be empty")

        destination = destination_path.resolve()

        if not destination.parent.is_dir():
            raise ValueError("Destination directory does not exist")

        try:
            self._client.fget_object(
                bucket_name=self._bucket_name,
                object_name=normalized_key,
                file_path=str(destination),
            )
        except (
            ConnectionError,
            HTTPError,
            MinioException,
            OSError,
            TimeoutError,
        ) as error:
            destination.unlink(missing_ok=True)
            raise ObjectStorageUnavailableError(
                "Stored object download is unavailable"
            ) from error

        if calculate_file_sha256(destination) != expected_checksum:
            destination.unlink(missing_ok=True)
            raise StoredObjectIntegrityError(
                "Downloaded object checksum is invalid"
            )

        return destination

    def delete_object(self, object_key: str) -> None:
        normalized_key = object_key.strip()

        if not normalized_key:
            raise ValueError("Object key must not be empty")

        self._client.remove_object(
            bucket_name=self._bucket_name,
            object_name=normalized_key,
        )


def create_object_storage(settings: Settings) -> MinioObjectStorage:
    """Build the S3-compatible adapter only at the composition boundary."""

    return MinioObjectStorage(
        client=_create_minio_client(settings),
        bucket_name=settings.minio_bucket,
    )


def ensure_object_storage_bucket(settings: Settings) -> None:
    """Create the configured bucket once and verify it is reachable."""

    client = _create_minio_client(settings)

    try:
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
    except (
        ConnectionError,
        HTTPError,
        MinioException,
        OSError,
        TimeoutError,
    ) as error:
        raise ObjectStorageUnavailableError(
            "Configured object bucket could not be initialized"
        ) from error

    MinioObjectStorage(
        client=client,
        bucket_name=settings.minio_bucket,
    ).verify_bucket()
