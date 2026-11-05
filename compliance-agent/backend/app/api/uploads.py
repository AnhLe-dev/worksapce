"""Bounded temporary upload handling for multipart API requests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import UploadFile, status

from app.api.errors import ApiError

_CHUNK_SIZE = 1024 * 1024
_SUPPORTED_SUFFIXES = {".pdf", ".txt"}


@asynccontextmanager
async def temporary_upload(
    upload: UploadFile,
    *,
    max_bytes: int,
) -> AsyncIterator[Path]:
    """Stream an upload to a bounded temporary file and always remove it."""

    filename = Path(upload.filename or "").name
    suffix = Path(filename).suffix.casefold()

    if not filename or suffix not in _SUPPORTED_SUFFIXES:
        raise ApiError(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_document_format",
            "Only .pdf and .txt documents are supported",
        )

    with TemporaryDirectory(prefix="compliance-upload-") as directory:
        path = Path(directory) / f"upload{suffix}"
        byte_count = 0

        try:
            with path.open("wb") as target:
                while block := await upload.read(_CHUNK_SIZE):
                    byte_count += len(block)

                    if byte_count > max_bytes:
                        raise ApiError(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "upload_too_large",
                            "Uploaded document exceeds the configured limit",
                        )

                    target.write(block)

            if byte_count == 0:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "empty_document",
                    "Uploaded document is empty",
                )

            yield path
        finally:
            await upload.close()
