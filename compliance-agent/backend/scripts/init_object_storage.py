"""Initialize the private S3-compatible object-storage bucket."""

from app.core.config import get_settings
from app.documents.storage import ensure_object_storage_bucket


def main() -> int:
    """Create the configured bucket idempotently without printing secrets."""

    settings = get_settings()
    ensure_object_storage_bucket(settings)
    print(f"Object-storage bucket {settings.minio_bucket!r} is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
