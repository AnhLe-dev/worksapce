"""Reusable offline adapters for backend tests."""

from hashlib import sha256
from pathlib import Path


class FakeObjectStorage:
    """Record storage side effects without network access."""

    def __init__(self) -> None:
        self.put_calls: list[str] = []
        self.deleted_keys: list[str] = []
        self.objects: dict[str, bytes] = {}

    def verify_bucket(self) -> None:
        return None

    def put_file(
        self,
        *,
        object_key: str,
        file_path: Path,
        media_type: str,
        expected_checksum: str,
    ) -> None:
        assert file_path.is_file()
        assert media_type in {"application/pdf", "text/plain"}
        assert len(expected_checksum) == 64
        content = file_path.read_bytes()
        assert sha256(content).hexdigest() == expected_checksum
        self.objects[object_key] = content
        self.put_calls.append(object_key)

    def download_file(
        self,
        *,
        object_key: str,
        destination_path: Path,
        expected_checksum: str,
    ) -> Path:
        content = self.objects[object_key]
        assert sha256(content).hexdigest() == expected_checksum
        destination_path.write_bytes(content)
        return destination_path

    def delete_object(self, object_key: str) -> None:
        self.objects.pop(object_key, None)
        self.deleted_keys.append(object_key)
