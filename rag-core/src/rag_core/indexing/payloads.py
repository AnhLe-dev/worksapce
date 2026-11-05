"""Versioned payload encoding shared by vector-store adapters."""

from enum import Enum
from typing import Any

from rag_core.indexing.errors import IndexingValidationError
from rag_core.indexing.ports import IndexableChunk

INDEX_PAYLOAD_SCHEMA_VERSION = 2
_FORBIDDEN_METADATA_KEYS = frozenset({"knowledge_base_id"})


def _enum_value(value: object) -> object:
    """Serialize enums without coupling to a concrete document model."""

    if isinstance(value, Enum):
        return value.value

    return value


def build_index_payload(
    chunk: IndexableChunk,
) -> dict[str, Any]:
    """Encode one generic chunk without copying KB membership."""

    metadata = dict(getattr(chunk, "metadata", {}))
    forbidden_keys = _FORBIDDEN_METADATA_KEYS.intersection(metadata)

    if forbidden_keys:
        joined_keys = ", ".join(sorted(forbidden_keys))
        raise IndexingValidationError(
            "Vector payload metadata contains backend-owned fields: "
            f"{joined_keys}"
        )

    payload: dict[str, Any] = {
        "schema_version": INDEX_PAYLOAD_SCHEMA_VERSION,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source_version": chunk.source_version,
        "source_checksum_sha256": (
            chunk.source_checksum_sha256
        ),
        "access_scope": chunk.access_scope,
        "text": chunk.text,
        "structure_path": list(chunk.structure_path),
        "source_spans": [
            span.model_dump(mode="json")
            for span in chunk.source_spans
        ],
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }

    optional_fields = (
        "parent_chunk_id",
        "document_format",
        "role",
    )

    for field_name in optional_fields:
        if not hasattr(chunk, field_name):
            continue

        value = getattr(chunk, field_name)
        payload[field_name] = _enum_value(value)

    payload["metadata"] = metadata

    return payload