"""Qdrant adapter enforcing immutable source-version snapshots."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag_core.indexing.errors import (
    IndexingValidationError,
    IndexSnapshotConflictError,
    IndexVerificationError,
    SourceVersionConflictError,
)
from rag_core.indexing.models import (
    IndexingResult,
    IndexingStatus,
    IndexSnapshotInspection,
    IndexSnapshotStatus,
)
from rag_core.indexing.payloads import (
    INDEX_PAYLOAD_SCHEMA_VERSION,
    build_index_payload,
)
from rag_core.indexing.ports import IndexableChunk
from rag_core.retrieval.errors import (
    RetrievalValidationError,
    UnsupportedPayloadSchemaError,
)
from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalScope,
)

_SCROLL_PAGE_SIZE = 256
_IDENTITY_FIELDS = (
    "access_scope",
    "source_id",
    "source_version",
    "source_checksum_sha256",
    "index_profile",
)


def _normalize_required(value: str, *, field_name: str) -> str:
    """Normalize one required adapter configuration value."""

    normalized = value.strip()

    if not normalized:
        raise IndexingValidationError(
            f"{field_name} must not be empty"
        )

    return normalized


def ensure_qdrant_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_dimensions: int,
) -> None:
    """Create or validate one cosine collection and identity indexes."""

    normalized_collection = _normalize_required(
        collection_name,
        field_name="Collection name",
    )

    if vector_dimensions <= 0:
        raise IndexingValidationError(
            "Vector dimensions must be greater than zero"
        )

    if not client.collection_exists(
        collection_name=normalized_collection
    ):
        client.create_collection(
            collection_name=normalized_collection,
            vectors_config=VectorParams(
                size=vector_dimensions,
                distance=Distance.COSINE,
            ),
        )

    collection = client.get_collection(normalized_collection)
    vector_config = collection.config.params.vectors

    if not isinstance(vector_config, VectorParams):
        raise IndexVerificationError(
            "Expected an unnamed dense vector configuration"
        )

    if vector_config.size != vector_dimensions:
        raise IndexVerificationError(
            "Qdrant vector size does not match configuration: "
            f"expected={vector_dimensions}, actual={vector_config.size}"
        )

    if vector_config.distance != Distance.COSINE:
        raise IndexVerificationError(
            "Qdrant collection must use cosine distance"
        )

    for field_name in _IDENTITY_FIELDS:
        existing_index = collection.payload_schema.get(field_name)

        if existing_index is None:
            client.create_payload_index(
                collection_name=normalized_collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            continue

        if existing_index.data_type != PayloadSchemaType.KEYWORD:
            raise IndexVerificationError(
                "Unexpected Qdrant payload index type: "
                f"field={field_name}, "
                f"actual={existing_index.data_type.value}"
            )


def _identity_filter(
    chunk: IndexableChunk,
    *,
    index_profile: str,
    include_checksum: bool,
) -> Filter:
    """Build the exact source identity used for index isolation."""

    conditions = [
        FieldCondition(
            key="access_scope",
            match=MatchValue(
                value=chunk.access_scope
            ),
        ),
        FieldCondition(
            key="source_id",
            match=MatchValue(
                value=chunk.source_id
            ),
        ),
        FieldCondition(
            key="source_version",
            match=MatchValue(
                value=chunk.source_version
            ),
        ),
        FieldCondition(
            key="index_profile",
            match=MatchValue(value=index_profile),
        ),
    ]

    if include_checksum:
        conditions.append(
            FieldCondition(
                key="source_checksum_sha256",
                match=MatchValue(
                    value=(
                        chunk.source_checksum_sha256
                    )
                ),
            )
        )

    return Filter(must=conditions)


class QdrantVectorIndex:
    """Store chunks without mixing revisions under one source version."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        vector_dimensions: int,
        index_profile: str,
        batch_size: int = 64,
    ) -> None:
        normalized_collection = _normalize_required(
            collection_name,
            field_name="Collection name",
        )
        normalized_profile = _normalize_required(
            index_profile,
            field_name="Index profile",
        )

        if vector_dimensions <= 0:
            raise IndexingValidationError(
                "Vector dimensions must be greater than zero"
            )

        if batch_size <= 0:
            raise IndexingValidationError(
                "Index batch size must be greater than zero"
            )

        self._client = client
        self._collection_name = normalized_collection
        self._vector_dimensions = vector_dimensions
        self._index_profile = normalized_profile
        self._batch_size = batch_size

    def _load_existing_snapshot(
        self,
        chunk: IndexableChunk,
    ) -> dict[str, str]:
        """Load every stored chunk ID and checksum for one identity."""

        existing: dict[str, str] = {}
        offset = None
        seen_offsets: set[object] = set()

        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=_identity_filter(
                    chunk,
                    index_profile=self._index_profile,
                    include_checksum=False,
                ),
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=[
                    "chunk_id",
                    "source_checksum_sha256",
                ],
                with_vectors=False,
            )

            for record in records:
                payload = record.payload

                if payload is None:
                    raise IndexVerificationError(
                        "Stored vector point has no payload"
                    )

                chunk_id = payload.get("chunk_id")
                checksum = payload.get(
                    "source_checksum_sha256"
                )

                if (
                    not isinstance(chunk_id, str)
                    or not isinstance(checksum, str)
                ):
                    raise IndexVerificationError(
                        "Stored vector point lacks chunk lineage"
                    )

                existing[chunk_id] = checksum

            if next_offset is None:
                break

            if next_offset in seen_offsets:
                raise IndexVerificationError(
                    "Qdrant scroll returned a repeated offset"
                )

            seen_offsets.add(next_offset)
            offset = next_offset

        return existing

    def _build_point(
        self,
        *,
        chunk: IndexableChunk,
        vector: Sequence[float],
    ) -> PointStruct:
        """Encode one deterministic Qdrant point."""

        point_identity = (
            f"{chunk.access_scope}\x1f"
            f"{chunk.source_id}\x1f"
            f"{chunk.source_version}\x1f"
            f"{chunk.source_checksum_sha256}\x1f"
            f"{self._index_profile}\x1f"
            f"{chunk.chunk_id}"
        )

        return PointStruct(
            id=str(
                uuid5(
                    NAMESPACE_URL,
                    point_identity,
                )
            ),
            vector=list(vector),
            payload={
                **build_index_payload(chunk),
                "index_profile": self._index_profile,
            },
        )

    def inspect_snapshot(
        self,
        chunks: tuple[IndexableChunk, ...],
    ) -> IndexSnapshotInspection:
        """Compare stored lineage with an expected immutable snapshot."""

        if not chunks:
            raise IndexingValidationError(
                "Cannot inspect an empty chunk collection"
            )

        identities = {
            (
                chunk.source_id,
                chunk.source_version,
                chunk.source_checksum_sha256,
                chunk.access_scope,
            )
            for chunk in chunks
        }

        if len(identities) != 1:
            raise IndexingValidationError(
                "Snapshot inspection requires one source identity"
            )

        expected_ids = {
            chunk.chunk_id
            for chunk in chunks
        }

        if len(expected_ids) != len(chunks):
            raise IndexingValidationError(
                "Snapshot inspection received duplicate chunk IDs"
            )

        first_chunk = chunks[0]
        existing = self._load_existing_snapshot(
            first_chunk
        )
        existing_ids = set(existing)
        stored_checksums = set(existing.values())
        missing_ids = expected_ids - existing_ids
        unexpected_ids = existing_ids - expected_ids
        expected_checksum = (
            first_chunk.source_checksum_sha256
        )

        if (
            stored_checksums - {expected_checksum}
            or unexpected_ids
        ):
            status = IndexSnapshotStatus.CONFLICT
        elif not existing_ids:
            status = IndexSnapshotStatus.EMPTY
        elif missing_ids:
            status = IndexSnapshotStatus.PARTIAL
        else:
            status = IndexSnapshotStatus.COMPLETE

        return IndexSnapshotInspection(
            collection_name=self._collection_name,
            source_id=first_chunk.source_id,
            source_version=first_chunk.source_version,
            source_checksum_sha256=expected_checksum,
            access_scope=first_chunk.access_scope,
            status=status,
            expected_count=len(expected_ids),
            stored_count=len(existing_ids),
            missing_chunk_ids=tuple(sorted(missing_ids)),
            unexpected_chunk_ids=tuple(
                sorted(unexpected_ids)
            ),
            stored_checksums=tuple(
                sorted(stored_checksums)
            ),
        )

    def upsert(
        self,
        *,
        chunks: tuple[IndexableChunk, ...],
        vectors: tuple[tuple[float, ...], ...],
    ) -> IndexingResult:
        """Upsert a compatible snapshot or reject stale evidence."""

        if not chunks:
            raise IndexingValidationError(
                "Cannot index an empty chunk collection"
            )

        if len(chunks) != len(vectors):
            raise IndexingValidationError(
                "Chunk count does not match vector count"
            )

        if any(
            len(vector) != self._vector_dimensions
            for vector in vectors
        ):
            raise IndexingValidationError(
                "Vector dimensions do not match the index"
            )

        identities = {
            (
                chunk.source_id,
                chunk.source_version,
                chunk.source_checksum_sha256,
                chunk.access_scope,
            )
            for chunk in chunks
        }

        if len(identities) != 1:
            raise IndexingValidationError(
                "Vector index requires one source snapshot"
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise IndexingValidationError(
                "Vector index received duplicate chunk IDs"
            )

        first_chunk = chunks[0]
        expected_checksum = (
            first_chunk.source_checksum_sha256
        )
        existing = self._load_existing_snapshot(
            first_chunk
        )
        existing_checksums = set(existing.values())

        if existing_checksums - {expected_checksum}:
            raise SourceVersionConflictError(
                "Source ID and version already use another checksum"
            )

        requested_ids = {
            chunk.chunk_id
            for chunk in chunks
        }
        existing_ids = set(existing)
        unexpected_ids = existing_ids - requested_ids

        if unexpected_ids:
            raise IndexSnapshotConflictError(
                "Stored snapshot contains chunk IDs absent from "
                "the requested snapshot"
            )

        missing_ids = requested_ids - existing_ids

        if not missing_ids:
            return IndexingResult(
                collection_name=self._collection_name,
                source_id=first_chunk.source_id,
                source_version=first_chunk.source_version,
                source_checksum_sha256=expected_checksum,
                access_scope=first_chunk.access_scope,
                status=IndexingStatus.UNCHANGED,
                chunk_count=len(chunks),
                uploaded_count=0,
                stored_count=len(existing_ids),
                vector_dimensions=self._vector_dimensions,
            )

        points = [
            self._build_point(
                chunk=chunk,
                vector=vector,
            )
            for chunk, vector in zip(
                chunks,
                vectors,
                strict=True,
            )
            if chunk.chunk_id in missing_ids
        ]

        for start in range(0, len(points), self._batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[
                    start:start + self._batch_size
                ],
                wait=True,
            )

        stored_count = self._client.count(
            collection_name=self._collection_name,
            count_filter=_identity_filter(
                first_chunk,
                index_profile=self._index_profile,
                include_checksum=True,
            ),
            exact=True,
        ).count

        if stored_count != len(chunks):
            raise IndexVerificationError(
                "Stored source snapshot count does not match: "
                f"expected={len(chunks)}, actual={stored_count}"
            )

        return IndexingResult(
            collection_name=self._collection_name,
            source_id=first_chunk.source_id,
            source_version=first_chunk.source_version,
            source_checksum_sha256=expected_checksum,
            access_scope=first_chunk.access_scope,
            status=IndexingStatus.INDEXED,
            chunk_count=len(chunks),
            uploaded_count=len(points),
            stored_count=stored_count,
            vector_dimensions=self._vector_dimensions,
        )


def _decode_knowledge_hit(
    *,
    payload: dict[str, object],
    score: float,
) -> KnowledgeHit:
    """Decode one schema-v2 payload without legacy domain fields."""

    schema_version = payload.get("schema_version")

    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != INDEX_PAYLOAD_SCHEMA_VERSION
    ):
        raise UnsupportedPayloadSchemaError(
            "Unsupported vector payload schema: "
            f"{schema_version!r}"
        )

    return KnowledgeHit.model_validate({
        "chunk_id": payload.get("chunk_id"),
        "source_id": payload.get("source_id"),
        "source_version": payload.get("source_version"),
        "source_checksum_sha256": payload.get(
            "source_checksum_sha256"
        ),
        "access_scope": payload.get("access_scope"),
        "index_profile": payload.get("index_profile"),
        "text": payload.get("text"),
        "structure_path": payload.get("structure_path"),
        "source_spans": payload.get("source_spans"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "score": score,
        "document_format": payload.get("document_format"),
        "role": payload.get("role"),
        "metadata": payload.get("metadata", {}),
    })


class QdrantVectorSearch:
    """Search Qdrant with exact immutable-snapshot filters."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        vector_dimensions: int,
    ) -> None:
        self._collection_name = _normalize_required(
            collection_name,
            field_name="Collection name",
        )

        if vector_dimensions <= 0:
            raise RetrievalValidationError(
                "Vector dimensions must be greater than zero"
            )

        self._client = client
        self._vector_dimensions = vector_dimensions

    def search(
        self,
        *,
        query_vector: tuple[float, ...],
        scope: RetrievalScope,
        limit: int,
    ) -> tuple[KnowledgeHit, ...]:
        """Query one authorized source snapshot and decode its hits."""

        if limit <= 0:
            raise RetrievalValidationError(
                "Retrieval limit must be greater than zero"
            )

        if len(query_vector) != self._vector_dimensions:
            raise RetrievalValidationError(
                "Query vector dimensions do not match the index"
            )

        query_response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="access_scope",
                        match=MatchValue(
                            value=scope.access_scope
                        ),
                    ),
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(
                            value=scope.source_id
                        ),
                    ),
                    FieldCondition(
                        key="source_version",
                        match=MatchValue(
                            value=scope.source_version
                        ),
                    ),
                    FieldCondition(
                        key="source_checksum_sha256",
                        match=MatchValue(
                            value=scope.source_checksum_sha256
                        ),
                    ),
                    FieldCondition(
                        key="index_profile",
                        match=MatchValue(
                            value=scope.index_profile
                        ),
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[KnowledgeHit] = []

        for point in query_response.points:
            if point.payload is None:
                raise RetrievalValidationError(
                    f"Qdrant point {point.id} has no payload"
                )

            hits.append(
                _decode_knowledge_hit(
                    payload=point.payload,
                    score=point.score,
                )
            )

        return tuple(hits)
