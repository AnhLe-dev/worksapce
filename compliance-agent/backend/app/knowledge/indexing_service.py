"""Coordinate rag-core indexing with durable knowledge snapshot state."""

from pathlib import Path
from uuid import UUID

from rag_core.indexing.models import IndexingResult
from rag_core.indexing.service import DocumentIndexingService

from app.knowledge.errors import (
    KnowledgeIndexingConflictError,
    KnowledgeIsolationError,
)
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.models import (
    KnowledgeIndexingOutcome,
    SnapshotStatus,
)
from app.knowledge.repository import KnowledgeRepository, to_snapshot


class KnowledgeIndexingService:
    """Index once per exact source/profile and activate memberships."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        ingestion: KnowledgeIngestionService,
        indexer: DocumentIndexingService,
    ) -> None:
        self._repository = repository
        self._ingestion = ingestion
        self._indexer = indexer

    def index(
        self,
        *,
        snapshot_id: UUID,
        trusted_scope: str,
        file_path: Path,
    ) -> KnowledgeIndexingOutcome:
        context = self._repository.get_snapshot_context(
            snapshot_id=snapshot_id,
            access_scope=trusted_scope,
        )

        if context is None:
            raise KnowledgeIsolationError("Knowledge snapshot was not found")

        snapshot_record, document_record, _ = context

        if snapshot_record.status is SnapshotStatus.READY:
            return KnowledgeIndexingOutcome(
                snapshot=to_snapshot(snapshot_record),
                reused_existing=True,
            )

        self._repository.set_indexing(
            snapshot=snapshot_record,
            document=document_record,
        )
        prepared = self._ingestion.prepare_snapshot(
            snapshot_id=snapshot_id,
            trusted_scope=trusted_scope,
            file_path=file_path,
        )
        result = self._indexer.index(prepared.chunks)
        self._validate_result(result, prepared.snapshot.index_profile)

        if (
            result.source_id != prepared.snapshot.source_id
            or result.source_version != prepared.snapshot.source_version
            or result.source_checksum_sha256
            != prepared.snapshot.source_checksum_sha256
            or result.access_scope != prepared.snapshot.access_scope
            or result.chunk_count != len(prepared.chunks)
        ):
            raise KnowledgeIndexingConflictError(
                "Index result does not match the prepared source snapshot"
            )

        self._repository.replace_reference_units(
            snapshot_id=snapshot_id,
            units=prepared.reference_units,
        )
        completed = self._repository.complete_indexing(
            snapshot=snapshot_record,
            document=document_record,
            collection_name=result.collection_name,
            chunk_count=result.chunk_count,
            vector_dimensions=result.vector_dimensions,
        )
        return KnowledgeIndexingOutcome(
            snapshot=completed,
            reused_existing=False,
        )

    @staticmethod
    def _validate_result(
        result: IndexingResult,
        index_profile: str,
    ) -> None:
        """Keep the profile check explicit at the app composition boundary."""

        if not index_profile.strip():
            raise KnowledgeIndexingConflictError(
                "Index profile must not be empty"
            )

        if result.stored_count != result.chunk_count:
            raise KnowledgeIndexingConflictError(
                "Vector index did not verify every prepared chunk"
            )
