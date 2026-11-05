"""Register knowledge sources and prepare exact indexable snapshots."""

from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from rag_core.ingestion.models import DocumentInput, DocumentRole
from rag_core.ingestion.service import DocumentProcessor

from app.documents.models import StoredObject, StoredObjectStatus
from app.knowledge.db_models import (
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
    KnowledgeSourceSnapshotRecord,
)
from app.knowledge.errors import (
    KnowledgeBaseNotFoundError,
    KnowledgeDocumentConflictError,
    KnowledgeIsolationError,
)
from app.knowledge.models import (
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeDocumentStatus,
    KnowledgeReferenceUnit,
    KnowledgeRegistration,
    MembershipStatus,
    PreparedKnowledgeSnapshot,
    SnapshotStatus,
)
from app.knowledge.repository import (
    KnowledgeRepository,
    to_document,
    to_snapshot,
)


class KnowledgeIngestionService:
    """Own KB registration without assigning authority to Qdrant."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        processor: DocumentProcessor | None = None,
    ) -> None:
        self._repository = repository
        self._processor = processor

    def create_base(
        self,
        *,
        name: str,
        trusted_scope: str,
        applicability_metadata: dict[str, str] | None = None,
    ) -> KnowledgeBase:
        normalized_name = name.strip()
        normalized_scope = trusted_scope.strip()

        if not normalized_name or not normalized_scope:
            raise ValueError("Knowledge base name and scope are required")

        return self._repository.add_base(
            KnowledgeBaseRecord(
                name=normalized_name,
                access_scope=normalized_scope,
                status=KnowledgeBaseStatus.ACTIVE,
                applicability_metadata=applicability_metadata or {},
            )
        )

    def register_document(
        self,
        *,
        knowledge_base_id: UUID,
        stored_object: StoredObject,
        source_id: str,
        source_version: str,
        filename: str,
        index_profile: str,
        trusted_scope: str,
    ) -> KnowledgeRegistration:
        normalized_scope = trusted_scope.strip()
        normalized_source = source_id.strip()
        normalized_version = source_version.strip()
        normalized_filename = filename.strip()
        normalized_profile = index_profile.strip()

        if not all(
            (
                normalized_scope,
                normalized_source,
                normalized_version,
                normalized_filename,
                normalized_profile,
            )
        ):
            raise ValueError("Knowledge registration fields are required")

        if stored_object.status is not StoredObjectStatus.VERIFIED:
            raise ValueError("Knowledge requires a verified stored object")

        base_record = self._repository.get_base_record(
            knowledge_base_id=knowledge_base_id,
            access_scope=normalized_scope,
            for_update=True,
        )

        if (
            base_record is None
            or base_record.status is not KnowledgeBaseStatus.ACTIVE
        ):
            raise KnowledgeBaseNotFoundError(
                "Active knowledge base was not found in the trusted scope"
            )

        document_record = self._repository.get_document_record_by_source(
            source_id=normalized_source,
            source_version=normalized_version,
            access_scope=normalized_scope,
        )

        if document_record is None:
            document = self._repository.add_document(
                KnowledgeDocumentRecord(
                    role="knowledge",
                    stored_object_id=stored_object.stored_object_id,
                    source_id=normalized_source,
                    source_version=normalized_version,
                    filename=normalized_filename,
                    access_scope=normalized_scope,
                    status=KnowledgeDocumentStatus.UPLOADED,
                )
            )
            document_record = self._repository.get_document_record_by_source(
                source_id=normalized_source,
                source_version=normalized_version,
                access_scope=normalized_scope,
            )
            assert document_record is not None
        else:
            if (
                document_record.stored_object_id
                != stored_object.stored_object_id
            ):
                raise KnowledgeDocumentConflictError(
                    "Source version already points to different bytes"
                )
            document = to_document(document_record)

        snapshot_record = self._repository.get_snapshot_record_by_identity(
            source_id=normalized_source,
            source_version=normalized_version,
            index_profile=normalized_profile,
            access_scope=normalized_scope,
        )
        reused_snapshot = snapshot_record is not None

        if snapshot_record is None:
            snapshot = self._repository.add_snapshot(
                KnowledgeSourceSnapshotRecord(
                    knowledge_document_id=document.knowledge_document_id,
                    access_scope=normalized_scope,
                    source_id=normalized_source,
                    source_version=normalized_version,
                    source_checksum_sha256=(
                        stored_object.checksum_sha256
                    ),
                    index_profile=normalized_profile,
                    status=SnapshotStatus.PENDING,
                )
            )
            snapshot_record = self._repository.get_snapshot_record(
                snapshot_id=snapshot.snapshot_id,
                access_scope=normalized_scope,
            )
            assert snapshot_record is not None
        else:
            if (
                snapshot_record.knowledge_document_id
                != document.knowledge_document_id
                or snapshot_record.source_checksum_sha256
                != stored_object.checksum_sha256
            ):
                raise KnowledgeDocumentConflictError(
                    "Snapshot identity conflicts with registered source"
                )
            snapshot = to_snapshot(snapshot_record)

        membership_status = (
            MembershipStatus.READY
            if snapshot.status is SnapshotStatus.READY
            else MembershipStatus.PENDING
        )
        membership, _ = self._repository.add_membership(
            knowledge_base_id=base_record.knowledge_base_id,
            snapshot_id=snapshot.snapshot_id,
            access_scope=normalized_scope,
            status=membership_status,
        )
        return KnowledgeRegistration(
            document=document,
            snapshot=snapshot,
            membership=membership,
            reused_snapshot=reused_snapshot,
        )

    def prepare_snapshot(
        self,
        *,
        snapshot_id: UUID,
        trusted_scope: str,
        file_path: Path,
    ) -> PreparedKnowledgeSnapshot:
        """Run rag-core ingestion and retain full reference blocks."""

        if self._processor is None:
            raise RuntimeError("Document processor is not configured")

        context = self._repository.get_snapshot_context(
            snapshot_id=snapshot_id,
            access_scope=trusted_scope,
        )

        if context is None:
            raise KnowledgeIsolationError("Knowledge snapshot was not found")

        snapshot_record, document_record, stored_object = context
        processed = self._processor.process(
            DocumentInput(
                path=file_path,
                source_id=snapshot_record.source_id,
                source_version=snapshot_record.source_version,
                access_scope=trusted_scope,
                role=DocumentRole.KNOWLEDGE,
            )
        )

        if (
            processed.source.checksum_sha256
            != stored_object.checksum_sha256
            or processed.source.checksum_sha256
            != snapshot_record.source_checksum_sha256
            or document_record.access_scope != trusted_scope
        ):
            raise KnowledgeIsolationError(
                "Processed knowledge does not match the trusted snapshot"
            )

        references: list[KnowledgeReferenceUnit] = []

        for section in processed.structured.sections:
            for block in section.blocks:
                span = block.source_span
                references.append(
                    KnowledgeReferenceUnit(
                        reference_unit_id=uuid5(
                            NAMESPACE_URL,
                            f"{snapshot_id}\x1f{block.order_index}\x1f"
                            f"{snapshot_record.source_checksum_sha256}",
                        ),
                        snapshot_id=snapshot_id,
                        order_index=len(references),
                        text=block.text,
                        structure_path=section.structure_path,
                        source_spans=(span,),
                        page_start=span.page_number,
                        page_end=span.page_number,
                    )
                )

        if not references:
            raise ValueError("Knowledge processing produced no references")

        return PreparedKnowledgeSnapshot(
            snapshot=to_snapshot(snapshot_record),
            chunks=processed.chunks,
            reference_units=tuple(references),
        )
