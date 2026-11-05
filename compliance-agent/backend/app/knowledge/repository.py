"""Scoped PostgreSQL repository for the knowledge lifecycle."""

from uuid import UUID

from rag_core.ingestion.models import SourceSpan
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.documents.db_models import StoredObjectRecord
from app.knowledge.db_models import (
    KnowledgeBaseMembershipRecord,
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
    KnowledgeReferenceUnitRecord,
    KnowledgeSourceSnapshotRecord,
)
from app.knowledge.models import (
    KnowledgeBase,
    KnowledgeBaseMembership,
    KnowledgeDocument,
    KnowledgeDocumentStatus,
    KnowledgeReferenceUnit,
    KnowledgeSourceSnapshot,
    MembershipStatus,
    SnapshotStatus,
)


def to_document(record: KnowledgeDocumentRecord) -> KnowledgeDocument:
    return KnowledgeDocument(
        knowledge_document_id=record.knowledge_document_id,
        role="knowledge",
        stored_object_id=record.stored_object_id,
        source_id=record.source_id,
        source_version=record.source_version,
        filename=record.filename,
        access_scope=record.access_scope,
        status=record.status,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def to_base(record: KnowledgeBaseRecord) -> KnowledgeBase:
    return KnowledgeBase(
        knowledge_base_id=record.knowledge_base_id,
        name=record.name,
        access_scope=record.access_scope,
        status=record.status,
        applicability_metadata=dict(record.applicability_metadata),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def to_snapshot(
    record: KnowledgeSourceSnapshotRecord,
) -> KnowledgeSourceSnapshot:
    return KnowledgeSourceSnapshot(
        snapshot_id=record.snapshot_id,
        knowledge_document_id=record.knowledge_document_id,
        access_scope=record.access_scope,
        source_id=record.source_id,
        source_version=record.source_version,
        source_checksum_sha256=record.source_checksum_sha256,
        index_profile=record.index_profile,
        status=record.status,
        collection_name=record.collection_name,
        chunk_count=record.chunk_count,
        vector_dimensions=record.vector_dimensions,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def to_membership(
    record: KnowledgeBaseMembershipRecord,
) -> KnowledgeBaseMembership:
    return KnowledgeBaseMembership(
        membership_id=record.membership_id,
        knowledge_base_id=record.knowledge_base_id,
        snapshot_id=record.snapshot_id,
        access_scope=record.access_scope,
        membership_version=record.membership_version,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def to_reference(
    record: KnowledgeReferenceUnitRecord,
) -> KnowledgeReferenceUnit:
    return KnowledgeReferenceUnit(
        reference_unit_id=record.reference_unit_id,
        snapshot_id=record.snapshot_id,
        order_index=record.order_index,
        text=record.text,
        structure_path=tuple(record.structure_path),
        source_spans=tuple(
            SourceSpan.model_validate(span)
            for span in record.source_spans
        ),
        page_start=record.page_start,
        page_end=record.page_end,
    )


class KnowledgeRepository:
    """Persist knowledge state while requiring a trusted scope on reads."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_base(self, record: KnowledgeBaseRecord) -> KnowledgeBase:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_base(record)

    def get_base_record(
        self,
        *,
        knowledge_base_id: UUID,
        access_scope: str,
        for_update: bool = False,
    ) -> KnowledgeBaseRecord | None:
        statement = select(KnowledgeBaseRecord).where(
            KnowledgeBaseRecord.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseRecord.access_scope == access_scope,
        )

        if for_update:
            statement = statement.with_for_update()

        return self._session.scalar(statement)

    def list_bases(self, access_scope: str) -> tuple[KnowledgeBase, ...]:
        records = self._session.scalars(
            select(KnowledgeBaseRecord)
            .where(KnowledgeBaseRecord.access_scope == access_scope)
            .order_by(KnowledgeBaseRecord.name)
        ).all()
        return tuple(to_base(record) for record in records)

    def add_document(
        self,
        record: KnowledgeDocumentRecord,
    ) -> KnowledgeDocument:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_document(record)

    def get_document_record_by_source(
        self,
        *,
        source_id: str,
        source_version: str,
        access_scope: str,
    ) -> KnowledgeDocumentRecord | None:
        return self._session.scalar(
            select(KnowledgeDocumentRecord).where(
                KnowledgeDocumentRecord.source_id == source_id,
                KnowledgeDocumentRecord.source_version == source_version,
                KnowledgeDocumentRecord.access_scope == access_scope,
            )
        )

    def add_snapshot(
        self,
        record: KnowledgeSourceSnapshotRecord,
    ) -> KnowledgeSourceSnapshot:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_snapshot(record)

    def get_snapshot_record(
        self,
        *,
        snapshot_id: UUID,
        access_scope: str,
        for_update: bool = False,
    ) -> KnowledgeSourceSnapshotRecord | None:
        statement = select(KnowledgeSourceSnapshotRecord).where(
            KnowledgeSourceSnapshotRecord.snapshot_id == snapshot_id,
            KnowledgeSourceSnapshotRecord.access_scope == access_scope,
        )

        if for_update:
            statement = statement.with_for_update()

        return self._session.scalar(statement)

    def get_snapshot_record_by_identity(
        self,
        *,
        source_id: str,
        source_version: str,
        index_profile: str,
        access_scope: str,
    ) -> KnowledgeSourceSnapshotRecord | None:
        return self._session.scalar(
            select(KnowledgeSourceSnapshotRecord).where(
                KnowledgeSourceSnapshotRecord.source_id == source_id,
                KnowledgeSourceSnapshotRecord.source_version
                == source_version,
                KnowledgeSourceSnapshotRecord.index_profile == index_profile,
                KnowledgeSourceSnapshotRecord.access_scope == access_scope,
            )
        )

    def get_snapshot_context(
        self,
        *,
        snapshot_id: UUID,
        access_scope: str,
    ) -> tuple[
        KnowledgeSourceSnapshotRecord,
        KnowledgeDocumentRecord,
        StoredObjectRecord,
    ] | None:
        row = self._session.execute(
            select(
                KnowledgeSourceSnapshotRecord,
                KnowledgeDocumentRecord,
                StoredObjectRecord,
            )
            .join(
                KnowledgeDocumentRecord,
                KnowledgeDocumentRecord.knowledge_document_id
                == KnowledgeSourceSnapshotRecord.knowledge_document_id,
            )
            .join(
                StoredObjectRecord,
                StoredObjectRecord.stored_object_id
                == KnowledgeDocumentRecord.stored_object_id,
            )
            .where(
                KnowledgeSourceSnapshotRecord.snapshot_id == snapshot_id,
                KnowledgeSourceSnapshotRecord.access_scope == access_scope,
            )
        ).one_or_none()

        if row is None:
            return None

        return row[0], row[1], row[2]

    def add_membership(
        self,
        *,
        knowledge_base_id: UUID,
        snapshot_id: UUID,
        access_scope: str,
        status: MembershipStatus,
    ) -> tuple[KnowledgeBaseMembership, bool]:
        existing = self._session.scalar(
            select(KnowledgeBaseMembershipRecord)
            .where(
                KnowledgeBaseMembershipRecord.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMembershipRecord.snapshot_id == snapshot_id,
                KnowledgeBaseMembershipRecord.access_scope == access_scope,
                KnowledgeBaseMembershipRecord.status
                != MembershipStatus.REMOVED,
            )
            .order_by(
                KnowledgeBaseMembershipRecord.membership_version.desc()
            )
        )

        if existing is not None:
            return to_membership(existing), True

        latest_version = self._session.scalar(
            select(
                func.max(KnowledgeBaseMembershipRecord.membership_version)
            ).where(
                KnowledgeBaseMembershipRecord.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMembershipRecord.snapshot_id == snapshot_id,
            )
        )
        record = KnowledgeBaseMembershipRecord(
            knowledge_base_id=knowledge_base_id,
            snapshot_id=snapshot_id,
            access_scope=access_scope,
            membership_version=int(latest_version or 0) + 1,
            status=status,
        )
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_membership(record), False

    def list_ready_snapshots(
        self,
        *,
        knowledge_base_id: UUID,
        access_scope: str,
    ) -> tuple[KnowledgeSourceSnapshot, ...]:
        records = self._session.scalars(
            select(KnowledgeSourceSnapshotRecord)
            .join(
                KnowledgeBaseMembershipRecord,
                KnowledgeBaseMembershipRecord.snapshot_id
                == KnowledgeSourceSnapshotRecord.snapshot_id,
            )
            .where(
                KnowledgeBaseMembershipRecord.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMembershipRecord.access_scope == access_scope,
                KnowledgeBaseMembershipRecord.status
                == MembershipStatus.READY,
                KnowledgeSourceSnapshotRecord.status == SnapshotStatus.READY,
            )
            .order_by(
                KnowledgeSourceSnapshotRecord.source_id,
                KnowledgeSourceSnapshotRecord.source_version,
            )
        ).all()
        return tuple(to_snapshot(record) for record in records)

    def has_memberships(
        self,
        *,
        knowledge_base_id: UUID,
        access_scope: str,
    ) -> bool:
        count = self._session.scalar(
            select(func.count())
            .select_from(KnowledgeBaseMembershipRecord)
            .where(
                KnowledgeBaseMembershipRecord.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMembershipRecord.access_scope == access_scope,
                KnowledgeBaseMembershipRecord.status
                != MembershipStatus.REMOVED,
            )
        )
        return bool(count)

    def list_memberships(
        self,
        *,
        knowledge_base_id: UUID,
        access_scope: str,
    ) -> tuple[
        tuple[KnowledgeBaseMembership, KnowledgeSourceSnapshot],
        ...,
    ]:
        rows = self._session.execute(
            select(
                KnowledgeBaseMembershipRecord,
                KnowledgeSourceSnapshotRecord,
            )
            .join(
                KnowledgeSourceSnapshotRecord,
                KnowledgeSourceSnapshotRecord.snapshot_id
                == KnowledgeBaseMembershipRecord.snapshot_id,
            )
            .where(
                KnowledgeBaseMembershipRecord.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMembershipRecord.access_scope == access_scope,
                KnowledgeBaseMembershipRecord.status
                != MembershipStatus.REMOVED,
            )
            .order_by(
                KnowledgeSourceSnapshotRecord.source_id,
                KnowledgeSourceSnapshotRecord.source_version,
            )
        ).all()
        return tuple(
            (to_membership(row[0]), to_snapshot(row[1]))
            for row in rows
        )

    def set_indexing(
        self,
        *,
        snapshot: KnowledgeSourceSnapshotRecord,
        document: KnowledgeDocumentRecord,
    ) -> None:
        snapshot.status = SnapshotStatus.INDEXING
        snapshot.error_code = None
        snapshot.error_message = None
        document.status = KnowledgeDocumentStatus.INDEXING
        document.error_code = None
        document.error_message = None
        self._session.flush()

    def complete_indexing(
        self,
        *,
        snapshot: KnowledgeSourceSnapshotRecord,
        document: KnowledgeDocumentRecord,
        collection_name: str,
        chunk_count: int,
        vector_dimensions: int,
    ) -> KnowledgeSourceSnapshot:
        snapshot.status = SnapshotStatus.READY
        snapshot.collection_name = collection_name
        snapshot.chunk_count = chunk_count
        snapshot.vector_dimensions = vector_dimensions
        snapshot.error_code = None
        snapshot.error_message = None
        document.status = KnowledgeDocumentStatus.READY
        document.error_code = None
        document.error_message = None
        memberships = self._session.scalars(
            select(KnowledgeBaseMembershipRecord).where(
                KnowledgeBaseMembershipRecord.snapshot_id
                == snapshot.snapshot_id,
                KnowledgeBaseMembershipRecord.status
                == MembershipStatus.PENDING,
            )
        ).all()

        for membership in memberships:
            membership.status = MembershipStatus.READY

        self._session.flush()
        return to_snapshot(snapshot)

    def fail_indexing(
        self,
        *,
        snapshot_id: UUID,
        access_scope: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Fail a snapshot and its document without activating membership."""

        context = self.get_snapshot_context(
            snapshot_id=snapshot_id,
            access_scope=access_scope,
        )

        if context is None:
            raise LookupError("Knowledge snapshot was not found")

        snapshot, document, _ = context
        snapshot.status = SnapshotStatus.FAILED
        snapshot.error_code = error_code
        snapshot.error_message = error_message
        document.status = KnowledgeDocumentStatus.FAILED
        document.error_code = error_code
        document.error_message = error_message
        self._session.flush()

    def replace_reference_units(
        self,
        *,
        snapshot_id: UUID,
        units: tuple[KnowledgeReferenceUnit, ...],
    ) -> None:
        self._session.execute(
            delete(KnowledgeReferenceUnitRecord).where(
                KnowledgeReferenceUnitRecord.snapshot_id == snapshot_id
            )
        )

        for unit in units:
            self._session.add(
                KnowledgeReferenceUnitRecord(
                    reference_unit_id=unit.reference_unit_id,
                    snapshot_id=unit.snapshot_id,
                    order_index=unit.order_index,
                    text=unit.text,
                    structure_path=list(unit.structure_path),
                    source_spans=[
                        span.model_dump(mode="json")
                        for span in unit.source_spans
                    ],
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                )
            )

        self._session.flush()

    def get_reference(
        self,
        *,
        reference_unit_id: UUID,
        access_scope: str,
        authorized_snapshot_ids: tuple[UUID, ...],
    ) -> KnowledgeReferenceUnit | None:
        if not authorized_snapshot_ids:
            return None

        record = self._session.scalar(
            select(KnowledgeReferenceUnitRecord)
            .join(
                KnowledgeSourceSnapshotRecord,
                KnowledgeSourceSnapshotRecord.snapshot_id
                == KnowledgeReferenceUnitRecord.snapshot_id,
            )
            .where(
                KnowledgeReferenceUnitRecord.reference_unit_id
                == reference_unit_id,
                KnowledgeSourceSnapshotRecord.access_scope == access_scope,
                KnowledgeSourceSnapshotRecord.status == SnapshotStatus.READY,
                KnowledgeReferenceUnitRecord.snapshot_id.in_(
                    authorized_snapshot_ids
                ),
            )
        )
        return None if record is None else to_reference(record)

    def list_reference_units(
        self,
        *,
        snapshot_id: UUID,
        access_scope: str,
        authorized_snapshot_ids: tuple[UUID, ...],
    ) -> tuple[KnowledgeReferenceUnit, ...]:
        if snapshot_id not in authorized_snapshot_ids:
            return ()

        records = self._session.scalars(
            select(KnowledgeReferenceUnitRecord)
            .join(
                KnowledgeSourceSnapshotRecord,
                KnowledgeSourceSnapshotRecord.snapshot_id
                == KnowledgeReferenceUnitRecord.snapshot_id,
            )
            .where(
                KnowledgeReferenceUnitRecord.snapshot_id == snapshot_id,
                KnowledgeSourceSnapshotRecord.access_scope == access_scope,
                KnowledgeSourceSnapshotRecord.status == SnapshotStatus.READY,
            )
            .order_by(KnowledgeReferenceUnitRecord.order_index)
        ).all()
        return tuple(to_reference(record) for record in records)
