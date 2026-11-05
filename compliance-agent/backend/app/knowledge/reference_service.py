"""Return persisted full reference units independently from search chunks."""

from uuid import UUID

from rag_core.ingestion.models import SourceSpan

from app.knowledge.errors import KnowledgeReferenceNotFoundError
from app.knowledge.models import KnowledgeReferenceUnit
from app.knowledge.repository import KnowledgeRepository


class KnowledgeReferenceService:
    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    def get_reference(
        self,
        *,
        reference_unit_id: UUID,
        authorized_snapshot_ids: tuple[UUID, ...],
        trusted_scope: str,
    ) -> KnowledgeReferenceUnit:
        reference = self._repository.get_reference(
            reference_unit_id=reference_unit_id,
            access_scope=trusted_scope,
            authorized_snapshot_ids=authorized_snapshot_ids,
        )

        if reference is None:
            raise KnowledgeReferenceNotFoundError(
                "Reference unit was not found in the trusted scope"
            )

        return reference

    def list_references_for_evidence(
        self,
        *,
        snapshot_id: UUID,
        evidence_spans: tuple[SourceSpan, ...],
        authorized_snapshot_ids: tuple[UUID, ...],
        trusted_scope: str,
    ) -> tuple[KnowledgeReferenceUnit, ...]:
        """Resolve chunk spans to persisted full reference units."""

        if not evidence_spans:
            raise ValueError("Evidence spans must not be empty")

        evidence_block_indexes = {
            span.block_index for span in evidence_spans
        }
        references = self._repository.list_reference_units(
            snapshot_id=snapshot_id,
            access_scope=trusted_scope,
            authorized_snapshot_ids=authorized_snapshot_ids,
        )
        matched = tuple(
            reference
            for reference in references
            if any(
                span.block_index in evidence_block_indexes
                for span in reference.source_spans
            )
        )

        if not matched:
            raise KnowledgeReferenceNotFoundError(
                "Evidence does not map to an authorized reference unit"
            )

        return matched
