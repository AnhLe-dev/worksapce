"""Bind generic agent ports to server-authorized backend resources."""

from collections.abc import Mapping
from uuid import UUID

from compliance_agent import (
    CapabilityViolationError,
    ReferenceUnit,
    RetrievedEvidence,
    ReviewApplicabilityAssessment,
    ReviewApplicabilityStatus,
    ReviewArtifactNotFoundError,
    ReviewRequest,
    ReviewUnit,
)
from rag_core.retrieval.models import KnowledgeHit, RetrievalScope

from app.contracts.repository import ContractRepository
from app.knowledge.applicability import (
    ApplicabilityStatus,
    KnowledgeApplicabilityService,
)
from app.knowledge.models import (
    KnowledgeRetrievalRequest,
    KnowledgeSourceSnapshot,
    ResolvedKnowledgeBase,
)
from app.knowledge.reference_service import KnowledgeReferenceService
from app.knowledge.retrieval_service import KnowledgeRetrievalService


def _scope_for_hit(hit: KnowledgeHit) -> RetrievalScope:
    return RetrievalScope(
        access_scope=hit.access_scope,
        source_id=hit.source_id,
        source_version=hit.source_version,
        source_checksum_sha256=hit.source_checksum_sha256,
        index_profile=hit.index_profile,
    )


class BoundReviewUnitProvider:
    """Expose only one exact processed contract revision."""

    def __init__(
        self,
        *,
        repository: ContractRepository,
        contract_id: UUID,
        access_scope: str,
        processing_revision: int,
        authorized_review_unit_ids: tuple[UUID, ...],
    ) -> None:
        self._repository = repository
        self._contract_id = contract_id
        self._access_scope = access_scope
        self._processing_revision = processing_revision
        self._authorized_ids = authorized_review_unit_ids

    def get_review_units(
        self,
        *,
        review_unit_ids: tuple[UUID, ...],
    ) -> tuple[ReviewUnit, ...]:
        if review_unit_ids != self._authorized_ids:
            raise CapabilityViolationError(
                "Review-unit request escaped the authorized job payload"
            )

        stored_units = self._repository.list_review_units(
            contract_id=self._contract_id,
            access_scope=self._access_scope,
            processing_revision=self._processing_revision,
        )
        units_by_id = {
            unit.review_unit_id: unit for unit in stored_units
        }

        if not set(review_unit_ids).issubset(units_by_id):
            raise ReviewArtifactNotFoundError(
                "An authorized review unit no longer exists"
            )

        return tuple(
            ReviewUnit(
                review_unit_id=units_by_id[unit_id].review_unit_id,
                contract_id=units_by_id[unit_id].contract_id,
                processing_revision=(
                    units_by_id[unit_id].processing_revision
                ),
                order_index=units_by_id[unit_id].order_index,
                text=units_by_id[unit_id].text,
                label=units_by_id[unit_id].label,
                source_spans=units_by_id[unit_id].source_spans,
            )
            for unit_id in review_unit_ids
        )


class BoundSelectedKnowledgeRetriever:
    """Retrieve only from the exact KB snapshots bound to this job."""

    def __init__(
        self,
        *,
        knowledge_base_id: UUID,
        access_scope: str,
        authorized_review_unit_ids: tuple[UUID, ...],
        resolved: ResolvedKnowledgeBase,
        retrieval: KnowledgeRetrievalService,
        references: KnowledgeReferenceService,
    ) -> None:
        self._knowledge_base_id = knowledge_base_id
        self._access_scope = access_scope
        self._authorized_review_unit_ids = frozenset(
            authorized_review_unit_ids
        )
        self._resolved = resolved
        self._retrieval = retrieval
        self._references = references
        self._snapshot_by_scope = {
            snapshot.retrieval_scope: snapshot
            for snapshot in resolved.snapshots
        }
        self._authorized_snapshot_ids = tuple(
            snapshot.snapshot_id for snapshot in resolved.snapshots
        )

    def search_selected_knowledge(
        self,
        *,
        review_unit_id: UUID,
        query: str,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        if review_unit_id not in self._authorized_review_unit_ids:
            raise CapabilityViolationError(
                "Knowledge search escaped the authorized review units"
            )

        result = self._retrieval.retrieve(
            KnowledgeRetrievalRequest(
                knowledge_base_id=self._knowledge_base_id,
                access_scope=self._access_scope,
                query=query,
                limit=limit,
            )
        )
        current_snapshot_ids = tuple(
            snapshot.snapshot_id for snapshot in result.snapshots
        )

        if current_snapshot_ids != self._authorized_snapshot_ids:
            raise CapabilityViolationError(
                "Knowledge-base membership changed during review"
            )

        evidence: list[RetrievedEvidence] = []

        for hit in result.hits:
            snapshot = self._snapshot_by_scope.get(_scope_for_hit(hit))

            if snapshot is None:
                raise CapabilityViolationError(
                    "Retrieved evidence escaped authorized snapshots"
                )

            reference_units = (
                self._references.list_references_for_evidence(
                    snapshot_id=snapshot.snapshot_id,
                    evidence_spans=hit.source_spans,
                    authorized_snapshot_ids=(
                        self._authorized_snapshot_ids
                    ),
                    trusted_scope=self._access_scope,
                )
            )
            evidence.append(
                RetrievedEvidence(
                    knowledge_base_id=self._knowledge_base_id,
                    snapshot_id=snapshot.snapshot_id,
                    reference_unit_ids=tuple(
                        reference.reference_unit_id
                        for reference in reference_units
                    ),
                    hit=hit,
                )
            )

        return tuple(evidence)


class BoundReferenceUnitProvider:
    """Load complete references only from authorized KB snapshots."""

    def __init__(
        self,
        *,
        service: KnowledgeReferenceService,
        snapshots: tuple[KnowledgeSourceSnapshot, ...],
        access_scope: str,
    ) -> None:
        self._service = service
        self._access_scope = access_scope
        self._snapshots = {
            snapshot.snapshot_id: snapshot for snapshot in snapshots
        }
        self._authorized_snapshot_ids = tuple(self._snapshots)

    def get_reference_unit(
        self,
        *,
        reference_unit_id: UUID,
    ) -> ReferenceUnit:
        reference = self._service.get_reference(
            reference_unit_id=reference_unit_id,
            authorized_snapshot_ids=self._authorized_snapshot_ids,
            trusted_scope=self._access_scope,
        )
        snapshot = self._snapshots.get(reference.snapshot_id)

        if snapshot is None:
            raise CapabilityViolationError(
                "Reference escaped the authorized snapshots"
            )

        return ReferenceUnit(
            reference_unit_id=reference.reference_unit_id,
            snapshot_id=reference.snapshot_id,
            source_id=snapshot.source_id,
            source_version=snapshot.source_version,
            source_checksum_sha256=snapshot.source_checksum_sha256,
            access_scope=snapshot.access_scope,
            text=reference.text,
            structure_path=reference.structure_path,
            source_spans=reference.source_spans,
        )


class BackendApplicabilityProvider:
    """Translate a deterministic backend policy into the agent contract."""

    def __init__(
        self,
        *,
        resolved: ResolvedKnowledgeBase,
        trusted_contract_metadata: Mapping[str, str],
    ) -> None:
        self._resolved = resolved
        self._trusted_contract_metadata = dict(
            trusted_contract_metadata
        )

    def evaluate(
        self,
        *,
        request: ReviewRequest,
    ) -> ReviewApplicabilityAssessment:
        if (
            request.knowledge_base_id
            != self._resolved.knowledge_base.knowledge_base_id
        ):
            raise CapabilityViolationError(
                "Applicability request changed the selected knowledge base"
            )

        decision = KnowledgeApplicabilityService().evaluate(
            resolved=self._resolved,
            trusted_contract_metadata=self._trusted_contract_metadata,
        )
        status = {
            ApplicabilityStatus.APPLICABLE: (
                ReviewApplicabilityStatus.APPLICABLE
            ),
            ApplicabilityStatus.NOT_APPLICABLE: (
                ReviewApplicabilityStatus.NOT_APPLICABLE
            ),
            ApplicabilityStatus.UNDETERMINED: (
                ReviewApplicabilityStatus.UNDETERMINED
            ),
        }[decision.status]
        return ReviewApplicabilityAssessment(
            status=status,
            reason=decision.reason.value,
        )
