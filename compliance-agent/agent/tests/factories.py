"""Reusable generic artifacts and fake capability providers."""

from uuid import UUID

from rag_core.ingestion.models import SourceSpan
from rag_core.retrieval.models import KnowledgeHit

from compliance_agent.models import (
    ReferenceUnit,
    RetrievedEvidence,
    ReviewCoverage,
    ReviewFinding,
    ReviewFindingStatus,
    ReviewOutcome,
    ReviewRequest,
    ReviewResult,
    ReviewUnit,
    TrustedReviewClaims,
)

CONTRACT_ID = UUID("10000000-0000-0000-0000-000000000001")
KNOWLEDGE_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")
REVIEW_UNIT_ID = UUID("30000000-0000-0000-0000-000000000001")
SNAPSHOT_ID = UUID("40000000-0000-0000-0000-000000000001")
REFERENCE_UNIT_ID = UUID("50000000-0000-0000-0000-000000000001")
SOURCE_SPAN = SourceSpan(block_index=0, char_start=0, char_end=30)


def make_request() -> ReviewRequest:
    return ReviewRequest(
        contract_id=CONTRACT_ID,
        review_unit_ids=(REVIEW_UNIT_ID,),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        question="Does the contract require a receipt?",
        trusted_claims=TrustedReviewClaims(
            contract_id=CONTRACT_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            access_scope="system-a",
            contract_processing_revision=1,
            authorized_review_unit_ids=(REVIEW_UNIT_ID,),
            authorized_snapshot_ids=(SNAPSHOT_ID,),
        ),
    )


def make_review_unit() -> ReviewUnit:
    return ReviewUnit(
        review_unit_id=REVIEW_UNIT_ID,
        contract_id=CONTRACT_ID,
        processing_revision=1,
        order_index=0,
        text="Refund requests must include a receipt.",
        label=None,
        source_spans=(SOURCE_SPAN,),
    )


def make_evidence() -> RetrievedEvidence:
    return RetrievedEvidence(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        snapshot_id=SNAPSHOT_ID,
        reference_unit_ids=(REFERENCE_UNIT_ID,),
        hit=KnowledgeHit(
            chunk_id="chunk-1",
            source_id="refund-policy",
            source_version="1",
            source_checksum_sha256="a" * 64,
            access_scope="system-a",
            index_profile="fake:dimensions=3:block:v1",
            text="A receipt is required for every refund request.",
            structure_path=("root",),
            source_spans=(SOURCE_SPAN,),
            score=0.9,
        ),
    )


def make_reference() -> ReferenceUnit:
    return ReferenceUnit(
        reference_unit_id=REFERENCE_UNIT_ID,
        snapshot_id=SNAPSHOT_ID,
        source_id="refund-policy",
        source_version="1",
        source_checksum_sha256="a" * 64,
        access_scope="system-a",
        text="A receipt is required for every refund request.",
        structure_path=("root",),
        source_spans=(SOURCE_SPAN,),
    )


def make_result(
    *,
    knowledge_base_id: UUID = KNOWLEDGE_BASE_ID,
) -> ReviewResult:
    return ReviewResult(
        outcome=ReviewOutcome.REVIEW_COMPLETED,
        contract_id=CONTRACT_ID,
        knowledge_base_id=knowledge_base_id,
        question="Does the contract require a receipt?",
        summary="The selected policy supports the contract requirement.",
        coverage=ReviewCoverage(
            requested_review_unit_ids=(REVIEW_UNIT_ID,),
            reviewed_review_unit_ids=(REVIEW_UNIT_ID,),
        ),
        findings=(
            ReviewFinding(
                review_unit_id=REVIEW_UNIT_ID,
                status=ReviewFindingStatus.NO_ISSUE_FOUND,
                contract_quote="must include a receipt",
                reason="The full reference requires a receipt.",
                evidence_chunk_ids=("chunk-1",),
                reference_unit_ids=(REFERENCE_UNIT_ID,),
            ),
        ),
    )


class FakeReviewUnitProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, ...]] = []

    def get_review_units(
        self,
        *,
        review_unit_ids: tuple[UUID, ...],
    ) -> tuple[ReviewUnit, ...]:
        self.calls.append(review_unit_ids)
        return (make_review_unit(),)


class FakeKnowledgeRetriever:
    def __init__(
        self,
        evidence: tuple[RetrievedEvidence, ...] | None = None,
    ) -> None:
        self.evidence = (
            evidence if evidence is not None else (make_evidence(),)
        )
        self.queries: list[str] = []

    def search_selected_knowledge(
        self,
        *,
        review_unit_id: UUID,
        query: str,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        assert review_unit_id == REVIEW_UNIT_ID
        assert limit > 0
        self.queries.append(query)
        return self.evidence


class FakeReferenceUnitProvider:
    def __init__(self) -> None:
        self.calls: list[UUID] = []

    def get_reference_unit(
        self,
        *,
        reference_unit_id: UUID,
    ) -> ReferenceUnit:
        self.calls.append(reference_unit_id)
        return make_reference()
