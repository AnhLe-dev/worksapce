"""Lock the Batch 9 request, result, and authority invariants."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from compliance_agent import (
    ReviewCoverage,
    ReviewFinding,
    ReviewFindingStatus,
    ReviewOutcome,
    ReviewRequest,
    ReviewResult,
    TrustedReviewClaims,
)

CONTRACT_ID = UUID("10000000-0000-0000-0000-000000000001")
OTHER_CONTRACT_ID = UUID("10000000-0000-0000-0000-000000000002")
KNOWLEDGE_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")
REVIEW_UNIT_ID = UUID("30000000-0000-0000-0000-000000000001")
SNAPSHOT_ID = UUID("40000000-0000-0000-0000-000000000001")
REFERENCE_UNIT_ID = UUID("50000000-0000-0000-0000-000000000001")


def make_claims() -> TrustedReviewClaims:
    return TrustedReviewClaims(
        contract_id=CONTRACT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        access_scope=" system-a ",
        contract_processing_revision=1,
        authorized_review_unit_ids=(REVIEW_UNIT_ID,),
        authorized_snapshot_ids=(SNAPSHOT_ID,),
    )


def make_finding() -> ReviewFinding:
    return ReviewFinding(
        review_unit_id=REVIEW_UNIT_ID,
        status=ReviewFindingStatus.NO_ISSUE_FOUND,
        contract_quote="A receipt is required.",
        reason="The selected policy explicitly requires a receipt.",
        evidence_chunk_ids=("chunk-1",),
        reference_unit_ids=(REFERENCE_UNIT_ID,),
    )


def test_domain_outcome_enum_is_exact() -> None:
    assert {outcome.value for outcome in ReviewOutcome} == {
        "NO_KNOWLEDGE_BASE",
        "NO_APPLICABLE_KNOWLEDGE",
        "APPLICABILITY_UNDETERMINED",
        "INSUFFICIENT_EVIDENCE",
        "REVIEW_COMPLETED",
        "VALIDATION_FAILED",
        "NEEDS_HUMAN_REVIEW",
    }
    assert "PROCESSING_FAILED" not in ReviewOutcome.__members__


def test_request_requires_exact_backend_authority_binding() -> None:
    request = ReviewRequest(
        contract_id=CONTRACT_ID,
        review_unit_ids=(REVIEW_UNIT_ID,),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        question="  Check the receipt requirement.  ",
        trusted_claims=make_claims(),
    )

    assert request.question == "Check the receipt requirement."
    assert request.trusted_claims.access_scope == "system-a"

    with pytest.raises(
        ValidationError,
        match="another contract",
    ):
        ReviewRequest(
            contract_id=OTHER_CONTRACT_ID,
            review_unit_ids=(REVIEW_UNIT_ID,),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            trusted_claims=make_claims(),
        )


def test_conclusive_finding_requires_evidence_and_reference() -> None:
    with pytest.raises(
        ValidationError,
        match="requires evidence and references",
    ):
        ReviewFinding(
            review_unit_id=REVIEW_UNIT_ID,
            status=ReviewFindingStatus.NON_COMPLIANT,
            contract_quote="A receipt is required.",
            reason="The contract conflicts with the selected policy.",
        )


def test_completed_result_requires_exact_complete_coverage() -> None:
    result = ReviewResult(
        outcome=ReviewOutcome.REVIEW_COMPLETED,
        contract_id=CONTRACT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        question="Check the receipt requirement.",
        summary="The selected review unit is supported by evidence.",
        coverage=ReviewCoverage(
            requested_review_unit_ids=(REVIEW_UNIT_ID,),
            reviewed_review_unit_ids=(REVIEW_UNIT_ID,),
        ),
        findings=(make_finding(),),
    )

    assert result.coverage.complete is True
    assert result.coverage.missing_review_unit_ids == ()


def test_validation_failure_never_publishes_invalid_findings() -> None:
    with pytest.raises(
        ValidationError,
        match="Invalid findings must not be published",
    ):
        ReviewResult(
            outcome=ReviewOutcome.VALIDATION_FAILED,
            contract_id=CONTRACT_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            summary="The model output failed deterministic validation.",
            coverage=ReviewCoverage(
                requested_review_unit_ids=(REVIEW_UNIT_ID,),
                reviewed_review_unit_ids=(REVIEW_UNIT_ID,),
            ),
            findings=(make_finding(),),
        )
