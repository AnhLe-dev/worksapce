"""Deterministically validate quote, evidence, reference, and lineage."""

from __future__ import annotations

from uuid import UUID

from compliance_agent.artifacts import CollectedReviewArtifacts
from compliance_agent.models import (
    ReferenceUnit,
    RetrievedEvidence,
    ReviewFindingStatus,
    ReviewOutcome,
    ReviewRequest,
    ReviewResult,
    ReviewValidationIssue,
    ReviewValidationIssueCode,
)


class ReviewValidationError(ValueError):
    """Raised when model output is not supported by trusted artifacts."""

    def __init__(self, issues: tuple[ReviewValidationIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(
            f"{issue.code.value}: {issue.message}" for issue in issues
        )
        super().__init__(f"Review validation failed: {summary}")


def _normalize_source_text(value: str) -> str:
    return " ".join(value.split())


def validate_review_result(
    *,
    request: ReviewRequest,
    result: ReviewResult,
    artifacts: CollectedReviewArtifacts,
) -> ReviewResult:
    """Validate trace linkage without claiming semantic entailment."""

    issues: list[ReviewValidationIssue] = []

    if result.contract_id != request.contract_id:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.CONTRACT_MISMATCH,
                message="Result belongs to another contract",
            )
        )

    if result.knowledge_base_id != request.knowledge_base_id:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.KNOWLEDGE_BASE_MISMATCH,
                message="Result belongs to another knowledge base",
            )
        )

    if result.question != request.question:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.QUESTION_MISMATCH,
                message="Result changed the original review question",
            )
        )

    if set(result.coverage.requested_review_unit_ids) != set(
        request.review_unit_ids
    ):
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.REQUESTED_SCOPE_MISMATCH,
                message="Result changed the requested review-unit scope",
            )
        )

    if not result.coverage.complete:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.REQUESTED_SCOPE_MISMATCH,
                message="Agent result did not cover every requested unit",
            )
        )

    units_by_id = {
        unit.review_unit_id: unit for unit in artifacts.review_units
    }
    expected_ids = set(request.review_unit_ids)

    for review_unit_id in expected_ids - units_by_id.keys():
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.MISSING_REVIEW_UNIT,
                message="Requested review unit was not loaded",
                review_unit_id=review_unit_id,
            )
        )

    for review_unit_id in units_by_id.keys() - expected_ids:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.REQUESTED_SCOPE_MISMATCH,
                message="Tool loaded a review unit outside requested scope",
                review_unit_id=review_unit_id,
            )
        )

    claims = request.trusted_claims

    for unit in artifacts.review_units:
        if unit.contract_id != request.contract_id:
            issues.append(
                ReviewValidationIssue(
                    code=(
                        ReviewValidationIssueCode
                        .REVIEW_UNIT_CONTRACT_MISMATCH
                    ),
                    message="Review unit belongs to another contract",
                    review_unit_id=unit.review_unit_id,
                )
            )

        if unit.processing_revision != claims.contract_processing_revision:
            issues.append(
                ReviewValidationIssue(
                    code=(
                        ReviewValidationIssueCode
                        .REVIEW_UNIT_REVISION_MISMATCH
                    ),
                    message="Review unit belongs to another revision",
                    review_unit_id=unit.review_unit_id,
                )
            )

    searches_by_unit = {
        search.review_unit_id: search
        for search in artifacts.knowledge_searches
    }

    for review_unit_id in expected_ids - searches_by_unit.keys():
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.MISSING_KNOWLEDGE_SEARCH,
                message="Knowledge search did not run for review unit",
                review_unit_id=review_unit_id,
            )
        )

    for review_unit_id in searches_by_unit.keys() - expected_ids:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.REQUESTED_SCOPE_MISMATCH,
                message="Knowledge search ran outside requested scope",
                review_unit_id=review_unit_id,
            )
        )

    global_evidence: dict[
        str,
        list[tuple[UUID, RetrievedEvidence]],
    ] = {}

    for search in artifacts.knowledge_searches:
        for item in search.evidence:
            global_evidence.setdefault(item.hit.chunk_id, []).append(
                (search.review_unit_id, item)
            )
    references_by_id = {
        reference.reference_unit_id: reference
        for reference in artifacts.reference_units
    }

    for finding_index, finding in enumerate(result.findings):
        unit = units_by_id.get(finding.review_unit_id)

        if unit is None:
            issues.append(
                ReviewValidationIssue(
                    code=ReviewValidationIssueCode.FINDING_UNIT_NOT_FOUND,
                    message="Finding unit was not loaded",
                    finding_index=finding_index,
                    review_unit_id=finding.review_unit_id,
                )
            )
            continue

        if _normalize_source_text(finding.contract_quote) not in (
            _normalize_source_text(unit.text)
        ):
            issues.append(
                ReviewValidationIssue(
                    code=ReviewValidationIssueCode.CONTRACT_QUOTE_MISMATCH,
                    message="Contract quote is not present in review unit",
                    finding_index=finding_index,
                    review_unit_id=finding.review_unit_id,
                )
            )

        cited_evidence = []

        for chunk_id in finding.evidence_chunk_ids:
            located = global_evidence.get(chunk_id, [])

            if not located:
                issues.append(
                    ReviewValidationIssue(
                        code=ReviewValidationIssueCode.EVIDENCE_NOT_FOUND,
                        message="Cited evidence was not retrieved",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        evidence_chunk_id=chunk_id,
                    )
                )
                continue

            matched = tuple(
                item
                for searched_unit_id, item in located
                if searched_unit_id == finding.review_unit_id
            )

            if not matched:
                issues.append(
                    ReviewValidationIssue(
                        code=(
                            ReviewValidationIssueCode
                            .EVIDENCE_NOT_RETRIEVED_FOR_UNIT
                        ),
                        message="Evidence was retrieved for another unit",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        evidence_chunk_id=chunk_id,
                    )
                )
                continue

            evidence = matched[0]
            cited_evidence.append(evidence)

            if evidence.knowledge_base_id != request.knowledge_base_id:
                issues.append(
                    ReviewValidationIssue(
                        code=(
                            ReviewValidationIssueCode
                            .EVIDENCE_KNOWLEDGE_BASE_MISMATCH
                        ),
                        message="Evidence belongs to another knowledge base",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        evidence_chunk_id=chunk_id,
                    )
                )

            if evidence.snapshot_id not in claims.authorized_snapshot_ids:
                issues.append(
                    ReviewValidationIssue(
                        code=(
                            ReviewValidationIssueCode
                            .EVIDENCE_SNAPSHOT_MISMATCH
                        ),
                        message="Evidence belongs to unauthorized snapshot",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        evidence_chunk_id=chunk_id,
                    )
                )

        allowed_reference_ids = {
            reference_id
            for evidence in cited_evidence
            for reference_id in evidence.reference_unit_ids
        }

        for reference_id in finding.reference_unit_ids:
            reference = references_by_id.get(reference_id)

            if reference is None:
                issues.append(
                    ReviewValidationIssue(
                        code=ReviewValidationIssueCode.REFERENCE_NOT_LOADED,
                        message="Cited reference unit was not loaded",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        reference_unit_id=reference_id,
                    )
                )
                continue

            if reference_id not in allowed_reference_ids:
                issues.append(
                    ReviewValidationIssue(
                        code=(
                            ReviewValidationIssueCode.REFERENCE_NOT_RETRIEVED
                        ),
                        message="Reference was not linked by cited evidence",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        reference_unit_id=reference_id,
                    )
                )
                continue

            related_evidence = tuple(
                evidence
                for evidence in cited_evidence
                if reference_id in evidence.reference_unit_ids
            )

            if not any(
                _reference_matches_evidence(reference, evidence)
                for evidence in related_evidence
            ):
                issues.append(
                    ReviewValidationIssue(
                        code=(
                            ReviewValidationIssueCode
                            .REFERENCE_LINEAGE_MISMATCH
                        ),
                        message="Reference lineage differs from evidence",
                        finding_index=finding_index,
                        review_unit_id=finding.review_unit_id,
                        reference_unit_id=reference_id,
                    )
                )

    _validate_outcome_semantics(result=result, issues=issues)

    if issues:
        raise ReviewValidationError(tuple(issues))

    return result


def _reference_matches_evidence(
    reference: ReferenceUnit,
    evidence: RetrievedEvidence,
) -> bool:
    hit = evidence.hit
    reference_blocks = {span.block_index for span in reference.source_spans}
    evidence_blocks = {span.block_index for span in hit.source_spans}
    return (
        reference.snapshot_id == evidence.snapshot_id
        and reference.source_id == hit.source_id
        and reference.source_version == hit.source_version
        and reference.source_checksum_sha256
        == hit.source_checksum_sha256
        and reference.access_scope == hit.access_scope
        and bool(reference_blocks & evidence_blocks)
    )


def _validate_outcome_semantics(
    *,
    result: ReviewResult,
    issues: list[ReviewValidationIssue],
) -> None:
    statuses = {finding.status for finding in result.findings}
    invalid = False

    if result.outcome is ReviewOutcome.INSUFFICIENT_EVIDENCE:
        invalid = ReviewFindingStatus.INSUFFICIENT_EVIDENCE not in statuses
    elif result.outcome is ReviewOutcome.NEEDS_HUMAN_REVIEW:
        invalid = ReviewFindingStatus.NEEDS_HUMAN_REVIEW not in statuses
    elif result.outcome not in {ReviewOutcome.REVIEW_COMPLETED}:
        invalid = True

    if invalid:
        issues.append(
            ReviewValidationIssue(
                code=ReviewValidationIssueCode.OUTCOME_SEMANTICS_INVALID,
                message="Agent result uses an invalid aggregate outcome",
            )
        )
