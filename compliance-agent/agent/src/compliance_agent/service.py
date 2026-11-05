"""Preflight, execute, collect artifacts, and validate fail-closed."""

from __future__ import annotations

from compliance_agent.agent import BoundedReviewAgent
from compliance_agent.artifacts import (
    ReviewArtifactCollectionError,
    collect_review_artifacts,
)
from compliance_agent.models import (
    ReviewApplicabilityStatus,
    ReviewCoverage,
    ReviewOutcome,
    ReviewRequest,
    ReviewResult,
    ReviewValidationIssue,
    ReviewValidationIssueCode,
)
from compliance_agent.ports import ReviewApplicabilityProvider
from compliance_agent.tools import ReviewToolset
from compliance_agent.validator import (
    ReviewValidationError,
    validate_review_result,
)


class ReviewService:
    """Coordinate one bounded review without hiding technical errors."""

    def __init__(
        self,
        *,
        applicability: ReviewApplicabilityProvider,
        agent: BoundedReviewAgent,
    ) -> None:
        self._applicability = applicability
        self._agent = agent

    async def review(
        self,
        *,
        request: ReviewRequest,
        tools: ReviewToolset,
    ) -> ReviewResult:
        assessment = self._applicability.evaluate(request=request)
        preflight_outcome = {
            ReviewApplicabilityStatus.NO_KNOWLEDGE_BASE: (
                ReviewOutcome.NO_KNOWLEDGE_BASE
            ),
            ReviewApplicabilityStatus.NOT_APPLICABLE: (
                ReviewOutcome.NO_APPLICABLE_KNOWLEDGE
            ),
            ReviewApplicabilityStatus.UNDETERMINED: (
                ReviewOutcome.APPLICABILITY_UNDETERMINED
            ),
        }.get(assessment.status)

        if preflight_outcome is not None:
            return ReviewResult(
                outcome=preflight_outcome,
                contract_id=request.contract_id,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                summary=assessment.reason,
                coverage=ReviewCoverage(
                    requested_review_unit_ids=request.review_unit_ids,
                    reviewed_review_unit_ids=(),
                ),
            )

        execution = await self._agent.run(request=request, tools=tools)

        try:
            artifacts = collect_review_artifacts(execution.tool_executions)
            return validate_review_result(
                request=request,
                result=execution.result,
                artifacts=artifacts,
            )
        except ReviewArtifactCollectionError as error:
            issues = (
                ReviewValidationIssue(
                    code=ReviewValidationIssueCode.TOOL_ARTIFACT_INVALID,
                    message=str(error),
                ),
            )
        except ReviewValidationError as error:
            issues = error.issues

        return ReviewResult(
            outcome=ReviewOutcome.VALIDATION_FAILED,
            contract_id=request.contract_id,
            knowledge_base_id=request.knowledge_base_id,
            question=request.question,
            summary=(
                "Agent output failed deterministic source validation."
            ),
            coverage=ReviewCoverage(
                requested_review_unit_ids=request.review_unit_ids,
                reviewed_review_unit_ids=(),
            ),
            findings=(),
            validation_issues=issues,
        )
