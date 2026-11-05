"""Offline tests for applicability preflight and fail-closed service."""

import asyncio
from collections import deque

import pytest

from compliance_agent.agent import BoundedReviewAgent
from compliance_agent.models import (
    ReviewApplicabilityAssessment,
    ReviewApplicabilityStatus,
    ReviewOutcome,
)
from compliance_agent.service import ReviewService
from compliance_agent.tool_schemas import (
    ReviewModelAction,
    ReviewToolCall,
    ReviewToolName,
)
from compliance_agent.tools import ReviewToolset
from tests.factories import (
    REFERENCE_UNIT_ID,
    REVIEW_UNIT_ID,
    FakeKnowledgeRetriever,
    FakeReferenceUnitProvider,
    FakeReviewUnitProvider,
    make_request,
    make_result,
)


class FixedApplicability:
    def __init__(self, status: ReviewApplicabilityStatus) -> None:
        self.status = status

    def evaluate(self, **kwargs) -> ReviewApplicabilityAssessment:
        del kwargs
        return ReviewApplicabilityAssessment(
            status=self.status,
            reason=f"Applicability preflight: {self.status.value}",
        )


class SequenceModel:
    def __init__(self, actions: tuple[ReviewModelAction, ...]) -> None:
        self.actions = deque(actions)
        self.call_count = 0

    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        self.call_count += 1
        return self.actions.popleft()


class CrashingModel:
    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        raise RuntimeError("provider timeout")


def make_tools() -> ReviewToolset:
    return ReviewToolset(
        request=make_request(),
        review_units=FakeReviewUnitProvider(),
        knowledge=FakeKnowledgeRetriever(),
        references=FakeReferenceUnitProvider(),
        retrieval_limit=5,
    )


def make_actions(*, bad_quote: bool = False) -> tuple[ReviewModelAction, ...]:
    result = make_result()

    if bad_quote:
        finding = result.findings[0].model_copy(
            update={"contract_quote": "Fabricated contract quote"}
        )
        result = result.model_copy(update={"findings": (finding,)})

    return (
        ReviewModelAction(
            tool_calls=(
                ReviewToolCall(
                    call_id="units-1",
                    name=ReviewToolName.GET_REVIEW_UNITS,
                    arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
                ),
            )
        ),
        ReviewModelAction(
            tool_calls=(
                ReviewToolCall(
                    call_id="search-1",
                    name=ReviewToolName.SEARCH_SELECTED_KNOWLEDGE,
                    arguments={"review_unit_id": str(REVIEW_UNIT_ID)},
                ),
                ReviewToolCall(
                    call_id="reference-1",
                    name=ReviewToolName.GET_REFERENCE_UNIT,
                    arguments={
                        "reference_unit_id": str(REFERENCE_UNIT_ID)
                    },
                ),
            )
        ),
        ReviewModelAction(final_result=result),
    )


@pytest.mark.parametrize(
    ("status", "expected_outcome"),
    (
        (
            ReviewApplicabilityStatus.NO_KNOWLEDGE_BASE,
            ReviewOutcome.NO_KNOWLEDGE_BASE,
        ),
        (
            ReviewApplicabilityStatus.NOT_APPLICABLE,
            ReviewOutcome.NO_APPLICABLE_KNOWLEDGE,
        ),
        (
            ReviewApplicabilityStatus.UNDETERMINED,
            ReviewOutcome.APPLICABILITY_UNDETERMINED,
        ),
    ),
)
def test_preflight_outcome_skips_model_execution(
    status: ReviewApplicabilityStatus,
    expected_outcome: ReviewOutcome,
) -> None:
    model = SequenceModel(())
    result = asyncio.run(
        ReviewService(
            applicability=FixedApplicability(status),
            agent=BoundedReviewAgent(model=model),
        ).review(
            request=make_request(),
            tools=make_tools(),
        )
    )

    assert result.outcome is expected_outcome
    assert model.call_count == 0


def test_valid_agent_execution_returns_review_completed() -> None:
    result = asyncio.run(
        ReviewService(
            applicability=FixedApplicability(
                ReviewApplicabilityStatus.APPLICABLE
            ),
            agent=BoundedReviewAgent(
                model=SequenceModel(make_actions())
            ),
        ).review(
            request=make_request(),
            tools=make_tools(),
        )
    )

    assert result.outcome is ReviewOutcome.REVIEW_COMPLETED
    assert result.validation_issues == ()


def test_invalid_agent_output_is_replaced_with_validation_failure() -> None:
    result = asyncio.run(
        ReviewService(
            applicability=FixedApplicability(
                ReviewApplicabilityStatus.APPLICABLE
            ),
            agent=BoundedReviewAgent(
                model=SequenceModel(make_actions(bad_quote=True))
            ),
        ).review(
            request=make_request(),
            tools=make_tools(),
        )
    )

    assert result.outcome is ReviewOutcome.VALIDATION_FAILED
    assert result.findings == ()
    assert result.validation_issues


def test_technical_model_failure_is_not_a_domain_outcome() -> None:
    service = ReviewService(
        applicability=FixedApplicability(
            ReviewApplicabilityStatus.APPLICABLE
        ),
        agent=BoundedReviewAgent(model=CrashingModel()),
    )

    with pytest.raises(RuntimeError, match="provider timeout"):
        asyncio.run(
            service.review(
                request=make_request(),
                tools=make_tools(),
            )
        )
