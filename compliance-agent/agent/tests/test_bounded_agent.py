"""Offline tests for model, tool, identity, and timeout bounds."""

import asyncio
from collections import deque
from uuid import UUID

import pytest

from compliance_agent.agent import (
    AgentBudgetExceededError,
    AgentLimits,
    AgentResultBindingError,
    AgentTimeoutError,
    BoundedReviewAgent,
)
from compliance_agent.models import ReviewOutcome
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


class SequenceModel:
    def __init__(self, actions: tuple[ReviewModelAction, ...]) -> None:
        self.actions = deque(actions)

    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        return self.actions.popleft()


class SlowModel:
    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        await asyncio.sleep(1)
        raise AssertionError("timeout should cancel the model call")


def make_tools() -> ReviewToolset:
    return ReviewToolset(
        request=make_request(),
        review_units=FakeReviewUnitProvider(),
        knowledge=FakeKnowledgeRetriever(),
        references=FakeReferenceUnitProvider(),
        retrieval_limit=5,
    )


def successful_actions() -> tuple[ReviewModelAction, ...]:
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
        ReviewModelAction(final_result=make_result()),
    )


def test_agent_executes_bounded_successful_workflow() -> None:
    execution = asyncio.run(
        BoundedReviewAgent(
            model=SequenceModel(successful_actions())
        ).run(
            request=make_request(),
            tools=make_tools(),
        )
    )

    assert execution.result.outcome is ReviewOutcome.REVIEW_COMPLETED
    assert execution.model_call_count == 3
    assert len(execution.tool_executions) == 3


def test_agent_rejects_model_identity_change() -> None:
    actions = (*successful_actions()[:2],)
    actions += (
        ReviewModelAction(
            final_result=make_result(
                knowledge_base_id=UUID(
                    "20000000-0000-0000-0000-000000000099"
                )
            )
        ),
    )

    with pytest.raises(
        AgentResultBindingError,
        match="knowledge-base ID",
    ):
        asyncio.run(
            BoundedReviewAgent(model=SequenceModel(actions)).run(
                request=make_request(),
                tools=make_tools(),
            )
        )


def test_agent_enforces_per_tool_budget() -> None:
    actions = (
        successful_actions()[0],
        ReviewModelAction(
            tool_calls=(
                ReviewToolCall(
                    call_id="units-2",
                    name=ReviewToolName.GET_REVIEW_UNITS,
                    arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
                ),
            )
        ),
    )

    with pytest.raises(
        AgentBudgetExceededError,
        match="get_review_units",
    ):
        asyncio.run(
            BoundedReviewAgent(model=SequenceModel(actions)).run(
                request=make_request(),
                tools=make_tools(),
            )
        )


def test_agent_enforces_total_timeout() -> None:
    with pytest.raises(AgentTimeoutError):
        asyncio.run(
            BoundedReviewAgent(
                model=SlowModel(),
                limits=AgentLimits(timeout_seconds=0.01),
            ).run(
                request=make_request(),
                tools=make_tools(),
            )
        )
