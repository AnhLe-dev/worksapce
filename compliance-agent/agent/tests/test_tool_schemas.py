"""Validate model-action and tool-call schemas before execution."""

import pytest
from pydantic import ValidationError

from compliance_agent.tool_schemas import (
    ReviewModelAction,
    ReviewToolCall,
    ReviewToolExecution,
    ReviewToolName,
    ReviewToolObservation,
    SearchSelectedKnowledgeOutput,
)
from tests.factories import REVIEW_UNIT_ID, make_result


def test_model_action_cannot_mix_tools_and_final_result() -> None:
    with pytest.raises(
        ValidationError,
        match="tool calls or one final result",
    ):
        ReviewModelAction(
            tool_calls=(
                ReviewToolCall(
                    call_id="units-1",
                    name=ReviewToolName.GET_REVIEW_UNITS,
                    arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
                ),
            ),
            final_result=make_result(),
        )


def test_model_action_rejects_duplicate_call_ids() -> None:
    call = ReviewToolCall(
        call_id="duplicate",
        name=ReviewToolName.GET_REVIEW_UNITS,
        arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
    )

    with pytest.raises(ValidationError, match="must be unique"):
        ReviewModelAction(tool_calls=(call, call))


def test_tool_execution_rejects_mismatched_artifact_type() -> None:
    with pytest.raises(ValidationError, match="does not match tool name"):
        ReviewToolExecution(
            observation=ReviewToolObservation(
                call_id="units-1",
                name=ReviewToolName.GET_REVIEW_UNITS,
                content="{}",
            ),
            artifact=SearchSelectedKnowledgeOutput(
                review_unit_id=REVIEW_UNIT_ID,
                evidence=(),
            ),
        )
