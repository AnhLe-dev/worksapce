"""Offline tests for capability-bound tool behavior."""

import pytest

from compliance_agent.tool_schemas import (
    ReviewToolCall,
    ReviewToolName,
)
from compliance_agent.tools import (
    ReviewToolSequenceError,
    ReviewToolset,
)
from tests.factories import (
    REFERENCE_UNIT_ID,
    REVIEW_UNIT_ID,
    FakeKnowledgeRetriever,
    FakeReferenceUnitProvider,
    FakeReviewUnitProvider,
    make_request,
)


def make_toolset(
    knowledge: FakeKnowledgeRetriever | None = None,
) -> tuple[ReviewToolset, FakeKnowledgeRetriever]:
    resolved_knowledge = knowledge or FakeKnowledgeRetriever()
    return (
        ReviewToolset(
            request=make_request(),
            review_units=FakeReviewUnitProvider(),
            knowledge=resolved_knowledge,
            references=FakeReferenceUnitProvider(),
            retrieval_limit=5,
        ),
        resolved_knowledge,
    )


def test_tools_follow_units_search_reference_sequence() -> None:
    tools, knowledge = make_toolset()
    units = tools.execute(
        ReviewToolCall(
            call_id="units-1",
            name=ReviewToolName.GET_REVIEW_UNITS,
            arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
        )
    )
    evidence = tools.execute(
        ReviewToolCall(
            call_id="search-1",
            name=ReviewToolName.SEARCH_SELECTED_KNOWLEDGE,
            arguments={"review_unit_id": str(REVIEW_UNIT_ID)},
        )
    )
    reference = tools.execute(
        ReviewToolCall(
            call_id="reference-1",
            name=ReviewToolName.GET_REFERENCE_UNIT,
            arguments={"reference_unit_id": str(REFERENCE_UNIT_ID)},
        )
    )

    assert "Refund requests must include a receipt" in knowledge.queries[0]
    assert "Does the contract require a receipt" in knowledge.queries[0]
    assert "system-a" not in units.observation.content
    assert "system-a" not in evidence.observation.content
    assert "system-a" not in reference.observation.content


def test_reference_id_must_come_from_retrieved_evidence() -> None:
    tools, _ = make_toolset()

    with pytest.raises(
        ReviewToolSequenceError,
        match="originate from retrieved evidence",
    ):
        tools.execute(
            ReviewToolCall(
                call_id="reference-1",
                name=ReviewToolName.GET_REFERENCE_UNIT,
                arguments={"reference_unit_id": str(REFERENCE_UNIT_ID)},
            )
        )


def test_search_requires_review_units_to_be_loaded_first() -> None:
    tools, _ = make_toolset()

    with pytest.raises(
        ReviewToolSequenceError,
        match="must run before knowledge search",
    ):
        tools.execute(
            ReviewToolCall(
                call_id="search-1",
                name=ReviewToolName.SEARCH_SELECTED_KNOWLEDGE,
                arguments={"review_unit_id": str(REVIEW_UNIT_ID)},
            )
        )
