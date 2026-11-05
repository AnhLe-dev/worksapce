"""Offline tests for deterministic artifact and citation validation."""

import pytest

from compliance_agent.artifacts import collect_review_artifacts
from compliance_agent.models import ReferenceUnit
from compliance_agent.tool_schemas import (
    ReviewToolCall,
    ReviewToolExecution,
    ReviewToolName,
)
from compliance_agent.tools import ReviewToolset
from compliance_agent.validator import (
    ReviewValidationError,
    validate_review_result,
)
from tests.factories import (
    REFERENCE_UNIT_ID,
    REVIEW_UNIT_ID,
    FakeKnowledgeRetriever,
    FakeReferenceUnitProvider,
    FakeReviewUnitProvider,
    make_request,
    make_result,
)


def make_executions(
    reference_provider: FakeReferenceUnitProvider | None = None,
) -> tuple[ReviewToolExecution, ...]:
    tools = ReviewToolset(
        request=make_request(),
        review_units=FakeReviewUnitProvider(),
        knowledge=FakeKnowledgeRetriever(),
        references=reference_provider or FakeReferenceUnitProvider(),
        retrieval_limit=5,
    )
    return (
        tools.execute(
            ReviewToolCall(
                call_id="units-1",
                name=ReviewToolName.GET_REVIEW_UNITS,
                arguments={"review_unit_ids": [str(REVIEW_UNIT_ID)]},
            )
        ),
        tools.execute(
            ReviewToolCall(
                call_id="search-1",
                name=ReviewToolName.SEARCH_SELECTED_KNOWLEDGE,
                arguments={"review_unit_id": str(REVIEW_UNIT_ID)},
            )
        ),
        tools.execute(
            ReviewToolCall(
                call_id="reference-1",
                name=ReviewToolName.GET_REFERENCE_UNIT,
                arguments={"reference_unit_id": str(REFERENCE_UNIT_ID)},
            )
        ),
    )


def test_validator_accepts_exact_quote_and_lineage() -> None:
    result = make_result()
    validated = validate_review_result(
        request=make_request(),
        result=result,
        artifacts=collect_review_artifacts(make_executions()),
    )

    assert validated is result


def test_validator_rejects_fabricated_contract_quote() -> None:
    result = make_result()
    bad_finding = result.findings[0].model_copy(
        update={"contract_quote": "This quote is not in the contract."}
    )
    bad_result = result.model_copy(update={"findings": (bad_finding,)})

    with pytest.raises(
        ReviewValidationError,
        match="contract_quote_mismatch",
    ):
        validate_review_result(
            request=make_request(),
            result=bad_result,
            artifacts=collect_review_artifacts(make_executions()),
        )


def test_validator_rejects_reference_not_loaded() -> None:
    executions = make_executions()[:2]

    with pytest.raises(
        ReviewValidationError,
        match="reference_not_loaded",
    ):
        validate_review_result(
            request=make_request(),
            result=make_result(),
            artifacts=collect_review_artifacts(executions),
        )


class WrongLineageReferenceProvider(FakeReferenceUnitProvider):
    def get_reference_unit(self, **kwargs) -> ReferenceUnit:
        reference = super().get_reference_unit(**kwargs)
        return reference.model_copy(update={"source_version": "other"})


def test_validator_rejects_reference_lineage_mismatch() -> None:
    artifacts = collect_review_artifacts(
        make_executions(WrongLineageReferenceProvider())
    )

    with pytest.raises(
        ReviewValidationError,
        match="reference_lineage_mismatch",
    ):
        validate_review_result(
            request=make_request(),
            result=make_result(),
            artifacts=artifacts,
        )
