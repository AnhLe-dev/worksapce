"""Typed model-facing calls and trusted tool artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from compliance_agent.models import (
    ReferenceUnit,
    RetrievedEvidence,
    ReviewResult,
    ReviewUnit,
)


class _ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewToolName(StrEnum):
    GET_REVIEW_UNITS = "get_review_units"
    SEARCH_SELECTED_KNOWLEDGE = "search_selected_knowledge"
    GET_REFERENCE_UNIT = "get_reference_unit"


class GetReviewUnitsInput(_ToolSchema):
    review_unit_ids: tuple[UUID, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> GetReviewUnitsInput:
        if len(set(self.review_unit_ids)) != len(self.review_unit_ids):
            raise ValueError("Review-unit IDs must be unique")
        return self


class SearchSelectedKnowledgeInput(_ToolSchema):
    review_unit_id: UUID


class GetReferenceUnitInput(_ToolSchema):
    reference_unit_id: UUID


class GetReviewUnitsOutput(_ToolSchema):
    units: tuple[ReviewUnit, ...] = Field(min_length=1)


class SearchSelectedKnowledgeOutput(_ToolSchema):
    review_unit_id: UUID
    evidence: tuple[RetrievedEvidence, ...] = ()


class GetReferenceUnitOutput(_ToolSchema):
    reference: ReferenceUnit


ReviewToolArtifact: TypeAlias = (
    GetReviewUnitsOutput
    | SearchSelectedKnowledgeOutput
    | GetReferenceUnitOutput
)


class ReviewToolCall(_ToolSchema):
    call_id: str = Field(min_length=1, max_length=128)
    name: ReviewToolName
    arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("call_id", mode="before")
    @classmethod
    def normalize_call_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ReviewToolObservation(_ToolSchema):
    call_id: str = Field(min_length=1, max_length=128)
    name: ReviewToolName
    content: str = Field(min_length=1)


class ReviewToolExecution(_ToolSchema):
    observation: ReviewToolObservation
    artifact: ReviewToolArtifact

    @model_validator(mode="after")
    def validate_artifact_type(self) -> ReviewToolExecution:
        expected_types = {
            ReviewToolName.GET_REVIEW_UNITS: GetReviewUnitsOutput,
            ReviewToolName.SEARCH_SELECTED_KNOWLEDGE: (
                SearchSelectedKnowledgeOutput
            ),
            ReviewToolName.GET_REFERENCE_UNIT: GetReferenceUnitOutput,
        }

        expected_type = expected_types[self.observation.name]

        if not isinstance(self.artifact, expected_type):
            raise PydanticCustomError(
                "tool_artifact_type_mismatch",
                "Tool artifact type does not match tool name",
            )

        return self


class ReviewModelAction(_ToolSchema):
    """Require one model turn to call tools or return a final result."""

    tool_calls: tuple[ReviewToolCall, ...] = ()
    final_result: ReviewResult | None = None

    @model_validator(mode="after")
    def validate_action(self) -> ReviewModelAction:
        has_calls = bool(self.tool_calls)
        has_result = self.final_result is not None

        if has_calls == has_result:
            raise ValueError(
                "Model action must contain tool calls or one final result"
            )

        call_ids = tuple(call.call_id for call in self.tool_calls)

        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Tool call IDs must be unique within one turn")

        return self
