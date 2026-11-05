"""Capability-bound tools with fail-closed artifact validation."""

from __future__ import annotations

import json
from uuid import UUID

from pydantic import ValidationError

from compliance_agent.models import (
    RetrievedEvidence,
    ReviewRequest,
    ReviewUnit,
)
from compliance_agent.ports import (
    ReferenceUnitProvider,
    ReviewUnitProvider,
    SelectedKnowledgeRetriever,
)
from compliance_agent.tool_schemas import (
    GetReferenceUnitInput,
    GetReferenceUnitOutput,
    GetReviewUnitsInput,
    GetReviewUnitsOutput,
    ReviewToolCall,
    ReviewToolExecution,
    ReviewToolName,
    ReviewToolObservation,
    SearchSelectedKnowledgeInput,
    SearchSelectedKnowledgeOutput,
)


class ReviewToolError(RuntimeError):
    """Base class for rejected model tool calls or artifacts."""


class ReviewToolInputError(ReviewToolError):
    """Raised when model-provided arguments do not match a tool schema."""


class ReviewToolSequenceError(ReviewToolError):
    """Raised when tools are called outside the required workflow."""


class ReviewToolArtifactError(ReviewToolError):
    """Raised when a provider returns data outside trusted claims."""


def _json_content(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class ReviewToolset:
    """Expose only capabilities already bound by the backend."""

    def __init__(
        self,
        *,
        request: ReviewRequest,
        review_units: ReviewUnitProvider,
        knowledge: SelectedKnowledgeRetriever,
        references: ReferenceUnitProvider,
        retrieval_limit: int,
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError("Retrieval limit must be positive")

        self._request = request
        self._review_units = review_units
        self._knowledge = knowledge
        self._references = references
        self._retrieval_limit = retrieval_limit
        self._loaded_units: dict[UUID, ReviewUnit] = {}
        self._retrieved_evidence: dict[str, RetrievedEvidence] = {}
        self._allowed_reference_ids: set[UUID] = set()

    def execute(self, call: ReviewToolCall) -> ReviewToolExecution:
        try:
            if call.name is ReviewToolName.GET_REVIEW_UNITS:
                artifact = self._get_review_units(
                    GetReviewUnitsInput.model_validate(call.arguments)
                )
                content = self._render_review_units(artifact)
            elif call.name is ReviewToolName.SEARCH_SELECTED_KNOWLEDGE:
                artifact = self._search_selected_knowledge(
                    SearchSelectedKnowledgeInput.model_validate(
                        call.arguments
                    )
                )
                content = self._render_evidence(artifact)
            else:
                artifact = self._get_reference_unit(
                    GetReferenceUnitInput.model_validate(call.arguments)
                )
                content = self._render_reference(artifact)
        except ValidationError as error:
            raise ReviewToolInputError(
                f"Invalid arguments for {call.name.value}"
            ) from error

        return ReviewToolExecution(
            observation=ReviewToolObservation(
                call_id=call.call_id,
                name=call.name,
                content=content,
            ),
            artifact=artifact,
        )

    def _get_review_units(
        self,
        tool_input: GetReviewUnitsInput,
    ) -> GetReviewUnitsOutput:
        expected_ids = self._request.review_unit_ids

        if set(tool_input.review_unit_ids) != set(expected_ids):
            raise ReviewToolInputError(
                "Review-unit tool call must match the authorized request"
            )

        units = self._review_units.get_review_units(
            review_unit_ids=expected_ids
        )
        units_by_id = {unit.review_unit_id: unit for unit in units}

        if len(units_by_id) != len(units) or set(units_by_id) != set(
            expected_ids
        ):
            raise ReviewToolArtifactError(
                "Review-unit provider returned incomplete or duplicate units"
            )

        claims = self._request.trusted_claims

        for unit in units:
            if (
                unit.contract_id != self._request.contract_id
                or unit.processing_revision
                != claims.contract_processing_revision
            ):
                raise ReviewToolArtifactError(
                    "Review unit escaped the trusted contract revision"
                )

        ordered = tuple(units_by_id[unit_id] for unit_id in expected_ids)
        self._loaded_units = {unit.review_unit_id: unit for unit in ordered}
        return GetReviewUnitsOutput(units=ordered)

    def _search_selected_knowledge(
        self,
        tool_input: SearchSelectedKnowledgeInput,
    ) -> SearchSelectedKnowledgeOutput:
        unit = self._loaded_units.get(tool_input.review_unit_id)

        if unit is None:
            raise ReviewToolSequenceError(
                "get_review_units must run before knowledge search"
            )

        query_parts = tuple(
            part
            for part in (self._request.question, unit.text)
            if part is not None
        )
        query = "\n\n".join(query_parts)
        evidence = self._knowledge.search_selected_knowledge(
            review_unit_id=unit.review_unit_id,
            query=query,
            limit=self._retrieval_limit,
        )
        seen_chunk_ids: set[str] = set()
        claims = self._request.trusted_claims

        for item in evidence:
            if item.knowledge_base_id != self._request.knowledge_base_id:
                raise ReviewToolArtifactError(
                    "Evidence belongs to another knowledge base"
                )

            if item.snapshot_id not in claims.authorized_snapshot_ids:
                raise ReviewToolArtifactError(
                    "Evidence belongs to an unauthorized snapshot"
                )

            if item.hit.access_scope != claims.access_scope:
                raise ReviewToolArtifactError(
                    "Evidence belongs to another access scope"
                )

            if item.hit.chunk_id in seen_chunk_ids:
                raise ReviewToolArtifactError(
                    "Knowledge provider returned duplicate evidence"
                )

            seen_chunk_ids.add(item.hit.chunk_id)
            self._retrieved_evidence[item.hit.chunk_id] = item
            self._allowed_reference_ids.update(item.reference_unit_ids)

        return SearchSelectedKnowledgeOutput(
            review_unit_id=unit.review_unit_id,
            evidence=evidence,
        )

    def _get_reference_unit(
        self,
        tool_input: GetReferenceUnitInput,
    ) -> GetReferenceUnitOutput:
        if tool_input.reference_unit_id not in self._allowed_reference_ids:
            raise ReviewToolSequenceError(
                "Reference must originate from retrieved evidence"
            )

        reference = self._references.get_reference_unit(
            reference_unit_id=tool_input.reference_unit_id
        )
        claims = self._request.trusted_claims

        if (
            reference.reference_unit_id != tool_input.reference_unit_id
            or reference.snapshot_id not in claims.authorized_snapshot_ids
            or reference.access_scope != claims.access_scope
        ):
            raise ReviewToolArtifactError(
                "Reference unit escaped the trusted knowledge snapshots"
            )

        return GetReferenceUnitOutput(reference=reference)

    @staticmethod
    def _render_review_units(output: GetReviewUnitsOutput) -> str:
        return _json_content(
            {
                "review_units": [
                    {
                        "review_unit_id": str(unit.review_unit_id),
                        "label": unit.label,
                        "text": unit.text,
                    }
                    for unit in output.units
                ]
            }
        )

    @staticmethod
    def _render_evidence(output: SearchSelectedKnowledgeOutput) -> str:
        return _json_content(
            {
                "review_unit_id": str(output.review_unit_id),
                "evidence": [
                    {
                        "chunk_id": item.hit.chunk_id,
                        "source_id": item.hit.source_id,
                        "source_version": item.hit.source_version,
                        "text": item.hit.text,
                        "score": item.hit.score,
                        "reference_unit_ids": [
                            str(reference_id)
                            for reference_id in item.reference_unit_ids
                        ],
                    }
                    for item in output.evidence
                ],
            }
        )

    @staticmethod
    def _render_reference(output: GetReferenceUnitOutput) -> str:
        reference = output.reference
        return _json_content(
            {
                "reference_unit": {
                    "reference_unit_id": str(reference.reference_unit_id),
                    "source_id": reference.source_id,
                    "source_version": reference.source_version,
                    "text": reference.text,
                    "structure_path": list(reference.structure_path),
                }
            }
        )
