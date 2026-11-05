"""Collect trusted tool artifacts without accepting conflicting traces."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from compliance_agent.models import (
    ReferenceUnit,
    RetrievedEvidence,
    ReviewUnit,
)
from compliance_agent.tool_schemas import (
    GetReferenceUnitOutput,
    GetReviewUnitsOutput,
    ReviewToolExecution,
    SearchSelectedKnowledgeOutput,
)


class ReviewArtifactCollectionError(ValueError):
    """Raised when an execution trace contains conflicting artifacts."""


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewUnitEvidence(_ArtifactModel):
    review_unit_id: UUID
    evidence: tuple[RetrievedEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_unique_chunks(self) -> ReviewUnitEvidence:
        chunk_ids = tuple(item.hit.chunk_id for item in self.evidence)

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Unit evidence contains duplicate chunks")

        return self


class CollectedReviewArtifacts(_ArtifactModel):
    review_units: tuple[ReviewUnit, ...] = Field(min_length=1)
    knowledge_searches: tuple[ReviewUnitEvidence, ...] = ()
    reference_units: tuple[ReferenceUnit, ...] = ()


def collect_review_artifacts(
    executions: tuple[ReviewToolExecution, ...],
) -> CollectedReviewArtifacts:
    """Collect exact artifacts and reject duplicate conflicting identities."""

    loaded_units: dict[UUID, ReviewUnit] = {}
    evidence_by_unit: dict[UUID, dict[str, RetrievedEvidence]] = {}
    global_evidence: dict[str, RetrievedEvidence] = {}
    search_order: list[UUID] = []
    references: dict[UUID, ReferenceUnit] = {}
    unit_artifact_count = 0

    for execution in executions:
        artifact = execution.artifact

        if isinstance(artifact, GetReviewUnitsOutput):
            unit_artifact_count += 1

            if unit_artifact_count > 1:
                raise ReviewArtifactCollectionError(
                    "Review units were loaded more than once"
                )

            for unit in artifact.units:
                existing = loaded_units.get(unit.review_unit_id)

                if existing is not None and existing != unit:
                    raise ReviewArtifactCollectionError(
                        "Review unit has conflicting artifact data"
                    )

                loaded_units[unit.review_unit_id] = unit

            continue

        if isinstance(artifact, SearchSelectedKnowledgeOutput):
            if artifact.review_unit_id not in evidence_by_unit:
                evidence_by_unit[artifact.review_unit_id] = {}
                search_order.append(artifact.review_unit_id)

            unit_evidence = evidence_by_unit[artifact.review_unit_id]

            for item in artifact.evidence:
                existing = unit_evidence.get(item.hit.chunk_id)
                global_existing = global_evidence.get(item.hit.chunk_id)

                if (
                    (existing is not None and existing != item)
                    or (
                        global_existing is not None
                        and global_existing != item
                    )
                ):
                    raise ReviewArtifactCollectionError(
                        "Evidence chunk has conflicting artifact data"
                    )

                unit_evidence[item.hit.chunk_id] = item
                global_evidence[item.hit.chunk_id] = item

            continue

        if isinstance(artifact, GetReferenceUnitOutput):
            reference = artifact.reference
            existing = references.get(reference.reference_unit_id)

            if existing is not None and existing != reference:
                raise ReviewArtifactCollectionError(
                    "Reference unit has conflicting artifact data"
                )

            references[reference.reference_unit_id] = reference
            continue

        raise ReviewArtifactCollectionError(
            f"Unsupported artifact type: {type(artifact).__name__}"
        )

    if unit_artifact_count != 1 or not loaded_units:
        raise ReviewArtifactCollectionError(
            "Execution trace does not contain trusted review units"
        )

    return CollectedReviewArtifacts(
        review_units=tuple(loaded_units.values()),
        knowledge_searches=tuple(
            ReviewUnitEvidence(
                review_unit_id=review_unit_id,
                evidence=tuple(evidence_by_unit[review_unit_id].values()),
            )
            for review_unit_id in search_order
        ),
        reference_units=tuple(references.values()),
    )
