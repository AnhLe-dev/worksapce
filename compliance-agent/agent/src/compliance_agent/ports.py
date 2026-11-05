"""Capability ports injected by the trusted backend composition root."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from compliance_agent.models import (
    ReferenceUnit,
    RetrievedEvidence,
    ReviewApplicabilityAssessment,
    ReviewRequest,
    ReviewUnit,
)
from compliance_agent.tool_schemas import (
    ReviewModelAction,
    ReviewToolObservation,
)


class AgentPortError(RuntimeError):
    """Base class for expected capability-port failures."""


class CapabilityViolationError(AgentPortError):
    """Raised when a call escapes its backend-authorized capability."""


class ReviewArtifactNotFoundError(AgentPortError):
    """Raised when an authorized review artifact no longer exists."""


@runtime_checkable
class ReviewUnitProvider(Protocol):
    """Load units only from the contract bound by the backend."""

    def get_review_units(
        self,
        *,
        review_unit_ids: tuple[UUID, ...],
    ) -> tuple[ReviewUnit, ...]:
        """Return exact authorized units in requested order."""


@runtime_checkable
class SelectedKnowledgeRetriever(Protocol):
    """Search only the selected, authorized knowledge base."""

    def search_selected_knowledge(
        self,
        *,
        review_unit_id: UUID,
        query: str,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        """Return evidence bound to one review unit and selected KB."""


@runtime_checkable
class ReferenceUnitProvider(Protocol):
    """Read references only from snapshots authorized by the backend."""

    def get_reference_unit(
        self,
        *,
        reference_unit_id: UUID,
    ) -> ReferenceUnit:
        """Return one complete persisted reference unit."""


@runtime_checkable
class ReviewModel(Protocol):
    """Produce one bounded action without owning data authority."""

    async def next_action(
        self,
        *,
        system_prompt: str,
        request: ReviewRequest,
        observations: tuple[ReviewToolObservation, ...],
    ) -> ReviewModelAction:
        """Return tool calls or one structured final result."""


@runtime_checkable
class ReviewApplicabilityProvider(Protocol):
    """Evaluate the selected KB using backend-trusted policy metadata."""

    def evaluate(
        self,
        *,
        request: ReviewRequest,
    ) -> ReviewApplicabilityAssessment:
        """Return one deterministic preflight assessment."""
