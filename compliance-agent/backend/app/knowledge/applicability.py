"""Deterministic applicability policy over backend-verified metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from app.knowledge.models import ResolvedKnowledgeBase

POLICY_TYPE_KEY: Final = "policy.type"
POLICY_REQUIRED_PREFIX: Final = "policy.requires."
EXACT_MATCH_POLICY: Final = "exact_match_v1"


class ApplicabilityStatus(StrEnum):
    """Separate deterministic policy state from review outcomes."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNDETERMINED = "undetermined"


class ApplicabilityReason(StrEnum):
    """Explain why a selected knowledge base received one status."""

    EXPLICIT_POLICY_MATCH = "explicit_policy_match"
    EXPLICIT_POLICY_MISMATCH = "explicit_policy_mismatch"
    POLICY_MISSING = "policy_missing"
    POLICY_INVALID = "policy_invalid"
    TRUSTED_CLAIM_MISSING = "trusted_claim_missing"


@dataclass(frozen=True, slots=True)
class ApplicabilityDecision:
    """Return a traceable result without consulting retrieval scores."""

    status: ApplicabilityStatus
    reason: ApplicabilityReason
    evaluated_claims: tuple[str, ...]
    missing_claims: tuple[str, ...] = ()
    mismatched_claims: tuple[str, ...] = ()


def _normalize_metadata(
    metadata: Mapping[str, str],
) -> Mapping[str, str] | None:
    """Normalize policy text and reject ambiguous normalized keys."""

    normalized: dict[str, str] = {}

    for raw_key, raw_value in metadata.items():
        key = raw_key.strip().casefold()
        value = raw_value.strip().casefold()

        if not key or not value:
            return None

        existing = normalized.get(key)

        if existing is not None and existing != value:
            return None

        normalized[key] = value

    return MappingProxyType(normalized)


class KnowledgeApplicabilityService:
    """Evaluate only an explicitly selected and resolved knowledge base."""

    def evaluate(
        self,
        *,
        resolved: ResolvedKnowledgeBase,
        trusted_contract_metadata: Mapping[str, str],
    ) -> ApplicabilityDecision:
        policy_metadata = _normalize_metadata(
            resolved.knowledge_base.applicability_metadata
        )
        trusted_metadata = _normalize_metadata(trusted_contract_metadata)

        if policy_metadata is None or trusted_metadata is None:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.UNDETERMINED,
                reason=ApplicabilityReason.POLICY_INVALID,
                evaluated_claims=(),
            )

        policy_type = policy_metadata.get(POLICY_TYPE_KEY)

        if policy_type is None:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.UNDETERMINED,
                reason=ApplicabilityReason.POLICY_MISSING,
                evaluated_claims=(),
            )

        if policy_type != EXACT_MATCH_POLICY:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.UNDETERMINED,
                reason=ApplicabilityReason.POLICY_INVALID,
                evaluated_claims=(),
            )

        requirements = {
            key.removeprefix(POLICY_REQUIRED_PREFIX): value
            for key, value in policy_metadata.items()
            if key.startswith(POLICY_REQUIRED_PREFIX)
            and key != POLICY_REQUIRED_PREFIX
        }
        evaluated_claims = tuple(sorted(requirements))

        if not requirements:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.UNDETERMINED,
                reason=ApplicabilityReason.POLICY_INVALID,
                evaluated_claims=(),
            )

        missing_claims = tuple(
            claim
            for claim in evaluated_claims
            if claim not in trusted_metadata
        )

        if missing_claims:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.UNDETERMINED,
                reason=ApplicabilityReason.TRUSTED_CLAIM_MISSING,
                evaluated_claims=evaluated_claims,
                missing_claims=missing_claims,
            )

        mismatched_claims = tuple(
            claim
            for claim in evaluated_claims
            if trusted_metadata[claim] != requirements[claim]
        )

        if mismatched_claims:
            return ApplicabilityDecision(
                status=ApplicabilityStatus.NOT_APPLICABLE,
                reason=ApplicabilityReason.EXPLICIT_POLICY_MISMATCH,
                evaluated_claims=evaluated_claims,
                mismatched_claims=mismatched_claims,
            )

        return ApplicabilityDecision(
            status=ApplicabilityStatus.APPLICABLE,
            reason=ApplicabilityReason.EXPLICIT_POLICY_MATCH,
            evaluated_claims=evaluated_claims,
        )
