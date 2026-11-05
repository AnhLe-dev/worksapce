"""Offline tests for deterministic selected-KB applicability policy."""

from datetime import UTC, datetime
from uuid import UUID

from app.knowledge.applicability import (
    ApplicabilityReason,
    ApplicabilityStatus,
    KnowledgeApplicabilityService,
)
from app.knowledge.models import (
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeSourceSnapshot,
    ResolvedKnowledgeBase,
    SnapshotStatus,
)

KNOWLEDGE_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")
SNAPSHOT_ID = UUID("40000000-0000-0000-0000-000000000001")
DOCUMENT_ID = UUID("60000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 11, 5, tzinfo=UTC)


def make_resolved(
    applicability_metadata: dict[str, str],
) -> ResolvedKnowledgeBase:
    snapshot = KnowledgeSourceSnapshot(
        snapshot_id=SNAPSHOT_ID,
        knowledge_document_id=DOCUMENT_ID,
        access_scope="system-a",
        source_id="refund-policy",
        source_version="1",
        source_checksum_sha256="a" * 64,
        index_profile="fake:dimensions=3:block:v1",
        status=SnapshotStatus.READY,
        collection_name="knowledge_test",
        chunk_count=1,
        vector_dimensions=3,
        created_at=NOW,
        updated_at=NOW,
    )
    return ResolvedKnowledgeBase(
        knowledge_base=KnowledgeBase(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            name="Refund policy",
            access_scope="system-a",
            status=KnowledgeBaseStatus.ACTIVE,
            applicability_metadata=applicability_metadata,
            created_at=NOW,
            updated_at=NOW,
        ),
        snapshots=(snapshot,),
        scopes=(snapshot.retrieval_scope,),
    )


def test_explicit_policy_match_is_applicable() -> None:
    decision = KnowledgeApplicabilityService().evaluate(
        resolved=make_resolved(
            {
                "policy.type": "exact_match_v1",
                "policy.requires.jurisdiction": "Vietnam",
            }
        ),
        trusted_contract_metadata={"jurisdiction": "vietnam"},
    )

    assert decision.status is ApplicabilityStatus.APPLICABLE
    assert decision.reason is ApplicabilityReason.EXPLICIT_POLICY_MATCH
    assert decision.evaluated_claims == ("jurisdiction",)


def test_explicit_policy_mismatch_is_not_applicable() -> None:
    decision = KnowledgeApplicabilityService().evaluate(
        resolved=make_resolved(
            {
                "policy.type": "exact_match_v1",
                "policy.requires.jurisdiction": "Vietnam",
            }
        ),
        trusted_contract_metadata={"jurisdiction": "Singapore"},
    )

    assert decision.status is ApplicabilityStatus.NOT_APPLICABLE
    assert decision.mismatched_claims == ("jurisdiction",)


def test_missing_trusted_claim_is_undetermined() -> None:
    decision = KnowledgeApplicabilityService().evaluate(
        resolved=make_resolved(
            {
                "policy.type": "exact_match_v1",
                "policy.requires.jurisdiction": "Vietnam",
            }
        ),
        trusted_contract_metadata={},
    )

    assert decision.status is ApplicabilityStatus.UNDETERMINED
    assert decision.reason is ApplicabilityReason.TRUSTED_CLAIM_MISSING
    assert decision.missing_claims == ("jurisdiction",)


def test_descriptive_domain_metadata_does_not_prove_applicability() -> None:
    decision = KnowledgeApplicabilityService().evaluate(
        resolved=make_resolved(
            {
                "domain": "refunds",
                "document_type": "policy",
            }
        ),
        trusted_contract_metadata={
            "domain": "refunds",
            "document_type": "policy",
        },
    )

    assert decision.status is ApplicabilityStatus.UNDETERMINED
    assert decision.reason is ApplicabilityReason.POLICY_MISSING
