"""Immutable contracts for bounded, provider-neutral domain review."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from rag_core.ingestion.models import SourceSpan
from rag_core.retrieval.models import KnowledgeHit


class _FrozenModel(BaseModel):
    """Reject unknown fields and mutation at every agent boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewOutcome(StrEnum):
    """Represent the exact domain outcomes allowed by Release B."""

    NO_KNOWLEDGE_BASE = "NO_KNOWLEDGE_BASE"
    NO_APPLICABLE_KNOWLEDGE = "NO_APPLICABLE_KNOWLEDGE"
    APPLICABILITY_UNDETERMINED = "APPLICABILITY_UNDETERMINED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class ReviewFindingStatus(StrEnum):
    """Describe one review-unit conclusion, not technical execution."""

    NON_COMPLIANT = "non_compliant"
    NO_ISSUE_FOUND = "no_issue_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class ReviewApplicabilityStatus(StrEnum):
    """Represent deterministic preflight state supplied by the backend."""

    NO_KNOWLEDGE_BASE = "no_knowledge_base"
    NOT_APPLICABLE = "not_applicable"
    UNDETERMINED = "undetermined"
    APPLICABLE = "applicable"


class ReviewValidationIssueCode(StrEnum):
    """Identify one deterministic artifact or lineage failure."""

    TOOL_ARTIFACT_INVALID = "tool_artifact_invalid"
    CONTRACT_MISMATCH = "contract_mismatch"
    KNOWLEDGE_BASE_MISMATCH = "knowledge_base_mismatch"
    QUESTION_MISMATCH = "question_mismatch"
    REQUESTED_SCOPE_MISMATCH = "requested_scope_mismatch"
    MISSING_REVIEW_UNIT = "missing_review_unit"
    REVIEW_UNIT_CONTRACT_MISMATCH = "review_unit_contract_mismatch"
    REVIEW_UNIT_REVISION_MISMATCH = "review_unit_revision_mismatch"
    FINDING_UNIT_NOT_FOUND = "finding_unit_not_found"
    CONTRACT_QUOTE_MISMATCH = "contract_quote_mismatch"
    MISSING_KNOWLEDGE_SEARCH = "missing_knowledge_search"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    EVIDENCE_NOT_RETRIEVED_FOR_UNIT = (
        "evidence_not_retrieved_for_unit"
    )
    EVIDENCE_KNOWLEDGE_BASE_MISMATCH = (
        "evidence_knowledge_base_mismatch"
    )
    EVIDENCE_SNAPSHOT_MISMATCH = "evidence_snapshot_mismatch"
    REFERENCE_NOT_LOADED = "reference_not_loaded"
    REFERENCE_NOT_RETRIEVED = "reference_not_retrieved"
    REFERENCE_LINEAGE_MISMATCH = "reference_lineage_mismatch"
    OUTCOME_SEMANTICS_INVALID = "outcome_semantics_invalid"


class ReviewValidationIssue(_FrozenModel):
    """Describe one safe, structured validation failure."""

    code: ReviewValidationIssueCode
    message: str = Field(min_length=1)
    finding_index: int | None = Field(default=None, ge=0)
    review_unit_id: UUID | None = None
    evidence_chunk_id: str | None = None
    reference_unit_id: UUID | None = None

    @field_validator("message", "evidence_chunk_id", mode="before")
    @classmethod
    def normalize_issue_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ReviewApplicabilityAssessment(_FrozenModel):
    """Carry a backend policy decision without exposing policy internals."""

    status: ReviewApplicabilityStatus
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class TrustedReviewClaims(_FrozenModel):
    """Bind a request to authority resolved by the backend."""

    contract_id: UUID
    knowledge_base_id: UUID
    access_scope: str = Field(min_length=1, max_length=128)
    contract_processing_revision: int = Field(ge=1)
    authorized_review_unit_ids: tuple[UUID, ...] = Field(min_length=1)
    authorized_snapshot_ids: tuple[UUID, ...] = Field(min_length=1)

    @field_validator("access_scope", mode="before")
    @classmethod
    def normalize_scope(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_unique_authority(self) -> TrustedReviewClaims:
        if len(set(self.authorized_review_unit_ids)) != len(
            self.authorized_review_unit_ids
        ):
            raise ValueError("Authorized review-unit IDs must be unique")

        if len(set(self.authorized_snapshot_ids)) != len(
            self.authorized_snapshot_ids
        ):
            raise ValueError("Authorized snapshot IDs must be unique")

        return self


class ReviewRequest(_FrozenModel):
    """Describe one backend-authorized bounded review request."""

    contract_id: UUID
    review_unit_ids: tuple[UUID, ...] = Field(
        min_length=1,
        max_length=100,
    )
    knowledge_base_id: UUID
    question: str | None = Field(default=None, max_length=2000)
    trusted_claims: TrustedReviewClaims

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_trusted_binding(self) -> ReviewRequest:
        if len(set(self.review_unit_ids)) != len(self.review_unit_ids):
            raise ValueError("Review-unit IDs must be unique")

        if self.contract_id != self.trusted_claims.contract_id:
            raise ValueError("Trusted claims belong to another contract")

        if self.knowledge_base_id != self.trusted_claims.knowledge_base_id:
            raise ValueError("Trusted claims belong to another knowledge base")

        if set(self.review_unit_ids) != set(
            self.trusted_claims.authorized_review_unit_ids
        ):
            raise ValueError("Requested review units do not match authority")

        return self


class ReviewUnit(_FrozenModel):
    """Represent exact contract text exposed to the bounded agent."""

    review_unit_id: UUID
    contract_id: UUID
    processing_revision: int = Field(ge=1)
    order_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    label: str | None = Field(default=None, min_length=1, max_length=512)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)

    @field_validator("text", "label", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RetrievedEvidence(_FrozenModel):
    """Bind a core retrieval hit to authorized backend identities."""

    knowledge_base_id: UUID
    snapshot_id: UUID
    reference_unit_ids: tuple[UUID, ...] = Field(min_length=1)
    hit: KnowledgeHit

    @model_validator(mode="after")
    def validate_reference_ids(self) -> RetrievedEvidence:
        if len(set(self.reference_unit_ids)) != len(self.reference_unit_ids):
            raise ValueError("Reference-unit IDs must be unique")
        return self


class ReferenceUnit(_FrozenModel):
    """Represent complete persisted source text used for verification."""

    reference_unit_id: UUID
    snapshot_id: UUID
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_scope: str = Field(min_length=1)
    text: str = Field(min_length=1)
    structure_path: tuple[str, ...] = Field(min_length=1)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)

    @field_validator(
        "source_id",
        "source_version",
        "access_scope",
        "text",
        mode="before",
    )
    @classmethod
    def normalize_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ReviewFinding(_FrozenModel):
    """Represent one structured finding tied to trusted source artifacts."""

    review_unit_id: UUID
    status: ReviewFindingStatus
    contract_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_chunk_ids: tuple[str, ...] = ()
    reference_unit_ids: tuple[UUID, ...] = ()

    @field_validator("contract_quote", "reason", mode="before")
    @classmethod
    def normalize_explanation(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("evidence_chunk_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(
            item.strip() if isinstance(item, str) else item
            for item in value
        )

    @model_validator(mode="after")
    def validate_citations(self) -> ReviewFinding:
        if len(set(self.evidence_chunk_ids)) != len(
            self.evidence_chunk_ids
        ):
            raise ValueError("Evidence chunk IDs must be unique")

        if any(not chunk_id for chunk_id in self.evidence_chunk_ids):
            raise ValueError("Evidence chunk IDs must not be blank")

        if len(set(self.reference_unit_ids)) != len(self.reference_unit_ids):
            raise ValueError("Reference-unit IDs must be unique")

        conclusive = self.status in {
            ReviewFindingStatus.NON_COMPLIANT,
            ReviewFindingStatus.NO_ISSUE_FOUND,
        }

        if conclusive and (
            not self.evidence_chunk_ids or not self.reference_unit_ids
        ):
            raise ValueError(
                "A conclusive finding requires evidence and references"
            )

        return self


class ReviewCoverage(_FrozenModel):
    """Record requested and actually reviewed units without hiding gaps."""

    requested_review_unit_ids: tuple[UUID, ...] = Field(min_length=1)
    reviewed_review_unit_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> ReviewCoverage:
        requested = self.requested_review_unit_ids
        reviewed = self.reviewed_review_unit_ids

        if len(set(requested)) != len(requested):
            raise ValueError("Requested coverage IDs must be unique")

        if len(set(reviewed)) != len(reviewed):
            raise ValueError("Reviewed coverage IDs must be unique")

        if not set(reviewed).issubset(requested):
            raise ValueError("Reviewed coverage exceeds requested scope")

        return self

    @property
    def complete(self) -> bool:
        return set(self.reviewed_review_unit_ids) == set(
            self.requested_review_unit_ids
        )

    @property
    def missing_review_unit_ids(self) -> tuple[UUID, ...]:
        reviewed = set(self.reviewed_review_unit_ids)
        return tuple(
            unit_id
            for unit_id in self.requested_review_unit_ids
            if unit_id not in reviewed
        )


class ReviewResult(_FrozenModel):
    """Represent a validated domain result, separate from job status."""

    outcome: ReviewOutcome
    contract_id: UUID
    knowledge_base_id: UUID
    question: str | None = Field(default=None, max_length=2000)
    summary: str = Field(min_length=1)
    coverage: ReviewCoverage
    findings: tuple[ReviewFinding, ...] = ()
    validation_issues: tuple[ReviewValidationIssue, ...] = ()

    @field_validator("question", mode="before")
    @classmethod
    def normalize_result_question(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_result_scope(self) -> ReviewResult:
        finding_ids = tuple(
            finding.review_unit_id for finding in self.findings
        )

        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("Each review unit may have only one finding")

        if set(finding_ids) != set(self.coverage.reviewed_review_unit_ids):
            raise ValueError("Findings must match reviewed coverage exactly")

        if self.outcome is ReviewOutcome.REVIEW_COMPLETED:
            if not self.coverage.complete:
                raise ValueError("Completed review requires complete coverage")

            non_conclusive = {
                ReviewFindingStatus.INSUFFICIENT_EVIDENCE,
                ReviewFindingStatus.NEEDS_HUMAN_REVIEW,
            }

            if any(
                finding.status in non_conclusive
                for finding in self.findings
            ):
                raise ValueError(
                    "Completed review cannot contain unresolved findings"
                )

        if (
            self.outcome is ReviewOutcome.VALIDATION_FAILED
            and self.findings
        ):
            raise ValueError("Invalid findings must not be published")

        if self.outcome is ReviewOutcome.VALIDATION_FAILED:
            if not self.validation_issues:
                raise ValueError(
                    "Validation failure requires structured issues"
                )
        elif self.validation_issues:
            raise ValueError(
                "Only validation failure may expose validation issues"
            )

        return self
