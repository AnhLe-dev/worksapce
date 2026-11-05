"""Immutable output contracts for grounded generation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentRole,
    SourceSpan,
)
from rag_core.retrieval.models import KnowledgeHit


class _FrozenModel(BaseModel):
    """Reject unknown fields and mutation in generation contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class AnswerStatus(StrEnum):
    """Distinguish grounded answers from deterministic abstention."""

    GROUNDED = "grounded"
    ABSTAINED = "abstained"


class GenerationPrompt(_FrozenModel):
    """Represent provider-neutral instructions and user input."""

    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)

    @field_validator("instructions", "input_text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        """Reject whitespace-only prompt components."""

        if isinstance(value, str):
            return value.strip()

        return value


class GroundedCitation(_FrozenModel):
    """Map one answer label to exact retrieved evidence."""

    citation_id: str = Field(pattern=r"^S[1-9]\d*$")
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_scope: str = Field(min_length=1)
    index_profile: str = Field(min_length=1)
    text: str = Field(min_length=1)
    structure_path: tuple[str, ...] = Field(min_length=1)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    score: float
    document_format: DocumentFormat | None = None
    role: DocumentRole | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> GroundedCitation:
        """Reuse retrieval invariants for citation provenance."""

        KnowledgeHit(
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            source_version=self.source_version,
            source_checksum_sha256=self.source_checksum_sha256,
            access_scope=self.access_scope,
            index_profile=self.index_profile,
            text=self.text,
            structure_path=self.structure_path,
            source_spans=self.source_spans,
            page_start=self.page_start,
            page_end=self.page_end,
            score=self.score,
            document_format=self.document_format,
            role=self.role,
            metadata=dict(self.metadata),
        )

        return self


class GroundedAnswer(_FrozenModel):
    """Represent either a cited answer or an explicit abstention."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    status: AnswerStatus
    citations: tuple[GroundedCitation, ...] = ()
    retrieved_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_status(self) -> GroundedAnswer:
        """Keep answer status consistent with evidence usage."""

        if self.status is AnswerStatus.GROUNDED:
            if not self.citations or self.retrieved_count <= 0:
                raise ValueError(
                    "Grounded answers require retrieved citations"
                )
        elif self.citations:
            raise ValueError(
                "Abstained answers cannot contain citations"
            )

        return self


class GroundedExecutionTrace(_FrozenModel):
    """Record the exact evidence rendered for one generation call."""

    retrieved_contexts: tuple[str, ...] = ()
    retrieved_context_ids: tuple[str, ...] = ()
    rendered_context: str | None = None
    used_citation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_context_mapping(self) -> GroundedExecutionTrace:
        """Require one identity per section and honest empty traces."""

        if len(self.retrieved_contexts) != len(
            self.retrieved_context_ids
        ):
            raise ValueError(
                "Retrieved contexts and IDs must have the same length"
            )

        if self.retrieved_contexts and not self.rendered_context:
            raise ValueError(
                "Rendered context is required for retrieved evidence"
            )

        if not self.retrieved_contexts and self.rendered_context is not None:
            raise ValueError(
                "Empty retrieval cannot declare rendered context"
            )

        return self


class GroundedExecution(_FrozenModel):
    """Contain one grounded result and its exact execution trace."""

    result: GroundedAnswer
    trace: GroundedExecutionTrace
