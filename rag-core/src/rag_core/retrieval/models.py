"""Immutable contracts for provider-independent retrieval."""

from __future__ import annotations

from math import isfinite

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


class _FrozenModel(BaseModel):
    """Reject unknown fields and mutation in retrieval contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class RetrievalScope(_FrozenModel):
    """Identify one exact immutable source snapshot and index profile."""

    access_scope: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    index_profile: str = Field(min_length=1)

    @field_validator(
        "access_scope",
        "source_id",
        "source_version",
        "index_profile",
        mode="before",
    )
    @classmethod
    def normalize_identity(cls, value: object) -> object:
        """Remove boundary whitespace before validating identity."""

        if isinstance(value, str):
            return value.strip()

        return value


class SourceRetrievalRequest(_FrozenModel):
    """Describe retrieval from one exact source snapshot."""

    query: str = Field(min_length=1)
    scope: RetrievalScope
    limit: int = Field(gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> object:
        """Normalize user text at the core boundary."""

        if isinstance(value, str):
            return value.strip()

        return value


class RetrievalRequest(_FrozenModel):
    """Describe retrieval across backend-resolved source snapshots."""

    query: str = Field(min_length=1)
    scopes: tuple[RetrievalScope, ...] = Field(min_length=1)
    limit: int = Field(gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> object:
        """Normalize user text at the core boundary."""

        if isinstance(value, str):
            return value.strip()

        return value

    @model_validator(mode="after")
    def validate_unique_scopes(self) -> RetrievalRequest:
        """Reject duplicate snapshots before provider calls."""

        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError(
                "Retrieval request contains duplicate source scopes"
            )

        return self


class KnowledgeHit(_FrozenModel):
    """Represent one ranked chunk with exact immutable lineage."""

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

    @field_validator(
        "chunk_id",
        "source_id",
        "source_version",
        "access_scope",
        "index_profile",
        "text",
        mode="before",
    )
    @classmethod
    def normalize_required_text(cls, value: object) -> object:
        """Normalize required text fields before validation."""

        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Reject scores that cannot be ranked deterministically."""

        if not isfinite(value):
            raise ValueError(
                "Retrieval score must be finite"
            )

        return value

    @model_validator(mode="after")
    def validate_page_provenance(self) -> KnowledgeHit:
        """Require page ranges to reflect real source spans."""

        span_pages = tuple(
            span.page_number
            for span in self.source_spans
        )
        all_paged = all(page is not None for page in span_pages)
        all_unpaged = all(page is None for page in span_pages)

        if not (all_paged or all_unpaged):
            raise ValueError(
                "Knowledge hit cannot mix paged and unpaged spans"
            )

        if all_unpaged:
            if self.page_start is not None or self.page_end is not None:
                raise ValueError(
                    "Unpaged evidence cannot declare a page range"
                )

            return self

        if self.page_start is None or self.page_end is None:
            raise ValueError(
                "Paged evidence requires a page range"
            )

        concrete_pages = tuple(
            page
            for page in span_pages
            if page is not None
        )

        if (
            self.page_start != min(concrete_pages)
            or self.page_end != max(concrete_pages)
        ):
            raise ValueError(
                "Page range must match evidence spans"
            )

        return self
