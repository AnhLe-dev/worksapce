"""Release B request schema for bounded contract review jobs."""

from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.api.schemas.common import ApiSchema


class ReviewJobRequest(ApiSchema):
    knowledge_base_id: UUID
    expected_processing_revision: int = Field(ge=1)
    review_unit_ids: tuple[UUID, ...] = Field(
        min_length=1,
        max_length=100,
    )
    question: str | None = Field(default=None, max_length=2000)
    idempotency_key: str = Field(min_length=1, max_length=255)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_unique_units(self) -> "ReviewJobRequest":
        if len(set(self.review_unit_ids)) != len(self.review_unit_ids):
            raise ValueError("Review-unit IDs must be unique")
        return self
