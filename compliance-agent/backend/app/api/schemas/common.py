"""Shared HTTP response contracts."""

from pydantic import BaseModel, ConfigDict


class ApiSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ErrorResponse(ApiSchema):
    code: str
    message: str
