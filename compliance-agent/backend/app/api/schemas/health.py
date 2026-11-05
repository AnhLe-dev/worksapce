"""Liveness and readiness response schemas."""

from app.api.schemas.common import ApiSchema


class LiveResponse(ApiSchema):
    status: str = "live"


class ReadyResponse(ApiSchema):
    status: str
    checks: dict[str, str]
