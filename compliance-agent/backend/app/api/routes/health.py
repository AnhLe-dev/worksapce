"""Process liveness and dependency readiness endpoints."""

from fastapi import APIRouter, Response, status

from app.api.dependencies import RuntimeDependency
from app.api.schemas.health import LiveResponse, ReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LiveResponse)
def live() -> LiveResponse:
    return LiveResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(
    runtime: RuntimeDependency,
    response: Response,
) -> ReadyResponse:
    checks = runtime.readiness_checks()
    is_ready = all(value == "ready" for value in checks.values())

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        checks=checks,
    )
