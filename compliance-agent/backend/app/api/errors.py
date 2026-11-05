"""Stable HTTP failures at the API boundary."""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.schemas.common import ErrorResponse


class ApiError(RuntimeError):
    """Carry one safe client-facing error without leaking internals."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


async def api_error_handler(
    request: Request,
    error: ApiError,
) -> JSONResponse:
    del request
    body = ErrorResponse(code=error.code, message=error.message)
    return JSONResponse(
        status_code=error.status_code,
        content=body.model_dump(mode="json"),
    )
