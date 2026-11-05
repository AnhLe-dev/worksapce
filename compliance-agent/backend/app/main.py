"""FastAPI composition root for the Release A backend."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import ApiError, api_error_handler
from app.api.routes import contracts, health, jobs, knowledge, review
from app.api.runtime import ApiRuntime, create_api_runtime
from app.core.config import Settings, get_settings

RuntimeFactory = Callable[[Settings], ApiRuntime]


def create_app(
    *,
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = create_api_runtime,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = runtime_factory(resolved_settings)
        application.state.runtime = runtime

        try:
            yield
        finally:
            runtime.close()

    application = FastAPI(
        title="Compliance Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(ApiError, api_error_handler)
    application.include_router(health.router)
    application.include_router(contracts.router)
    application.include_router(knowledge.router)
    application.include_router(review.router)
    application.include_router(jobs.router)
    return application


app = create_app()
