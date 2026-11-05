"""FastAPI dependencies with transaction ownership per request."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.api.runtime import ApiRuntime


def get_runtime(request: Request) -> ApiRuntime:
    return request.app.state.runtime


def get_session(
    runtime: Annotated[ApiRuntime, Depends(get_runtime)],
) -> Generator[Session, None, None]:
    session = runtime.session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


RuntimeDependency = Annotated[ApiRuntime, Depends(get_runtime)]
SessionDependency = Annotated[Session, Depends(get_session)]
