"""Application composition and readiness checks for the HTTP process."""

from dataclasses import dataclass

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.db.session import (
    SessionFactory,
    create_database_engine,
    create_session_factory,
)
from app.documents.storage import (
    ObjectStorage,
    ObjectStorageUnavailableError,
    create_object_storage,
)
from app.knowledge.index_profile import build_index_profile


@dataclass(slots=True)
class ApiRuntime:
    """Resources shared by requests and owned by the application lifespan."""

    settings: Settings
    engine: Engine
    session_factory: SessionFactory
    storage: ObjectStorage
    index_profile: str

    def readiness_checks(self) -> dict[str, str]:
        checks: dict[str, str] = {}

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            checks["database"] = "ready"
        except SQLAlchemyError:
            checks["database"] = "unavailable"

        try:
            self.storage.verify_bucket()
            checks["object_storage"] = "ready"
        except ObjectStorageUnavailableError:
            checks["object_storage"] = "unavailable"

        return checks

    def close(self) -> None:
        self.engine.dispose()


def create_api_runtime(settings: Settings) -> ApiRuntime:
    engine = create_database_engine(settings)
    return ApiRuntime(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        storage=create_object_storage(settings),
        index_profile=build_index_profile(settings),
    )
