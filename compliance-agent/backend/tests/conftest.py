"""Shared offline database fixtures for backend unit tests."""

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.contracts import db_models as contract_db_models  # noqa: F401
from app.db.base import Base
from app.documents import db_models as document_db_models  # noqa: F401
from app.jobs import db_models as job_db_models  # noqa: F401
from app.knowledge import db_models as knowledge_db_models  # noqa: F401


@pytest.fixture
def session() -> Generator[Session, None, None]:
    """Provide a clean SQLite transaction with FK checks enabled."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session

    Base.metadata.drop_all(engine)
    engine.dispose()
