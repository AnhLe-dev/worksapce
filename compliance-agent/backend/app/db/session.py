"""Database engine, session factory, and transaction helpers."""

from collections.abc import Generator
from contextlib import contextmanager
from math import ceil

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

SessionFactory = sessionmaker[Session]


def create_database_engine(settings: Settings) -> Engine:
    """Create an engine without logging its credential-bearing URL."""

    database_url = settings.database_url.get_secret_value()
    options: dict[str, object] = {"pool_pre_ping": True}

    if database_url.startswith("postgresql"):
        options["connect_args"] = {
            "connect_timeout": ceil(settings.dependency_timeout_seconds)
        }
        options["pool_timeout"] = settings.dependency_timeout_seconds

    return create_engine(database_url, **options)


def create_session_factory(engine: Engine) -> SessionFactory:
    """Bind explicit non-expiring sessions to an engine."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(
    session_factory: SessionFactory,
) -> Generator[Session, None, None]:
    """Commit successful work and roll back every raised failure."""

    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
