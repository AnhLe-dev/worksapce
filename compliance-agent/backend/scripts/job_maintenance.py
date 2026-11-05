"""Continuously relay outbox events and reconcile expired job leases."""

import logging
from pathlib import Path
from socket import gethostname
from time import sleep

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.db.session import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from app.jobs.outbox_dispatcher import OutboxDispatcher
from app.jobs.reconciler import JobReconciler
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService
from app.worker.celery_app import celery_app
from app.worker.dispatcher import CeleryPublisher
from app.worker.handlers import DomainFailureRecorder

READY_FILE = Path("/tmp/job-maintenance.ready")
logger = logging.getLogger("job-maintenance")


def _job_service(session, settings) -> JobService:
    return JobService(
        jobs=JobRepository(session),
        outbox=OutboxRepository(session),
        lease_seconds=settings.job_lease_seconds,
        max_attempts=settings.job_max_attempts,
    )


def main() -> int:
    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    publisher = CeleryPublisher(celery_app)
    relay_id = f"relay-{gethostname()}"
    logging.basicConfig(level=settings.log_level)

    try:
        while True:
            try:
                with session_scope(session_factory) as session:
                    dispatch = OutboxDispatcher(
                        repository=OutboxRepository(session),
                        publisher=publisher,
                        relay_id=relay_id,
                        lease_seconds=settings.outbox_lease_seconds,
                        batch_size=settings.outbox_batch_size,
                    ).dispatch()

                with session_scope(session_factory) as session:
                    reconciliation = JobReconciler(
                        jobs=_job_service(session, settings),
                        outbox=OutboxRepository(session),
                        failure_handler=DomainFailureRecorder(session),
                    ).reconcile()

                READY_FILE.touch()
                logger.info(
                    "job maintenance cycle: claimed=%s published=%s "
                    "publish_failed=%s released=%s retried=%s failed=%s",
                    dispatch.claimed,
                    dispatch.published,
                    dispatch.failed,
                    reconciliation.released_outbox_events,
                    reconciliation.retried_jobs,
                    reconciliation.failed_jobs,
                )
            except SQLAlchemyError:
                logger.exception("job maintenance database cycle failed")

            sleep(settings.job_maintenance_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        READY_FILE.unlink(missing_ok=True)
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
