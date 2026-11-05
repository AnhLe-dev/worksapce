"""Celery entrypoint accepting only a durable job identifier."""

from uuid import UUID

from celery.exceptions import Reject
from celery.signals import worker_process_shutdown

from app.worker.celery_app import celery_app
from app.worker.runtime import clear_worker_runtime, get_worker_runtime


@celery_app.task(name="jobs.dispatch")
def dispatch_job(job_id: str) -> dict[str, object] | None:
    try:
        parsed_job_id = UUID(job_id)
    except (TypeError, ValueError):
        raise Reject("Job ID is invalid", requeue=False) from None

    result = get_worker_runtime().execute(parsed_job_id)

    if result is None:
        return None

    return {
        "job_id": str(result.job_id),
        "status": result.status.value,
    }


@worker_process_shutdown.connect
def close_runtime_on_worker_shutdown(**kwargs) -> None:
    del kwargs
    clear_worker_runtime()
