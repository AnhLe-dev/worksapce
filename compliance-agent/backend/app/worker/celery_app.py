"""Celery transport configuration; trusted payload remains in PostgreSQL."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()
celery_app = Celery(
    "compliance-worker",
    broker=settings.celery_broker_url.get_secret_value(),
    include=("app.worker.tasks",),
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=("json",),
    result_backend=None,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=max(settings.job_lease_seconds - 10, 1),
    task_time_limit=settings.job_lease_seconds - 1,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_routes={
        "jobs.dispatch": {"queue": "compliance_jobs"},
    },
)
