"""Run Release A through live Compose dependencies and write evidence."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from uuid import UUID, uuid4

import httpx
from alembic.config import Config
from alembic.script import ScriptDirectory
from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from rag_core.adapters.openai_embeddings import OpenAIEmbeddingProvider
from rag_core.adapters.qdrant import QdrantVectorSearch
from rag_core.retrieval.dense import DenseSourceRetriever
from rag_core.retrieval.models import RetrievalScope, SourceRetrievalRequest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings, get_settings
from app.db.session import create_database_engine, create_session_factory
from app.documents.storage import require_secret
from app.jobs.db_models import OutboxEventRecord, TechnicalJobRecord
from app.jobs.models import JobStatus
from app.jobs.outbox_models import OutboxStatus

CONTRACT_TEXT = (
    "The supplier must deliver the equipment by 30 June.\n\n"
    "Payment is due within ten days after verified delivery."
)
KNOWLEDGE_TEXT = (
    "A return request requires the original receipt and purchase date.\n\n"
    "Approved refunds are issued through the original payment method."
)
REQUIRED_PATHS = {
    "/api/v1/contracts",
    "/api/v1/contracts/{contract_id}",
    "/api/v1/contracts/{contract_id}/processing-jobs",
    "/api/v1/contracts/{contract_id}/review-jobs",
    "/api/v1/contracts/{contract_id}/review-units",
    "/api/v1/knowledge-bases",
    "/api/v1/knowledge-bases/{knowledge_base_id}/documents",
    "/api/v1/knowledge-bases/{knowledge_base_id}",
    "/api/v1/jobs/{job_id}",
    "/health/live",
    "/health/ready",
}
BLOCKED_JOB_CODES = {
    "dependency_unavailable",
    "index_provider_not_configured",
    "index_provider_rejected",
    "job_attempts_exhausted",
}


class AcceptanceFailure(RuntimeError):
    """Represent an assertion failure in the live release gate."""


class AcceptanceBlocked(RuntimeError):
    """Represent unavailable credentials or live infrastructure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    expected_status: int,
    **kwargs,
):
    try:
        response = client.request(method, path, **kwargs)
    except httpx.TransportError as error:
        raise AcceptanceBlocked(
            f"Backend is unavailable for {method} {path}"
        ) from error

    if response.status_code != expected_status:
        raise AcceptanceFailure(
            f"Unexpected HTTP status for {method} {path}: "
            f"expected={expected_status}, actual={response.status_code}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise AcceptanceFailure(
            f"Response is not JSON for {method} {path}"
        ) from error


def _poll_job(
    client: httpx.Client,
    job_id: str,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, object], list[str]]:
    deadline = monotonic() + timeout_seconds
    transitions: list[str] = []

    while monotonic() < deadline:
        job = _request_json(
            client,
            "GET",
            f"/api/v1/jobs/{job_id}",
            expected_status=200,
        )
        status = str(job["status"])

        if not transitions or transitions[-1] != status:
            transitions.append(status)

        if status == JobStatus.SUCCEEDED.value:
            return job, transitions

        if status == JobStatus.FAILED.value:
            error_code = str(job.get("error_code") or "job_failed")
            message = str(job.get("error_message") or "Job failed")

            if error_code in BLOCKED_JOB_CODES:
                raise AcceptanceBlocked(
                    f"Live job dependency failed: {error_code}: {message}"
                )

            raise AcceptanceFailure(
                f"Live job failed: {error_code}: {message}"
            )

        sleep(1)

    raise AcceptanceFailure(f"Job did not finish before timeout: {job_id}")


def _verify_database(
    settings: Settings,
    job_ids: tuple[str, ...],
) -> dict[str, object]:
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        with engine.connect() as connection:
            database_revision = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )

        alembic_config = Config("alembic.ini")
        expected_revision = ScriptDirectory.from_config(
            alembic_config
        ).get_current_head()
        _require(
            database_revision == expected_revision,
            "Database migration revision does not match Alembic head",
        )
        attempts: dict[str, int] = {}

        with session_factory() as session:
            for raw_job_id in job_ids:
                job_id = UUID(raw_job_id)
                job = session.get(TechnicalJobRecord, job_id)
                _require(job is not None, "Live job row was not persisted")
                assert job is not None
                _require(
                    job.status is JobStatus.SUCCEEDED,
                    "Persisted live job is not succeeded",
                )
                attempts[raw_job_id] = job.attempt_count
                events = session.scalars(
                    select(OutboxEventRecord).where(
                        OutboxEventRecord.job_id == job_id
                    )
                ).all()
                _require(bool(events), "Live job has no outbox event")
                _require(
                    all(
                        event.status is OutboxStatus.PUBLISHED
                        for event in events
                    ),
                    "Live job has an unpublished outbox event",
                )

        return {
            "migration_head": str(database_revision),
            "job_attempt_counts": attempts,
            "outbox": "published",
        }
    except SQLAlchemyError as error:
        raise AcceptanceBlocked("PostgreSQL verification failed") from error
    finally:
        engine.dispose()


def _verify_retrieval(
    settings: Settings,
    member: dict[str, object],
) -> dict[str, object]:
    try:
        api_key = require_secret(settings.openai_api_key, "OPENAI_API_KEY")
    except ValueError as error:
        raise AcceptanceBlocked(str(error)) from error

    if settings.qdrant_cloud_url is None:
        raise AcceptanceBlocked("QDRANT_CLOUD_URL is not configured")

    openai_client = OpenAI(
        api_key=api_key,
        timeout=settings.dependency_timeout_seconds,
    )
    qdrant_api_key = (
        None
        if settings.qdrant_cloud_api_key is None
        else settings.qdrant_cloud_api_key.get_secret_value() or None
    )
    qdrant_client = QdrantClient(
        url=settings.qdrant_cloud_url,
        api_key=qdrant_api_key,
        timeout=settings.dependency_timeout_seconds,
    )

    try:
        embedding = OpenAIEmbeddingProvider(
            client=openai_client,
            model=settings.default_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        retriever = DenseSourceRetriever(
            embedding_provider=embedding,
            vector_search=QdrantVectorSearch(
                client=qdrant_client,
                collection_name=settings.qdrant_collection_name,
                vector_dimensions=settings.embedding_dimensions,
            ),
            expected_dimensions=settings.embedding_dimensions,
        )
        scope = RetrievalScope(
            access_scope=settings.system_access_scope,
            source_id=str(member["source_id"]),
            source_version=str(member["source_version"]),
            source_checksum_sha256=str(
                member["source_checksum_sha256"]
            ),
            index_profile=str(member["index_profile"]),
        )
        hits = retriever.retrieve_source(
            SourceRetrievalRequest(
                query="What is required for a return request?",
                scope=scope,
                limit=3,
            )
        )
        _require(bool(hits), "Live exact-snapshot retrieval returned no hits")
        _require(
            all(hit.source_checksum_sha256 == scope.source_checksum_sha256 for hit in hits),
            "Live retrieval escaped the expected snapshot checksum",
        )
        return {
            "hit_count": len(hits),
            "chunk_ids": [hit.chunk_id for hit in hits],
            "exact_snapshot_verified": True,
        }
    except (
        OpenAIError,
        ResponseHandlingException,
        UnexpectedResponse,
        httpx.TransportError,
    ) as error:
        raise AcceptanceBlocked(
            "Live OpenAI or Qdrant retrieval is unavailable"
        ) from error
    finally:
        openai_client.close()
        qdrant_client.close()


def _run_workflow(
    *,
    base_url: str,
    timeout_seconds: float,
    commit_sha: str,
) -> dict[str, object]:
    settings = get_settings()
    run_id = uuid4().hex[:12]

    if settings.openai_api_key is None:
        raise AcceptanceBlocked("OPENAI_API_KEY is not configured")

    with httpx.Client(base_url=base_url, timeout=30) as client:
        live = _request_json(
            client,
            "GET",
            "/health/live",
            expected_status=200,
        )
        ready = _request_json(
            client,
            "GET",
            "/health/ready",
            expected_status=200,
        )
        schema = _request_json(
            client,
            "GET",
            "/openapi.json",
            expected_status=200,
        )
        paths = set(schema["paths"])
        _require(REQUIRED_PATHS <= paths, "Release OpenAPI is incomplete")

        knowledge_base = _request_json(
            client,
            "POST",
            "/api/v1/knowledge-bases",
            expected_status=201,
            json={
                "name": f"Live return policy {run_id}",
                "applicability_metadata": {"acceptance_run": run_id},
            },
        )
        knowledge_base_id = str(knowledge_base["knowledge_base_id"])
        knowledge_upload = _request_json(
            client,
            "POST",
            f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
            expected_status=202,
            files={
                "file": (
                    "return-policy.txt",
                    KNOWLEDGE_TEXT.encode(),
                    "text/plain",
                )
            },
            data={
                "source_id": f"live-return-policy-{run_id}",
                "source_version": "1",
                "idempotency_key": f"live-knowledge-{run_id}",
            },
        )
        knowledge_job_id = str(knowledge_upload["job"]["job_id"])
        knowledge_job, knowledge_transitions = _poll_job(
            client,
            knowledge_job_id,
            timeout_seconds=timeout_seconds,
        )
        knowledge_detail = _request_json(
            client,
            "GET",
            f"/api/v1/knowledge-bases/{knowledge_base_id}",
            expected_status=200,
        )
        _require(
            len(knowledge_detail["members"]) == 1,
            "Live knowledge base does not contain exactly one member",
        )
        member = knowledge_detail["members"][0]
        _require(
            member["snapshot_status"] == "ready",
            "Live knowledge snapshot is not ready",
        )

        contract = _request_json(
            client,
            "POST",
            "/api/v1/contracts",
            expected_status=201,
            files={
                "file": (
                    "equipment-contract.txt",
                    CONTRACT_TEXT.encode(),
                    "text/plain",
                )
            },
            data={"source_version": f"live-{run_id}"},
        )
        contract_id = str(contract["contract_id"])
        contract_job = _request_json(
            client,
            "POST",
            f"/api/v1/contracts/{contract_id}/processing-jobs",
            expected_status=202,
            json={
                "expected_revision": 0,
                "idempotency_key": f"live-contract-{run_id}",
            },
        )
        contract_job_id = str(contract_job["job_id"])
        completed_contract_job, contract_transitions = _poll_job(
            client,
            contract_job_id,
            timeout_seconds=timeout_seconds,
        )
        completed_contract = _request_json(
            client,
            "GET",
            f"/api/v1/contracts/{contract_id}",
            expected_status=200,
        )
        review_units = _request_json(
            client,
            "GET",
            f"/api/v1/contracts/{contract_id}/review-units",
            expected_status=200,
        )
        _require(
            completed_contract["status"] == "ready",
            "Live contract did not reach READY",
        )
        _require(
            completed_contract["processing_revision"] == 1,
            "Live contract processing revision is not one",
        )
        _require(len(review_units) == 2, "Live contract review units are incomplete")
        _require(
            all(unit["source_spans"] for unit in review_units),
            "Live review unit lost source provenance",
        )

    database = _verify_database(
        settings,
        (knowledge_job_id, contract_job_id),
    )
    retrieval = _verify_retrieval(settings, member)
    return {
        "status": "LIVE_ACCEPTED",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": "compose-local",
        "commit_sha": commit_sha,
        "run_id": run_id,
        "health": {"live": live, "ready": ready},
        "openapi_required_paths": len(REQUIRED_PATHS),
        "knowledge": {
            "knowledge_base_id": knowledge_base_id,
            "snapshot_id": member["snapshot_id"],
            "source_id": member["source_id"],
            "source_checksum_sha256": member[
                "source_checksum_sha256"
            ],
            "job_id": knowledge_job_id,
            "job_transitions": knowledge_transitions,
            "chunk_count": knowledge_job["result_payload"]["chunk_count"],
            "snapshot_status": member["snapshot_status"],
        },
        "contract": {
            "contract_id": contract_id,
            "job_id": contract_job_id,
            "job_transitions": contract_transitions,
            "processing_revision": completed_contract[
                "processing_revision"
            ],
            "review_unit_count": len(review_units),
            "job_result": completed_contract_job["result_payload"],
        },
        "database_and_outbox": database,
        "retrieval": retrieval,
        "claim_limits": [
            "Not production security acceptance",
            "Not multi-tenant AuthN/AuthZ acceptance",
            "Not legal or semantic correctness acceptance",
        ],
    }


def _write_report(
    report: dict[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Release A live acceptance\n\n"
        f"- Status: **{report['status']}**\n"
        f"- Timestamp: `{report['timestamp_utc']}`\n"
        f"- Environment: `{report['environment']}`\n"
        f"- Commit: `{report['commit_sha']}`\n"
        f"- Run ID: `{report.get('run_id', 'unavailable')}`\n\n"
        "The JSON artifact beside this file contains redacted IDs, state "
        "transitions, migration head, outbox evidence, exact-snapshot "
        "retrieval evidence, and explicit claim limits.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--poll-timeout", type=float, default=240)
    parser.add_argument("--commit-sha", default="unknown")
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("/workspace/reports/release-a-live.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("/workspace/reports/live-acceptance.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = _run_workflow(
            base_url=args.base_url,
            timeout_seconds=args.poll_timeout,
            commit_sha=args.commit_sha,
        )
        exit_code = 0
    except AcceptanceBlocked as error:
        report = {
            "status": "LIVE_BLOCKED",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "environment": "compose-local",
            "commit_sha": args.commit_sha,
            "error": str(error),
        }
        exit_code = 2
    except AcceptanceFailure as error:
        report = {
            "status": "FAIL",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "environment": "compose-local",
            "commit_sha": args.commit_sha,
            "error": str(error),
        }
        exit_code = 1

    _write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
