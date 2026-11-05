"""Offline contract tests for the canonical Release A HTTP surface."""

from collections.abc import Generator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from rag_core.indexing import IndexingResult, IndexingStatus
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from app.api.runtime import ApiRuntime
from app.contracts.processing_service import ContractProcessingService
from app.contracts.repository import ContractRepository
from app.core.config import Settings
from app.db.base import Base
from app.db.session import create_session_factory
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.repository import StoredObjectRepository
from app.knowledge.index_profile import build_index_profile
from app.knowledge.indexing_service import KnowledgeIndexingService
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.repository import KnowledgeRepository
from app.main import create_app
from app.worker.runtime import create_document_processor
from tests.fakes import FakeObjectStorage


class FakeIndexer:
    def index(self, chunks) -> IndexingResult:
        return IndexingResult(
            collection_name="api_phase3",
            source_id=chunks[0].source_id,
            source_version=chunks[0].source_version,
            source_checksum_sha256=chunks[0].source_checksum_sha256,
            access_scope=chunks[0].access_scope,
            status=IndexingStatus.INDEXED,
            chunk_count=len(chunks),
            uploaded_count=len(chunks),
            stored_count=len(chunks),
            vector_dimensions=3,
        )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    settings = Settings(
        _env_file=None,
        database_url="sqlite+pysqlite://",
        system_access_scope="api-test",
        max_document_upload_bytes=1024 * 1024,
        embedding_dimensions=3,
    )
    runtime = ApiRuntime(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        storage=FakeObjectStorage(),
        index_profile=build_index_profile(settings),
    )
    application = create_app(
        settings=settings,
        runtime_factory=lambda _: runtime,
    )

    with TestClient(application) as test_client:
        yield test_client

    Base.metadata.drop_all(engine)


def upload_contract(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/v1/contracts",
        files={"file": ("contract.txt", b"Payment within five days.")},
        data={"source_version": "v1"},
    )
    assert response.status_code == 201
    return response.json()


def test_openapi_exposes_release_a_and_b_job_routes(
    client: TestClient,
) -> None:
    document = client.get("/openapi.json").json()
    paths = set(document["paths"])

    assert {
        "/api/v1/contracts",
        "/api/v1/contracts/{contract_id}",
        "/api/v1/contracts/{contract_id}/processing-jobs",
        "/api/v1/contracts/{contract_id}/review-units",
        "/api/v1/contracts/{contract_id}/review-jobs",
        "/api/v1/knowledge-bases",
        "/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        "/api/v1/knowledge-bases/{knowledge_base_id}",
        "/api/v1/jobs/{job_id}",
        "/health/live",
        "/health/ready",
    } <= paths
    request_bodies = str(
        [
            operation.get("requestBody", {})
            for path in document["paths"].values()
            for operation in path.values()
            if isinstance(operation, dict)
        ]
    )
    assert "access_scope" not in request_bodies
    assert "index_profile" not in request_bodies


def test_health_distinguishes_liveness_from_readiness(
    client: TestClient,
) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "live"}
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "checks": {
            "database": "ready",
            "object_storage": "ready",
        },
    }


def test_contract_upload_job_and_status_are_scope_bound(
    client: TestClient,
) -> None:
    contract = upload_contract(client)
    contract_id = contract["contract_id"]

    fetched = client.get(f"/api/v1/contracts/{contract_id}")
    job = client.post(
        f"/api/v1/contracts/{contract_id}/processing-jobs",
        json={"expected_revision": 0, "idempotency_key": "process-v1"},
    )

    assert fetched.status_code == 200
    assert fetched.json()["role"] == "contract"
    assert job.status_code == 202
    assert job.json()["contract_id"] == contract_id
    job_id = job.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
    units = client.get(
        f"/api/v1/contracts/{contract_id}/review-units"
    )
    assert units.status_code == 200
    assert units.json() == []


def test_review_job_requires_a_ready_contract(client: TestClient) -> None:
    contract = upload_contract(client)
    response = client.post(
        f"/api/v1/contracts/{contract['contract_id']}/review-jobs",
        json={
            "knowledge_base_id": "20000000-0000-0000-0000-000000000001",
            "expected_processing_revision": 1,
            "review_unit_ids": [
                "30000000-0000-0000-0000-000000000001"
            ],
            "question": "Does the selected policy apply?",
            "idempotency_key": "review-before-processing",
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "contract_not_ready"


def test_review_job_requires_explicit_knowledge_base(
    client: TestClient,
) -> None:
    contract = upload_contract(client)
    response = client.post(
        f"/api/v1/contracts/{contract['contract_id']}/review-jobs",
        json={
            "expected_processing_revision": 1,
            "review_unit_ids": [
                "30000000-0000-0000-0000-000000000001"
            ],
            "question": "Does any policy apply?",
            "idempotency_key": "missing-selected-kb",
        },
    )

    assert response.status_code == 422


def test_review_job_accepts_current_units_and_ready_selected_kb(
    client: TestClient,
    tmp_path: Path,
) -> None:
    contract = upload_contract(client)
    runtime = client.app.state.runtime
    contract_path = tmp_path / "contract.txt"
    contract_path.write_text(
        "Payment within five days.",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.txt"
    policy_path.write_text(
        "Payment must occur within five days.",
        encoding="utf-8",
    )

    with runtime.session_factory() as session:
        ContractProcessingService(
            repository=ContractRepository(session),
            processor=create_document_processor(runtime.settings),
        ).process(
            contract_id=UUID(contract["contract_id"]),
            trusted_scope=runtime.settings.system_access_scope,
            expected_revision=0,
            file_path=contract_path,
        )
        stored = StoredObjectIngestionService(
            repository=StoredObjectRepository(session),
            storage=runtime.storage,
            max_upload_bytes=4096,
        ).ingest(policy_path).stored_object
        knowledge_repository = KnowledgeRepository(session)
        knowledge = KnowledgeIngestionService(
            repository=knowledge_repository,
        )
        knowledge_base = knowledge.create_base(
            name="API selected policy",
            trusted_scope=runtime.settings.system_access_scope,
        )
        registration = knowledge.register_document(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            stored_object=stored,
            source_id="api-policy",
            source_version="1",
            filename=policy_path.name,
            index_profile=runtime.index_profile,
            trusted_scope=runtime.settings.system_access_scope,
        )
        KnowledgeIndexingService(
            repository=knowledge_repository,
            ingestion=KnowledgeIngestionService(
                repository=knowledge_repository,
                processor=create_document_processor(runtime.settings),
            ),
            indexer=FakeIndexer(),
        ).index(
            snapshot_id=registration.snapshot.snapshot_id,
            trusted_scope=runtime.settings.system_access_scope,
            file_path=policy_path,
        )
        session.commit()

    units = client.get(
        f"/api/v1/contracts/{contract['contract_id']}/review-units"
    ).json()
    response = client.post(
        f"/api/v1/contracts/{contract['contract_id']}/review-jobs",
        json={
            "knowledge_base_id": str(knowledge_base.knowledge_base_id),
            "expected_processing_revision": 1,
            "review_unit_ids": [units[0]["review_unit_id"]],
            "question": "Does payment follow the selected policy?",
            "idempotency_key": "api-review-ready",
        },
    )

    assert response.status_code == 202
    assert response.json()["kind"] == "contract_review"
    assert response.json()["status"] == "queued"
    assert response.json()["contract_id"] == contract["contract_id"]


def test_knowledge_base_registration_creates_indexing_job(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/knowledge-bases",
        json={
            "name": "Public Rules",
            "applicability_metadata": {"jurisdiction": "global"},
        },
    )
    assert created.status_code == 201
    knowledge_base_id = created.json()["knowledge_base_id"]

    uploaded = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("rules.txt", b"Rule one applies.")},
        data={
            "source_id": "rules",
            "source_version": "2026-01",
            "idempotency_key": "rules-2026-01",
        },
    )

    assert uploaded.status_code == 202
    body = uploaded.json()
    assert body["document"]["role"] == "knowledge"
    assert body["member"]["snapshot_status"] == "pending"
    assert body["job"]["status"] == "queued"

    detail = client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}"
    )
    assert detail.status_code == 200
    assert len(detail.json()["members"]) == 1
    assert len(client.get("/api/v1/knowledge-bases").json()) == 1


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/contracts",
        files={"file": ("contract.docx", b"not supported")},
    )

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_document_format"
