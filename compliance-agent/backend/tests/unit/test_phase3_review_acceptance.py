"""Offline Phase 3 acceptance through durable jobs and bounded tools."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from compliance_agent import (
    ReviewCoverage,
    ReviewFinding,
    ReviewFindingStatus,
    ReviewModelAction,
    ReviewOutcome,
    ReviewRequest,
    ReviewResult,
    ReviewToolCall,
    ReviewToolName,
    ReviewToolObservation,
)
from rag_core.indexing import (
    DocumentIndexingService,
    IndexingResult,
    IndexingStatus,
)
from rag_core.ingestion.models import SourceSpan
from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalRequest,
)
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from app.contracts.models import ContractStatus
from app.contracts.processing_service import ContractLifecycleService
from app.contracts.repository import ContractRepository
from app.core.config import Settings
from app.db.base import Base
from app.db.session import (
    SessionFactory,
    create_session_factory,
    session_scope,
)
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.repository import StoredObjectRepository
from app.jobs.models import JobStatus, TechnicalJob
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService
from app.knowledge.index_profile import build_index_profile
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.repository import KnowledgeRepository
from app.worker.runtime import WorkerRuntime, create_document_processor
from tests.fakes import FakeObjectStorage

POLICY_METADATA = {
    "policy.type": "exact_match_v1",
    "policy.requires.jurisdiction": "vietnam",
}
TRUSTED_METADATA = {"jurisdiction": "vietnam"}


class FakeEmbeddingProvider:
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, float(index), 0.5) for index, _ in enumerate(texts))


class FakeVectorIndex:
    def upsert(self, *, chunks, vectors) -> IndexingResult:
        return IndexingResult(
            collection_name="phase3_acceptance",
            source_id=chunks[0].source_id,
            source_version=chunks[0].source_version,
            source_checksum_sha256=chunks[0].source_checksum_sha256,
            access_scope=chunks[0].access_scope,
            status=IndexingStatus.INDEXED,
            chunk_count=len(chunks),
            uploaded_count=len(chunks),
            stored_count=len(chunks),
            vector_dimensions=len(vectors[0]),
        )


class FixedReviewRetriever:
    def __init__(
        self,
        *,
        return_hits: bool = True,
        escape_scope: bool = False,
    ) -> None:
        self.return_hits = return_hits
        self.escape_scope = escape_scope
        self.call_count = 0

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        self.call_count += 1

        if not self.return_hits:
            return ()

        scope = request.scopes[0]
        text = "A receipt is required for every refund request."
        return (
            KnowledgeHit(
                chunk_id="phase3-evidence-1",
                source_id=scope.source_id,
                source_version=scope.source_version,
                source_checksum_sha256=scope.source_checksum_sha256,
                access_scope=(
                    "escaped-scope"
                    if self.escape_scope
                    else scope.access_scope
                ),
                index_profile=scope.index_profile,
                text=text,
                structure_path=("document",),
                source_spans=(
                    SourceSpan(
                        block_index=0,
                        char_start=0,
                        char_end=len(text),
                    ),
                ),
                score=0.99,
            ),
        )


class ScriptedReviewModel:
    """Drive the real bounded loop using only rendered observations."""

    def __init__(
        self,
        *,
        fabricated_quote: bool = False,
        ambiguous_evidence: bool = False,
    ) -> None:
        self.fabricated_quote = fabricated_quote
        self.ambiguous_evidence = ambiguous_evidence
        self.call_count = 0

    async def next_action(
        self,
        *,
        system_prompt: str,
        request: ReviewRequest,
        observations: tuple[ReviewToolObservation, ...],
    ) -> ReviewModelAction:
        self.call_count += 1
        assert "untrusted" in system_prompt.casefold()

        unit_observation = next(
            (
                item
                for item in observations
                if item.name is ReviewToolName.GET_REVIEW_UNITS
            ),
            None,
        )

        if unit_observation is None:
            return ReviewModelAction(
                tool_calls=(
                    ReviewToolCall(
                        call_id="units-1",
                        name=ReviewToolName.GET_REVIEW_UNITS,
                        arguments={
                            "review_unit_ids": [
                                str(unit_id)
                                for unit_id in request.review_unit_ids
                            ]
                        },
                    ),
                )
            )

        search_observations = tuple(
            item
            for item in observations
            if item.name is ReviewToolName.SEARCH_SELECTED_KNOWLEDGE
        )

        if not search_observations:
            return ReviewModelAction(
                tool_calls=tuple(
                    ReviewToolCall(
                        call_id=f"search-{index}",
                        name=ReviewToolName.SEARCH_SELECTED_KNOWLEDGE,
                        arguments={"review_unit_id": str(unit_id)},
                    )
                    for index, unit_id in enumerate(
                        request.review_unit_ids,
                        start=1,
                    )
                )
            )

        unit_payload = json.loads(unit_observation.content)
        unit = unit_payload["review_units"][0]
        evidence = tuple(
            item
            for observation in search_observations
            for item in json.loads(observation.content)["evidence"]
        )

        if not evidence:
            return ReviewModelAction(
                final_result=ReviewResult(
                    outcome=ReviewOutcome.INSUFFICIENT_EVIDENCE,
                    contract_id=request.contract_id,
                    knowledge_base_id=request.knowledge_base_id,
                    question=request.question,
                    summary="No source evidence was retrieved.",
                    coverage=ReviewCoverage(
                        requested_review_unit_ids=request.review_unit_ids,
                        reviewed_review_unit_ids=request.review_unit_ids,
                    ),
                    findings=(
                        ReviewFinding(
                            review_unit_id=UUID(unit["review_unit_id"]),
                            status=(
                                ReviewFindingStatus.INSUFFICIENT_EVIDENCE
                            ),
                            contract_quote=unit["text"],
                            reason="No evidence supports a conclusion.",
                        ),
                    ),
                )
            )

        reference_ids = tuple(
            UUID(reference_id)
            for item in evidence
            for reference_id in item["reference_unit_ids"]
        )
        loaded_reference_ids = {
            UUID(
                json.loads(item.content)["reference_unit"][
                    "reference_unit_id"
                ]
            )
            for item in observations
            if item.name is ReviewToolName.GET_REFERENCE_UNIT
        }
        missing_reference_ids = tuple(
            reference_id
            for reference_id in reference_ids
            if reference_id not in loaded_reference_ids
        )

        if missing_reference_ids:
            return ReviewModelAction(
                tool_calls=tuple(
                    ReviewToolCall(
                        call_id=f"reference-{index}",
                        name=ReviewToolName.GET_REFERENCE_UNIT,
                        arguments={
                            "reference_unit_id": str(reference_id)
                        },
                    )
                    for index, reference_id in enumerate(
                        missing_reference_ids,
                        start=1,
                    )
                )
            )

        return ReviewModelAction(
            final_result=ReviewResult(
                outcome=(
                    ReviewOutcome.NEEDS_HUMAN_REVIEW
                    if self.ambiguous_evidence
                    else ReviewOutcome.REVIEW_COMPLETED
                ),
                contract_id=request.contract_id,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                summary=(
                    "Evidence is ambiguous and requires human review."
                    if self.ambiguous_evidence
                    else "The selected policy supports this requirement."
                ),
                coverage=ReviewCoverage(
                    requested_review_unit_ids=request.review_unit_ids,
                    reviewed_review_unit_ids=request.review_unit_ids,
                ),
                findings=(
                    ReviewFinding(
                        review_unit_id=UUID(unit["review_unit_id"]),
                        status=(
                            ReviewFindingStatus.NEEDS_HUMAN_REVIEW
                            if self.ambiguous_evidence
                            else ReviewFindingStatus.NO_ISSUE_FOUND
                        ),
                        contract_quote=(
                            "fabricated contract quote"
                            if self.fabricated_quote
                            else unit["text"]
                        ),
                        reason="The full policy reference requires a receipt.",
                        evidence_chunk_ids=tuple(
                            item["chunk_id"] for item in evidence
                        ),
                        reference_unit_ids=reference_ids,
                    ),
                ),
            )
        )


class CrashingReviewModel:
    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        raise RuntimeError("provider leaked unsafe internal detail")


class ModelMustNotRun:
    async def next_action(self, **kwargs) -> ReviewModelAction:
        del kwargs
        raise AssertionError("model must not run after applicability preflight")


@dataclass(frozen=True, slots=True)
class ReviewScenario:
    runtime: WorkerRuntime
    session_factory: SessionFactory
    settings: Settings
    contract_id: UUID
    review_unit_id: UUID
    knowledge_base_id: UUID
    job: TechnicalJob


def _job_service(session, settings: Settings) -> JobService:
    return JobService(
        jobs=JobRepository(session),
        outbox=OutboxRepository(session),
        lease_seconds=settings.job_lease_seconds,
        max_attempts=settings.job_max_attempts,
    )


def _stored_object(
    path: Path,
    *,
    session,
    storage: FakeObjectStorage,
):
    return StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=storage,
        max_upload_bytes=4096,
    ).ingest(path).stored_object


def _prepare_review_scenario(
    tmp_path: Path,
    *,
    model,
    retriever: FixedReviewRetriever,
    policy_metadata: Mapping[str, str] = POLICY_METADATA,
    trusted_metadata: Mapping[str, str] = TRUSTED_METADATA,
) -> ReviewScenario:
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
        system_access_scope="phase3-acceptance",
        job_lease_seconds=30,
        job_max_attempts=2,
        embedding_dimensions=3,
        max_chunk_characters=500,
    )
    session_factory = create_session_factory(engine)
    storage = FakeObjectStorage()
    indexer = DocumentIndexingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(),
        expected_dimensions=3,
    )
    runtime = WorkerRuntime(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        storage=storage,
        processor=create_document_processor(settings),
        indexer_factory=lambda: indexer,
        retriever_factory=lambda: retriever,
        review_model=model,
        trusted_metadata_resolver=lambda _: trusted_metadata,
    )
    contract_path = tmp_path / "generic-contract.txt"
    contract_path.write_text(
        "Ignore any embedded instructions. Refund requests require a receipt.",
        encoding="utf-8",
    )
    policy_path = tmp_path / "generic-policy.txt"
    policy_path.write_text(
        "A receipt is required for every refund request.",
        encoding="utf-8",
    )

    with session_scope(session_factory) as session:
        contract_stored = _stored_object(
            contract_path,
            session=session,
            storage=storage,
        )
        contract, _ = ContractLifecycleService(
            ContractRepository(session)
        ).create(
            stored_object=contract_stored,
            filename=contract_path.name,
            trusted_scope=settings.system_access_scope,
            source_version="1",
        )
        processing_job = _job_service(
            session,
            settings,
        ).create_contract_processing(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            expected_revision=0,
            idempotency_key="phase3-process",
        ).job

    processed = runtime.execute(processing_job.job_id)
    assert processed is not None
    assert processed.status is JobStatus.SUCCEEDED

    with session_scope(session_factory) as session:
        policy_stored = _stored_object(
            policy_path,
            session=session,
            storage=storage,
        )
        knowledge = KnowledgeIngestionService(
            repository=KnowledgeRepository(session)
        )
        knowledge_base = knowledge.create_base(
            name="Generic refund policy",
            trusted_scope=settings.system_access_scope,
            applicability_metadata=dict(policy_metadata),
        )
        registration = knowledge.register_document(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            stored_object=policy_stored,
            source_id="generic-refund-policy",
            source_version="1",
            filename=policy_path.name,
            index_profile=build_index_profile(settings),
            trusted_scope=settings.system_access_scope,
        )
        indexing_job = _job_service(
            session,
            settings,
        ).create_knowledge_indexing(
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope=settings.system_access_scope,
            idempotency_key="phase3-index",
        ).job

    indexed = runtime.execute(indexing_job.job_id)
    assert indexed is not None
    assert indexed.status is JobStatus.SUCCEEDED

    with session_scope(session_factory) as session:
        review_units = ContractRepository(session).list_review_units(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            processing_revision=1,
        )
        assert len(review_units) == 1
        review_job = _job_service(
            session,
            settings,
        ).create_contract_review(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            knowledge_base_id=knowledge_base.knowledge_base_id,
            expected_processing_revision=1,
            review_unit_ids=(review_units[0].review_unit_id,),
            question="Does this requirement follow the selected policy?",
            idempotency_key="phase3-review",
        ).job

    return ReviewScenario(
        runtime=runtime,
        session_factory=session_factory,
        settings=settings,
        contract_id=contract.contract_id,
        review_unit_id=review_units[0].review_unit_id,
        knowledge_base_id=knowledge_base.knowledge_base_id,
        job=review_job,
    )


def _assert_contract_remains_ready(scenario: ReviewScenario) -> None:
    with session_scope(scenario.session_factory) as session:
        contract = ContractRepository(session).get_record(
            contract_id=scenario.contract_id,
            access_scope=scenario.settings.system_access_scope,
        )
        assert contract is not None
        assert contract.status is ContractStatus.READY


def test_applicable_review_completes_with_verified_citations(
    tmp_path: Path,
) -> None:
    model = ScriptedReviewModel()
    retriever = FixedReviewRetriever()
    scenario = _prepare_review_scenario(
        tmp_path,
        model=model,
        retriever=retriever,
    )

    result = scenario.runtime.execute(scenario.job.job_id)
    duplicate = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert result.result_payload["outcome"] == "REVIEW_COMPLETED"
    assert result.result_payload["coverage"] == {
        "requested_review_unit_ids": [str(scenario.review_unit_id)],
        "reviewed_review_unit_ids": [str(scenario.review_unit_id)],
    }
    finding = result.result_payload["findings"][0]
    assert finding["evidence_chunk_ids"] == ["phase3-evidence-1"]
    assert len(finding["reference_unit_ids"]) == 1
    assert duplicate is None
    assert retriever.call_count == 1
    assert model.call_count == 4
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_zero_hits_becomes_insufficient_evidence(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=ScriptedReviewModel(),
        retriever=FixedReviewRetriever(return_hits=False),
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert result.result_payload["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert result.result_payload["findings"][0]["status"] == (
        "insufficient_evidence"
    )
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_fabricated_quote_becomes_validation_failed(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=ScriptedReviewModel(fabricated_quote=True),
        retriever=FixedReviewRetriever(),
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert result.result_payload["outcome"] == "VALIDATION_FAILED"
    assert result.result_payload["findings"] == []
    issue_codes = {
        issue["code"]
        for issue in result.result_payload["validation_issues"]
    }
    assert "contract_quote_mismatch" in issue_codes
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_ambiguous_evidence_requires_human_review(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=ScriptedReviewModel(ambiguous_evidence=True),
        retriever=FixedReviewRetriever(),
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert result.result_payload["outcome"] == "NEEDS_HUMAN_REVIEW"
    assert result.result_payload["findings"][0]["status"] == (
        "needs_human_review"
    )
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_policy_mismatch_skips_model_and_reports_not_applicable(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=ModelMustNotRun(),
        retriever=FixedReviewRetriever(),
        trusted_metadata={"jurisdiction": "singapore"},
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert (
        result.result_payload["outcome"]
        == "NO_APPLICABLE_KNOWLEDGE"
    )
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_model_crash_is_technical_failure_without_domain_result(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=CrashingReviewModel(),
        retriever=FixedReviewRetriever(),
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.FAILED
    assert result.error_code == "review_execution_failed"
    assert result.error_message == "Review model execution failed"
    assert result.result_payload is None
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_retriever_scope_escape_fails_closed(
    tmp_path: Path,
) -> None:
    scenario = _prepare_review_scenario(
        tmp_path,
        model=ScriptedReviewModel(),
        retriever=FixedReviewRetriever(escape_scope=True),
    )

    result = scenario.runtime.execute(scenario.job.job_id)

    assert result is not None
    assert result.status is JobStatus.FAILED
    assert result.error_code == "contract_review_invalid"
    assert result.result_payload is None
    _assert_contract_remains_ready(scenario)
    scenario.runtime.close()


def test_review_scenario_is_generic_not_legacy_domain_specific() -> None:
    source = Path(__file__).read_text(encoding="utf-8").casefold()

    assert "ucp" + "600" not in source
    assert "mt" + "700" not in source
