"""Process-local worker composition with crash-recoverable transactions."""

from collections.abc import Callable
from threading import Lock
from typing import NoReturn
from uuid import UUID

from compliance_agent import (
    AgentLimits,
    ReviewModel,
    ReviewRequest,
    ReviewToolObservation,
)
from openai import OpenAI
from qdrant_client import QdrantClient
from rag_core.adapters.openai_embeddings import OpenAIEmbeddingProvider
from rag_core.adapters.qdrant import (
    QdrantVectorIndex,
    ensure_qdrant_collection,
)
from rag_core.chunking import BlockChunkingStrategy
from rag_core.indexing.service import DocumentIndexingService
from rag_core.ingestion.extractors.pdf import PdfDocumentExtractor
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor
from rag_core.retrieval.dense import DenseSourceRetriever
from rag_core.retrieval.ports import Retriever
from rag_core.retrieval.service import RetrievalService
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.contracts.repository import ContractRepository
from app.core.config import Settings, get_settings
from app.db.session import (
    SessionFactory,
    create_database_engine,
    create_session_factory,
    session_scope,
)
from app.documents.storage import (
    ObjectStorage,
    create_object_storage,
    require_secret,
)
from app.jobs.models import JobKind, TechnicalJob
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService
from app.knowledge.index_profile import build_index_profile
from app.knowledge.repository import KnowledgeRepository
from app.worker.dispatcher import WorkerDispatcher
from app.worker.handlers import (
    ContractProcessingHandler,
    ContractReviewHandler,
    DomainFailureRecorder,
    IndexerFactory,
    KnowledgeIndexingHandler,
    ProviderConfigurationError,
    TrustedMetadataResolver,
)


class _UnavailableReviewModel:
    """Fail explicitly when no domain model adapter was injected."""

    async def next_action(
        self,
        *,
        system_prompt: str,
        request: ReviewRequest,
        observations: tuple[ReviewToolObservation, ...],
    ) -> NoReturn:
        del system_prompt, request, observations
        raise ProviderConfigurationError(
            "A ReviewModel adapter is not configured"
        )


def create_document_processor(settings: Settings) -> DocumentProcessor:
    return DocumentProcessor(
        extractors=(PdfDocumentExtractor(), TextDocumentExtractor()),
        parser=FlatStructureParser(),
        chunking_strategy=BlockChunkingStrategy(
            max_characters=settings.max_chunk_characters
        ),
    )


class WorkerRuntime:
    """Own worker resources and commit a claim before external work."""

    def __init__(
        self,
        *,
        settings: Settings,
        engine: Engine,
        session_factory: SessionFactory,
        storage: ObjectStorage,
        processor: DocumentProcessor,
        indexer_factory: IndexerFactory | None = None,
        retriever_factory: Callable[[], Retriever] | None = None,
        review_model: ReviewModel | None = None,
        trusted_metadata_resolver: TrustedMetadataResolver | None = None,
    ) -> None:
        self.settings = settings
        self._engine = engine
        self._session_factory = session_factory
        self._storage = storage
        self._processor = processor
        self._injected_indexer_factory = indexer_factory
        self._injected_retriever_factory = retriever_factory
        self._review_model = review_model or _UnavailableReviewModel()
        self._trusted_metadata_resolver = trusted_metadata_resolver
        self._indexer: DocumentIndexingService | None = None
        self._retriever: Retriever | None = None
        self._openai_client: OpenAI | None = None
        self._qdrant_client: QdrantClient | None = None
        self._provider_lock = Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _job_service(self, session: Session) -> JobService:
        return JobService(
            jobs=JobRepository(session),
            outbox=OutboxRepository(session),
            lease_seconds=self.settings.job_lease_seconds,
            max_attempts=self.settings.job_max_attempts,
        )

    def _create_provider_resources(
        self,
    ) -> tuple[DocumentIndexingService, Retriever]:
        if self.settings.openai_api_key is None:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is not configured"
            )

        if self.settings.qdrant_cloud_url is None:
            raise ProviderConfigurationError(
                "QDRANT_CLOUD_URL is not configured"
            )

        try:
            openai_api_key = require_secret(
                self.settings.openai_api_key,
                "OPENAI_API_KEY",
            )
        except ValueError as error:
            raise ProviderConfigurationError(str(error)) from error

        openai_client = OpenAI(
            api_key=openai_api_key,
            timeout=self.settings.dependency_timeout_seconds,
        )
        qdrant_api_key = (
            None
            if self.settings.qdrant_cloud_api_key is None
            else self.settings.qdrant_cloud_api_key.get_secret_value() or None
        )
        qdrant_client = QdrantClient(
            url=self.settings.qdrant_cloud_url,
            api_key=qdrant_api_key,
            timeout=self.settings.dependency_timeout_seconds,
        )

        provider_ready = False

        try:
            ensure_qdrant_collection(
                qdrant_client,
                collection_name=self.settings.qdrant_collection_name,
                vector_dimensions=self.settings.embedding_dimensions,
            )
            embedding = OpenAIEmbeddingProvider(
                client=openai_client,
                model=self.settings.default_embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
            vector_index = QdrantVectorIndex(
                client=qdrant_client,
                collection_name=self.settings.qdrant_collection_name,
                vector_dimensions=self.settings.embedding_dimensions,
                index_profile=build_index_profile(self.settings),
            )
            indexer = DocumentIndexingService(
                embedding_provider=embedding,
                vector_index=vector_index,
                expected_dimensions=self.settings.embedding_dimensions,
            )
            self._openai_client = openai_client
            self._qdrant_client = qdrant_client
            provider_ready = True
            retriever = RetrievalService(
                source_retriever=DenseSourceRetriever(
                    embedding_provider=embedding,
                    vector_search=vector_index,
                    expected_dimensions=self.settings.embedding_dimensions,
                )
            )
            return indexer, retriever
        finally:
            if not provider_ready:
                openai_client.close()
                qdrant_client.close()

    def _get_indexer(self) -> DocumentIndexingService:
        if self._injected_indexer_factory is not None:
            return self._injected_indexer_factory()

        with self._provider_lock:
            if self._indexer is None:
                self._indexer, self._retriever = (
                    self._create_provider_resources()
                )

            return self._indexer

    def _get_retriever(self) -> Retriever:
        if self._injected_retriever_factory is not None:
            return self._injected_retriever_factory()

        with self._provider_lock:
            if self._retriever is None:
                self._indexer, self._retriever = (
                    self._create_provider_resources()
                )

            return self._retriever

    def execute(self, job_id: UUID) -> TechnicalJob | None:
        if self._closed:
            raise RuntimeError("Worker runtime is closed")

        with session_scope(self._session_factory) as session:
            claim = self._job_service(session).claim(job_id)

        if claim is None:
            return None

        with session_scope(self._session_factory) as session:
            dispatcher = WorkerDispatcher(
                jobs=self._job_service(session),
                handlers={
                    JobKind.CONTRACT_PROCESSING: ContractProcessingHandler(
                        repository=ContractRepository(session),
                        processor=self._processor,
                        storage=self._storage,
                    ),
                    JobKind.CONTRACT_REVIEW: ContractReviewHandler(
                        contracts=ContractRepository(session),
                        knowledge=KnowledgeRepository(session),
                        retriever_factory=self._get_retriever,
                        model=self._review_model,
                        limits=AgentLimits(
                            max_model_calls=(
                                self.settings.review_max_model_calls
                            ),
                            max_total_tool_calls=(
                                self.settings.review_max_total_tool_calls
                            ),
                            max_knowledge_search_calls=(
                                self.settings
                                .review_max_knowledge_search_calls
                            ),
                            max_reference_calls=(
                                self.settings.review_max_reference_calls
                            ),
                            timeout_seconds=(
                                self.settings.review_timeout_seconds
                            ),
                        ),
                        retrieval_limit=(
                            self.settings.review_retrieval_limit
                        ),
                        trusted_metadata_resolver=(
                            self._trusted_metadata_resolver
                        ),
                    ),
                    JobKind.KNOWLEDGE_INDEXING: KnowledgeIndexingHandler(
                        repository=KnowledgeRepository(session),
                        processor=self._processor,
                        storage=self._storage,
                        indexer_factory=self._get_indexer,
                    ),
                },
                failure_handler=DomainFailureRecorder(session),
            )
            return dispatcher.execute_claim(claim)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._openai_client is not None:
            self._openai_client.close()

        if self._qdrant_client is not None:
            self._qdrant_client.close()

        self._engine.dispose()


def create_worker_runtime(
    settings: Settings | None = None,
) -> WorkerRuntime:
    resolved_settings = settings or get_settings()
    engine = create_database_engine(resolved_settings)
    return WorkerRuntime(
        settings=resolved_settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        storage=create_object_storage(resolved_settings),
        processor=create_document_processor(resolved_settings),
    )


_runtime: WorkerRuntime | None = None
_runtime_lock = Lock()


def configure_worker_runtime(runtime: WorkerRuntime) -> None:
    global _runtime

    with _runtime_lock:
        if _runtime is not None and _runtime is not runtime:
            _runtime.close()

        _runtime = runtime


def get_worker_runtime() -> WorkerRuntime:
    global _runtime

    with _runtime_lock:
        if _runtime is None or _runtime.closed:
            _runtime = create_worker_runtime()

        return _runtime


def clear_worker_runtime() -> None:
    global _runtime

    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()

        _runtime = None
