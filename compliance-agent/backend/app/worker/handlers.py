"""Handlers for processing, indexing, and bounded contract review."""

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from compliance_agent import (
    AgentExecutionError,
    AgentLimits,
    AgentPortError,
    BoundedReviewAgent,
    ReviewModel,
    ReviewRequest,
    ReviewResult,
    ReviewService,
    ReviewToolError,
    ReviewToolset,
    TrustedReviewClaims,
)
from httpx import TransportError
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from rag_core.indexing.errors import IndexingError
from rag_core.indexing.service import DocumentIndexingService
from rag_core.ingestion.errors import DocumentProcessingError
from rag_core.ingestion.models import DocumentFormat
from rag_core.ingestion.service import DocumentProcessor
from rag_core.retrieval.models import KnowledgeHit, RetrievalRequest
from rag_core.retrieval.ports import Retriever
from sqlalchemy.orm import Session

from app.contracts.models import ContractDocument, ContractStatus
from app.contracts.processing_service import (
    ContractNotFoundError,
    ContractProcessingService,
    ContractRevisionConflictError,
    ContractSourceMismatchError,
)
from app.contracts.repository import ContractRepository, to_contract
from app.documents.storage import (
    ObjectStorage,
    ObjectStorageUnavailableError,
    StoredObjectIntegrityError,
)
from app.jobs.models import JobClaim, JobKind, TechnicalJob
from app.knowledge.errors import KnowledgeError
from app.knowledge.indexing_service import KnowledgeIndexingService
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.reference_service import KnowledgeReferenceService
from app.knowledge.registry import KnowledgeRegistry
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.retrieval_service import KnowledgeRetrievalService
from app.review.capabilities import (
    BackendApplicabilityProvider,
    BoundReferenceUnitProvider,
    BoundReviewUnitProvider,
    BoundSelectedKnowledgeRetriever,
)
from app.worker.dispatcher import PermanentJobError, RetryableJobError

IndexerFactory = Callable[[], DocumentIndexingService]
RetrieverFactory = Callable[[], Retriever]
TrustedMetadataResolver = Callable[
    [ContractDocument],
    Mapping[str, str],
]
JobTarget = JobClaim | TechnicalJob


class _LazyRetriever:
    """Delay provider construction until applicability permits search."""

    def __init__(self, factory: RetrieverFactory) -> None:
        self._factory = factory

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        return self._factory().retrieve(request)

_SUFFIXES = {
    DocumentFormat.PDF: ".pdf",
    DocumentFormat.TXT: ".txt",
}
_RETRYABLE_PROVIDER_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    ResponseHandlingException,
    TransportError,
)
_REJECTED_PROVIDER_ERRORS = (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)


class ProviderConfigurationError(RuntimeError):
    """Raised when a knowledge provider is not configured on the worker."""


def _empty_trusted_contract_metadata(
    contract: ContractDocument,
) -> Mapping[str, str]:
    """Fail closed until a domain adapter persists verified claims."""

    del contract
    return {}


def _uuid_payload(value: object, *, field_name: str) -> UUID:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a UUID string")

    try:
        return UUID(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a UUID string") from error


def _review_unit_ids_payload(value: object) -> tuple[UUID, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("review_unit_ids must be a non-empty list")

    unit_ids = tuple(
        _uuid_payload(item, field_name="review_unit_ids")
        for item in value
    )

    if len(unit_ids) > 100 or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("review_unit_ids must be unique and at most 100")

    return unit_ids


class ContractProcessingHandler:
    def __init__(
        self,
        *,
        repository: ContractRepository,
        processor: DocumentProcessor,
        storage: ObjectStorage,
    ) -> None:
        self._repository = repository
        self._processor = processor
        self._storage = storage

    def __call__(self, claim: JobClaim) -> dict[str, object]:
        if claim.contract_id is None:
            raise PermanentJobError(
                code="invalid_job_target",
                message="Contract processing job has no contract target",
            )

        expected_revision = claim.request_payload.get("expected_revision")

        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision,
            int,
        ):
            raise PermanentJobError(
                code="invalid_job_payload",
                message="Contract job expected_revision is invalid",
            )

        record = self._repository.get_record(
            contract_id=claim.contract_id,
            access_scope=claim.access_scope,
        )

        if record is None:
            raise PermanentJobError(
                code="contract_not_found",
                message="Contract document was not found",
            )

        suffix = _SUFFIXES[record.stored_object.document_format]

        try:
            with TemporaryDirectory(prefix="contract-job-") as directory:
                path = Path(directory) / f"source{suffix}"
                self._storage.download_file(
                    object_key=record.stored_object.object_key,
                    destination_path=path,
                    expected_checksum=(
                        record.stored_object.checksum_sha256
                    ),
                )
                outcome = ContractProcessingService(
                    repository=self._repository,
                    processor=self._processor,
                ).process(
                    contract_id=claim.contract_id,
                    trusted_scope=claim.access_scope,
                    expected_revision=expected_revision,
                    file_path=path,
                )
        except ObjectStorageUnavailableError as error:
            raise RetryableJobError(
                "Contract source storage is unavailable"
            ) from error
        except DocumentProcessingError as error:
            raise PermanentJobError(
                code=error.code.value,
                message=error.message,
            ) from error
        except StoredObjectIntegrityError as error:
            raise PermanentJobError(
                code="stored_object_integrity_failed",
                message=str(error),
            ) from error
        except (
            ContractNotFoundError,
            ContractRevisionConflictError,
            ContractSourceMismatchError,
            ValueError,
        ) as error:
            raise PermanentJobError(
                code="contract_processing_invalid",
                message=str(error),
            ) from error

        return {
            "contract_id": str(outcome.contract.contract_id),
            "processing_revision": outcome.contract.processing_revision,
            "review_unit_count": len(outcome.review_units),
        }


class KnowledgeIndexingHandler:
    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        processor: DocumentProcessor,
        storage: ObjectStorage,
        indexer_factory: IndexerFactory,
    ) -> None:
        self._repository = repository
        self._processor = processor
        self._storage = storage
        self._indexer_factory = indexer_factory

    def __call__(self, claim: JobClaim) -> dict[str, object]:
        if claim.snapshot_id is None:
            raise PermanentJobError(
                code="invalid_job_target",
                message="Knowledge indexing job has no snapshot target",
            )

        context = self._repository.get_snapshot_context(
            snapshot_id=claim.snapshot_id,
            access_scope=claim.access_scope,
        )

        if context is None:
            raise PermanentJobError(
                code="knowledge_snapshot_not_found",
                message="Knowledge snapshot was not found",
            )

        _, _, stored_object = context
        suffix = _SUFFIXES[stored_object.document_format]

        try:
            with TemporaryDirectory(prefix="knowledge-job-") as directory:
                path = Path(directory) / f"source{suffix}"
                self._storage.download_file(
                    object_key=stored_object.object_key,
                    destination_path=path,
                    expected_checksum=stored_object.checksum_sha256,
                )
                ingestion = KnowledgeIngestionService(
                    repository=self._repository,
                    processor=self._processor,
                )
                outcome = KnowledgeIndexingService(
                    repository=self._repository,
                    ingestion=ingestion,
                    indexer=self._indexer_factory(),
                ).index(
                    snapshot_id=claim.snapshot_id,
                    trusted_scope=claim.access_scope,
                    file_path=path,
                )
        except ObjectStorageUnavailableError as error:
            raise RetryableJobError(
                "Knowledge source storage is unavailable"
            ) from error
        except _RETRYABLE_PROVIDER_ERRORS as error:
            raise RetryableJobError(
                "Knowledge index provider is unavailable"
            ) from error
        except _REJECTED_PROVIDER_ERRORS as error:
            raise PermanentJobError(
                code="index_provider_rejected",
                message="Knowledge index provider rejected the request",
            ) from error
        except UnexpectedResponse as error:
            if error.status_code >= 500:
                raise RetryableJobError(
                    "Qdrant index provider is unavailable"
                ) from error

            raise PermanentJobError(
                code="index_provider_rejected",
                message="Qdrant index provider rejected the request",
            ) from error
        except ProviderConfigurationError as error:
            raise PermanentJobError(
                code="index_provider_not_configured",
                message=str(error),
            ) from error
        except DocumentProcessingError as error:
            raise PermanentJobError(
                code=error.code.value,
                message=error.message,
            ) from error
        except StoredObjectIntegrityError as error:
            raise PermanentJobError(
                code="stored_object_integrity_failed",
                message=str(error),
            ) from error
        except (
            IndexingError,
            KnowledgeError,
            ValueError,
        ) as error:
            raise PermanentJobError(
                code="knowledge_indexing_invalid",
                message=str(error),
            ) from error

        return {
            "snapshot_id": str(outcome.snapshot.snapshot_id),
            "source_id": outcome.snapshot.source_id,
            "source_version": outcome.snapshot.source_version,
            "chunk_count": outcome.snapshot.chunk_count or 0,
            "reused_existing": outcome.reused_existing,
        }


class ContractReviewHandler:
    """Rebuild trusted capabilities from DB before invoking the agent."""

    def __init__(
        self,
        *,
        contracts: ContractRepository,
        knowledge: KnowledgeRepository,
        retriever_factory: RetrieverFactory,
        model: ReviewModel,
        limits: AgentLimits,
        retrieval_limit: int,
        trusted_metadata_resolver: TrustedMetadataResolver | None = None,
    ) -> None:
        self._contracts = contracts
        self._knowledge = knowledge
        self._retriever = _LazyRetriever(retriever_factory)
        self._model = model
        self._limits = limits
        self._retrieval_limit = retrieval_limit
        self._trusted_metadata_resolver = (
            trusted_metadata_resolver
            or _empty_trusted_contract_metadata
        )

    def __call__(self, claim: JobClaim) -> dict[str, object]:
        if claim.contract_id is None:
            raise PermanentJobError(
                code="invalid_job_target",
                message="Contract review job has no contract target",
            )

        try:
            result = self._review(claim)
        except _RETRYABLE_PROVIDER_ERRORS as error:
            raise RetryableJobError(
                "Review dependency is unavailable"
            ) from error
        except _REJECTED_PROVIDER_ERRORS as error:
            raise PermanentJobError(
                code="review_provider_rejected",
                message="Review provider rejected the request",
            ) from error
        except UnexpectedResponse as error:
            if error.status_code >= 500:
                raise RetryableJobError(
                    "Review vector provider is unavailable"
                ) from error

            raise PermanentJobError(
                code="review_provider_rejected",
                message="Review vector provider rejected the request",
            ) from error
        except ProviderConfigurationError as error:
            raise PermanentJobError(
                code="review_provider_not_configured",
                message=str(error),
            ) from error
        except (
            AgentExecutionError,
            AgentPortError,
            KnowledgeError,
            ReviewToolError,
            TypeError,
            ValueError,
        ) as error:
            raise PermanentJobError(
                code="contract_review_invalid",
                message=str(error),
            ) from error
        except RuntimeError as error:
            raise PermanentJobError(
                code="review_execution_failed",
                message="Review model execution failed",
            ) from error

        return result.model_dump(mode="json")

    def _review(self, claim: JobClaim) -> ReviewResult:
        assert claim.contract_id is not None
        knowledge_base_id = _uuid_payload(
            claim.request_payload.get("knowledge_base_id"),
            field_name="knowledge_base_id",
        )
        revision = claim.request_payload.get(
            "expected_processing_revision"
        )

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise ValueError(
                "expected_processing_revision must be a positive integer"
            )

        review_unit_ids = _review_unit_ids_payload(
            claim.request_payload.get("review_unit_ids")
        )
        question = claim.request_payload.get("question")

        if question is not None and not isinstance(question, str):
            raise ValueError("question must be text or null")

        contract_record = self._contracts.get_record(
            contract_id=claim.contract_id,
            access_scope=claim.access_scope,
        )

        if contract_record is None:
            raise ValueError("Contract was not found in the trusted scope")

        if contract_record.status is not ContractStatus.READY:
            raise ValueError("Contract is not READY for review")

        if contract_record.processing_revision != revision:
            raise ValueError("Contract processing revision changed")

        stored_units = self._contracts.list_review_units(
            contract_id=claim.contract_id,
            access_scope=claim.access_scope,
            processing_revision=revision,
        )

        if not set(review_unit_ids).issubset(
            unit.review_unit_id for unit in stored_units
        ):
            raise ValueError(
                "A review unit is absent from the trusted contract revision"
            )

        registry = KnowledgeRegistry(self._knowledge)
        resolved = registry.resolve(
            knowledge_base_id=knowledge_base_id,
            trusted_scope=claim.access_scope,
        )
        snapshot_ids = tuple(
            snapshot.snapshot_id for snapshot in resolved.snapshots
        )
        request = ReviewRequest(
            contract_id=claim.contract_id,
            review_unit_ids=review_unit_ids,
            knowledge_base_id=knowledge_base_id,
            question=question,
            trusted_claims=TrustedReviewClaims(
                contract_id=claim.contract_id,
                knowledge_base_id=knowledge_base_id,
                access_scope=claim.access_scope,
                contract_processing_revision=revision,
                authorized_review_unit_ids=review_unit_ids,
                authorized_snapshot_ids=snapshot_ids,
            ),
        )
        reference_service = KnowledgeReferenceService(self._knowledge)
        retrieval_service = KnowledgeRetrievalService(
            registry=registry,
            retriever=self._retriever,
        )
        tools = ReviewToolset(
            request=request,
            review_units=BoundReviewUnitProvider(
                repository=self._contracts,
                contract_id=claim.contract_id,
                access_scope=claim.access_scope,
                processing_revision=revision,
                authorized_review_unit_ids=review_unit_ids,
            ),
            knowledge=BoundSelectedKnowledgeRetriever(
                knowledge_base_id=knowledge_base_id,
                access_scope=claim.access_scope,
                authorized_review_unit_ids=review_unit_ids,
                resolved=resolved,
                retrieval=retrieval_service,
                references=reference_service,
            ),
            references=BoundReferenceUnitProvider(
                service=reference_service,
                snapshots=resolved.snapshots,
                access_scope=claim.access_scope,
            ),
            retrieval_limit=self._retrieval_limit,
        )
        contract = to_contract(contract_record)
        return asyncio.run(
            ReviewService(
                applicability=BackendApplicabilityProvider(
                    resolved=resolved,
                    trusted_contract_metadata=(
                        self._trusted_metadata_resolver(contract)
                    ),
                ),
                agent=BoundedReviewAgent(
                    model=self._model,
                    limits=self._limits,
                ),
            ).review(request=request, tools=tools)
        )


class DomainFailureRecorder:
    """Keep target-domain failure consistent with terminal job failure."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def __call__(
        self,
        target: JobTarget,
        error_code: str,
        error_message: str,
    ) -> None:
        if target.kind is JobKind.CONTRACT_REVIEW:
            # Review failures belong to technical job state only.
            return

        if target.kind is JobKind.CONTRACT_PROCESSING:
            if target.contract_id is None:
                raise ValueError("Contract job has no contract target")

            repository = ContractRepository(self._session)
            record = repository.get_record(
                contract_id=target.contract_id,
                access_scope=target.access_scope,
                for_update=True,
            )

            if record is None:
                return

            repository.set_status(
                record,
                status=ContractStatus.FAILED,
                error_code=error_code,
                error_message=error_message,
            )
            return

        if target.snapshot_id is None:
            raise ValueError("Knowledge job has no snapshot target")

        KnowledgeRepository(self._session).fail_indexing(
            snapshot_id=target.snapshot_id,
            access_scope=target.access_scope,
            error_code=error_code,
            error_message=error_message,
        )
