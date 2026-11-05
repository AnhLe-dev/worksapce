"""Generic contract creation and processing orchestration."""

from pathlib import Path
from uuid import UUID

from rag_core.ingestion.models import DocumentInput, DocumentRole
from rag_core.ingestion.service import DocumentProcessor

from app.contracts.clause_segmenter import ClauseSegmenter
from app.contracts.db_models import ContractDocumentRecord
from app.contracts.models import (
    ContractDocument,
    ContractProcessingResult,
    ContractStatus,
)
from app.contracts.repository import ContractRepository, to_contract
from app.documents.models import StoredObject, StoredObjectStatus


class ContractNotFoundError(LookupError):
    """Raised when a contract is absent from the trusted scope."""


class ContractRevisionConflictError(RuntimeError):
    """Raised when a stale job tries to process another revision."""


class ContractSourceMismatchError(RuntimeError):
    """Raised when downloaded bytes do not match StoredObject."""


class ContractLifecycleService:
    """Create distinct, idempotent contract-domain records."""

    def __init__(self, repository: ContractRepository) -> None:
        self._repository = repository

    def create(
        self,
        *,
        stored_object: StoredObject,
        filename: str,
        trusted_scope: str,
        source_version: str,
    ) -> tuple[ContractDocument, bool]:
        normalized_scope = trusted_scope.strip()
        normalized_version = source_version.strip()

        if not normalized_scope or not normalized_version:
            raise ValueError("Trusted scope and source version are required")

        normalized_filename = filename.strip()

        if not normalized_filename:
            raise ValueError("Filename is required")

        if stored_object.status is not StoredObjectStatus.VERIFIED:
            raise ValueError("Contract requires a verified stored object")

        existing = self._repository.get_by_identity(
            stored_object_id=stored_object.stored_object_id,
            access_scope=normalized_scope,
            source_version=normalized_version,
        )

        if existing is not None:
            return existing, True

        contract = self._repository.add(
            ContractDocumentRecord(
                role="contract",
                stored_object_id=stored_object.stored_object_id,
                filename=normalized_filename,
                access_scope=normalized_scope,
                source_version=normalized_version,
                status=ContractStatus.UPLOADED,
            )
        )
        return contract, False


class ContractProcessingService:
    """Run generic core processing and persist traceable review units."""

    def __init__(
        self,
        *,
        repository: ContractRepository,
        processor: DocumentProcessor,
        segmenter: ClauseSegmenter | None = None,
    ) -> None:
        self._repository = repository
        self._processor = processor
        self._segmenter = segmenter or ClauseSegmenter()

    def process(
        self,
        *,
        contract_id: UUID,
        trusted_scope: str,
        expected_revision: int,
        file_path: Path,
    ) -> ContractProcessingResult:
        record = self._repository.get_record(
            contract_id=contract_id,
            access_scope=trusted_scope,
            for_update=True,
        )

        if record is None:
            raise ContractNotFoundError("Contract document was not found")

        if record.processing_revision != expected_revision:
            raise ContractRevisionConflictError(
                "Contract processing revision does not match"
            )

        self._repository.set_status(
            record,
            status=ContractStatus.PROCESSING,
        )
        next_revision = expected_revision + 1
        processed = self._processor.process(
            DocumentInput(
                path=file_path,
                source_id=str(contract_id),
                source_version=record.source_version,
                access_scope=trusted_scope,
                role=DocumentRole.CONTRACT,
            )
        )

        if (
            processed.source.checksum_sha256
            != record.stored_object.checksum_sha256
        ):
            raise ContractSourceMismatchError(
                "Processed bytes do not match the stored object"
            )

        review_units = self._segmenter.segment(
            processed,
            contract_id=contract_id,
            processing_revision=next_revision,
        )
        self._repository.replace_review_units(
            contract=record,
            review_units=review_units,
        )
        record.processing_revision = next_revision
        record.status = ContractStatus.READY
        record.page_count = (
            len(processed.extracted.page_geometries) or None
        )
        record.error_code = None
        record.error_message = None
        self._repository.set_status(record, status=ContractStatus.READY)

        return ContractProcessingResult(
            contract=to_contract(record),
            review_units=review_units,
        )

    def mark_failed(
        self,
        *,
        contract_id: UUID,
        trusted_scope: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Persist a typed failure in a transaction separate from rollback."""

        record = self._repository.get_record(
            contract_id=contract_id,
            access_scope=trusted_scope,
            for_update=True,
        )

        if record is None:
            raise ContractNotFoundError("Contract document was not found")

        self._repository.set_status(
            record,
            status=ContractStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )
