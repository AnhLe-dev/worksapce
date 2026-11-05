"""Tests for generic contract identity and provenance-preserving processing."""

from pathlib import Path

import pytest
from rag_core.chunking import BlockChunkingStrategy
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.models import ContractDocument, ContractStatus
from app.contracts.processing_service import (
    ContractLifecycleService,
    ContractNotFoundError,
    ContractProcessingService,
)
from app.contracts.repository import ContractRepository
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.models import StoredObject
from app.documents.repository import StoredObjectRepository
from tests.fakes import FakeObjectStorage


def make_processor() -> DocumentProcessor:
    return DocumentProcessor(
        extractors=(TextDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=BlockChunkingStrategy(max_characters=500),
    )


def create_contract(
    session: Session,
    tmp_path: Path,
) -> tuple[Path, ContractRepository, ContractDocument, StoredObject]:
    path = tmp_path / "contract.txt"
    path.write_text(
        "Delivery requires a receipt.\n\nPayment is due in thirty days.",
        encoding="utf-8",
    )
    stored = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=FakeObjectStorage(),
        max_upload_bytes=4096,
    ).ingest(path).stored_object
    repository = ContractRepository(session)
    contract, reused = ContractLifecycleService(repository).create(
        stored_object=stored,
        filename=path.name,
        trusted_scope="system-a",
        source_version="1",
    )
    assert reused is False
    return path, repository, contract, stored


def test_contract_create_is_idempotent_but_has_distinct_domain_id(
    session: Session,
    tmp_path: Path,
) -> None:
    path, repository, contract, stored = create_contract(session, tmp_path)

    existing, reused = ContractLifecycleService(repository).create(
        stored_object=stored,
        filename=path.name,
        trusted_scope="system-a",
        source_version="1",
    )

    assert reused is True
    assert existing.contract_id == contract.contract_id
    assert existing.contract_id != existing.stored_object_id


def test_database_rejects_non_contract_role(
    session: Session,
    tmp_path: Path,
) -> None:
    _, repository, contract, _ = create_contract(session, tmp_path)
    record = repository.get_record(
        contract_id=contract.contract_id,
        access_scope="system-a",
    )
    assert record is not None
    record.role = "knowledge"

    with pytest.raises(IntegrityError):
        session.flush()


def test_processing_creates_block_units_with_exact_text_offsets(
    session: Session,
    tmp_path: Path,
) -> None:
    path, repository, contract, _ = create_contract(session, tmp_path)
    result = ContractProcessingService(
        repository=repository,
        processor=make_processor(),
    ).process(
        contract_id=contract.contract_id,
        trusted_scope="system-a",
        expected_revision=0,
        file_path=path,
    )

    assert result.contract.status is ContractStatus.READY
    assert result.contract.processing_revision == 1
    assert len(result.review_units) == 2
    assert [unit.order_index for unit in result.review_units] == [0, 1]
    assert result.review_units[0].source_spans[0].char_start == 0
    assert result.review_units[0].source_spans[0].bbox is None
    assert all(unit.page_start is None for unit in result.review_units)


def test_processing_fails_closed_across_scope(
    session: Session,
    tmp_path: Path,
) -> None:
    path, repository, contract, _ = create_contract(session, tmp_path)
    service = ContractProcessingService(
        repository=repository,
        processor=make_processor(),
    )

    with pytest.raises(ContractNotFoundError):
        service.process(
            contract_id=contract.contract_id,
            trusted_scope="system-b",
            expected_revision=0,
            file_path=path,
        )
