"""Release A contract lifecycle endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status
from rag_core.ingestion.errors import DocumentProcessingError

from app.api.dependencies import RuntimeDependency, SessionDependency
from app.api.errors import ApiError
from app.api.schemas.contracts import (
    ContractResponse,
    ProcessingJobRequest,
    ReviewUnitResponse,
)
from app.api.schemas.jobs import JobResponse
from app.api.uploads import temporary_upload
from app.contracts.processing_service import ContractLifecycleService
from app.contracts.repository import ContractRepository, to_contract
from app.documents.ingestion_service import (
    StoredObjectIngestionService,
    UploadTooLargeError,
)
from app.documents.repository import StoredObjectRepository
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService, JobTargetNotFoundError

router = APIRouter(prefix="/api/v1/contracts", tags=["contracts"])


def _job_service(session: SessionDependency, runtime: RuntimeDependency) -> JobService:
    return JobService(
        jobs=JobRepository(session),
        outbox=OutboxRepository(session),
        lease_seconds=runtime.settings.job_lease_seconds,
        max_attempts=runtime.settings.job_max_attempts,
    )


@router.post(
    "",
    response_model=ContractResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_contract(
    runtime: RuntimeDependency,
    session: SessionDependency,
    file: Annotated[UploadFile, File()],
    source_version: Annotated[
        str,
        Form(min_length=1, max_length=128),
    ] = "1",
) -> ContractResponse:
    repository = StoredObjectRepository(session)
    ingestion = StoredObjectIngestionService(
        repository=repository,
        storage=runtime.storage,
        max_upload_bytes=runtime.settings.max_document_upload_bytes,
    )

    try:
        async with temporary_upload(
            file,
            max_bytes=runtime.settings.max_document_upload_bytes,
        ) as path:
            stored = ingestion.ingest(path).stored_object
    except UploadTooLargeError as error:
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "upload_too_large",
            str(error),
        ) from error
    except DocumentProcessingError as error:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            error.code.value,
            error.message,
        ) from error

    contract, _ = ContractLifecycleService(
        ContractRepository(session)
    ).create(
        stored_object=stored,
        filename=file.filename or path.name,
        trusted_scope=runtime.settings.system_access_scope,
        source_version=source_version,
    )
    return ContractResponse.model_validate(contract)


@router.get("/{contract_id}", response_model=ContractResponse)
def get_contract(
    contract_id: UUID,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> ContractResponse:
    record = ContractRepository(session).get_record(
        contract_id=contract_id,
        access_scope=runtime.settings.system_access_scope,
    )

    if record is None:
        raise ApiError(404, "contract_not_found", "Contract was not found")

    return ContractResponse.model_validate(to_contract(record))


@router.post(
    "/{contract_id}/processing-jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_processing_job(
    contract_id: UUID,
    body: ProcessingJobRequest,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> JobResponse:
    try:
        result = _job_service(session, runtime).create_contract_processing(
            contract_id=contract_id,
            access_scope=runtime.settings.system_access_scope,
            expected_revision=body.expected_revision,
            idempotency_key=body.idempotency_key,
        )
    except JobTargetNotFoundError as error:
        raise ApiError(404, "contract_not_found", str(error)) from error

    return JobResponse.model_validate(result.job)


@router.get(
    "/{contract_id}/review-units",
    response_model=tuple[ReviewUnitResponse, ...],
)
def list_review_units(
    contract_id: UUID,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> tuple[ReviewUnitResponse, ...]:
    repository = ContractRepository(session)
    record = repository.get_record(
        contract_id=contract_id,
        access_scope=runtime.settings.system_access_scope,
    )

    if record is None:
        raise ApiError(404, "contract_not_found", "Contract was not found")

    units = repository.list_review_units(
        contract_id=contract_id,
        access_scope=runtime.settings.system_access_scope,
        processing_revision=record.processing_revision,
    )
    return tuple(ReviewUnitResponse.model_validate(unit) for unit in units)
