"""Release A knowledge-base registration and indexing endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status
from rag_core.ingestion.errors import DocumentProcessingError
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import RuntimeDependency, SessionDependency
from app.api.errors import ApiError
from app.api.schemas.jobs import JobResponse
from app.api.schemas.knowledge import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDetailResponse,
    KnowledgeBaseResponse,
    KnowledgeDocumentResponse,
    KnowledgeMemberResponse,
    KnowledgeUploadResponse,
)
from app.api.uploads import temporary_upload
from app.documents.ingestion_service import (
    StoredObjectIngestionService,
    UploadTooLargeError,
)
from app.documents.repository import StoredObjectRepository
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService
from app.knowledge.errors import (
    KnowledgeBaseNotFoundError,
    KnowledgeDocumentConflictError,
)
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.models import (
    KnowledgeBaseMembership,
    KnowledgeSourceSnapshot,
    SnapshotStatus,
)
from app.knowledge.repository import KnowledgeRepository, to_base

router = APIRouter(
    prefix="/api/v1/knowledge-bases",
    tags=["knowledge-bases"],
)


def _member_response(
    membership: KnowledgeBaseMembership,
    snapshot: KnowledgeSourceSnapshot,
) -> KnowledgeMemberResponse:
    return KnowledgeMemberResponse(
        membership_id=membership.membership_id,
        membership_status=membership.status,
        membership_version=membership.membership_version,
        snapshot_id=snapshot.snapshot_id,
        snapshot_status=snapshot.status,
        source_id=snapshot.source_id,
        source_version=snapshot.source_version,
        source_checksum_sha256=snapshot.source_checksum_sha256,
        index_profile=snapshot.index_profile,
    )


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_knowledge_base(
    body: KnowledgeBaseCreateRequest,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> KnowledgeBaseResponse:
    try:
        knowledge_base = KnowledgeIngestionService(
            repository=KnowledgeRepository(session)
        ).create_base(
            name=body.name,
            trusted_scope=runtime.settings.system_access_scope,
            applicability_metadata=body.applicability_metadata,
        )
    except IntegrityError as error:
        raise ApiError(
            409,
            "knowledge_base_conflict",
            "A knowledge base with this name already exists",
        ) from error

    return KnowledgeBaseResponse.model_validate(knowledge_base)


@router.get("", response_model=tuple[KnowledgeBaseResponse, ...])
def list_knowledge_bases(
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> tuple[KnowledgeBaseResponse, ...]:
    bases = KnowledgeRepository(session).list_bases(
        runtime.settings.system_access_scope
    )
    return tuple(KnowledgeBaseResponse.model_validate(item) for item in bases)


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=KnowledgeUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def register_knowledge_document(
    knowledge_base_id: UUID,
    runtime: RuntimeDependency,
    session: SessionDependency,
    file: Annotated[UploadFile, File()],
    source_id: Annotated[str, Form(min_length=1, max_length=256)],
    source_version: Annotated[str, Form(min_length=1, max_length=128)],
    idempotency_key: Annotated[str, Form(min_length=1, max_length=255)],
) -> KnowledgeUploadResponse:
    ingestion = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
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
        raise ApiError(413, "upload_too_large", str(error)) from error
    except DocumentProcessingError as error:
        raise ApiError(422, error.code.value, error.message) from error

    repository = KnowledgeRepository(session)

    try:
        registration = KnowledgeIngestionService(
            repository=repository
        ).register_document(
            knowledge_base_id=knowledge_base_id,
            stored_object=stored,
            source_id=source_id,
            source_version=source_version,
            filename=file.filename or path.name,
            index_profile=runtime.index_profile,
            trusted_scope=runtime.settings.system_access_scope,
        )
    except KnowledgeBaseNotFoundError as error:
        raise ApiError(404, "knowledge_base_not_found", str(error)) from error
    except KnowledgeDocumentConflictError as error:
        raise ApiError(409, "knowledge_document_conflict", str(error)) from error

    job_response: JobResponse | None = None

    if registration.snapshot.status is not SnapshotStatus.READY:
        job = JobService(
            jobs=JobRepository(session),
            outbox=OutboxRepository(session),
            lease_seconds=runtime.settings.job_lease_seconds,
            max_attempts=runtime.settings.job_max_attempts,
        ).create_knowledge_indexing(
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope=runtime.settings.system_access_scope,
            idempotency_key=idempotency_key,
        ).job
        job_response = JobResponse.model_validate(job)

    return KnowledgeUploadResponse(
        document=KnowledgeDocumentResponse.model_validate(
            registration.document
        ),
        member=_member_response(
            registration.membership,
            registration.snapshot,
        ),
        job=job_response,
    )


@router.get(
    "/{knowledge_base_id}",
    response_model=KnowledgeBaseDetailResponse,
)
def get_knowledge_base(
    knowledge_base_id: UUID,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> KnowledgeBaseDetailResponse:
    repository = KnowledgeRepository(session)
    record = repository.get_base_record(
        knowledge_base_id=knowledge_base_id,
        access_scope=runtime.settings.system_access_scope,
    )

    if record is None:
        raise ApiError(
            404,
            "knowledge_base_not_found",
            "Knowledge base was not found",
        )

    knowledge_base = to_base(record)
    members = repository.list_memberships(
        knowledge_base_id=knowledge_base_id,
        access_scope=runtime.settings.system_access_scope,
    )
    return KnowledgeBaseDetailResponse(
        **KnowledgeBaseResponse.model_validate(knowledge_base).model_dump(),
        members=tuple(
            _member_response(membership, snapshot)
            for membership, snapshot in members
        ),
    )
