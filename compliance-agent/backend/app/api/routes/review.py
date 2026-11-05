"""Release B endpoint for durable, bounded domain review jobs."""

from uuid import UUID

from fastapi import APIRouter, status
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import RuntimeDependency, SessionDependency
from app.api.errors import ApiError
from app.api.schemas.jobs import JobResponse
from app.api.schemas.review import ReviewJobRequest
from app.contracts.models import ContractStatus
from app.contracts.repository import ContractRepository
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService, JobTargetNotFoundError
from app.knowledge.errors import (
    KnowledgeBaseNotFoundError,
    KnowledgeSnapshotNotReadyError,
)
from app.knowledge.registry import KnowledgeRegistry
from app.knowledge.repository import KnowledgeRepository

router = APIRouter(prefix="/api/v1/contracts", tags=["review"])


@router.post(
    "/{contract_id}/review-jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_review_job(
    contract_id: UUID,
    body: ReviewJobRequest,
    runtime: RuntimeDependency,
    session: SessionDependency,
) -> JobResponse:
    """Validate current authority, then persist intent for the worker."""

    scope = runtime.settings.system_access_scope
    contracts = ContractRepository(session)
    contract = contracts.get_record(
        contract_id=contract_id,
        access_scope=scope,
    )

    if contract is None:
        raise ApiError(404, "contract_not_found", "Contract was not found")

    if contract.status is not ContractStatus.READY:
        raise ApiError(
            409,
            "contract_not_ready",
            "Contract must be READY before review",
        )

    if contract.processing_revision != body.expected_processing_revision:
        raise ApiError(
            409,
            "contract_revision_conflict",
            "Contract processing revision changed",
        )

    units = contracts.list_review_units(
        contract_id=contract_id,
        access_scope=scope,
        processing_revision=contract.processing_revision,
    )
    available_ids = {unit.review_unit_id for unit in units}

    if not set(body.review_unit_ids).issubset(available_ids):
        raise ApiError(
            422,
            "review_unit_not_found",
            "A requested review unit is not in the current contract revision",
        )

    try:
        KnowledgeRegistry(KnowledgeRepository(session)).resolve(
            knowledge_base_id=body.knowledge_base_id,
            trusted_scope=scope,
        )
    except KnowledgeBaseNotFoundError as error:
        raise ApiError(
            404,
            "knowledge_base_not_found",
            str(error),
        ) from error
    except KnowledgeSnapshotNotReadyError as error:
        raise ApiError(
            409,
            "knowledge_base_not_ready",
            str(error),
        ) from error

    try:
        result = JobService(
            jobs=JobRepository(session),
            outbox=OutboxRepository(session),
            lease_seconds=runtime.settings.job_lease_seconds,
            max_attempts=runtime.settings.job_max_attempts,
        ).create_contract_review(
            contract_id=contract_id,
            access_scope=scope,
            knowledge_base_id=body.knowledge_base_id,
            expected_processing_revision=(
                body.expected_processing_revision
            ),
            review_unit_ids=body.review_unit_ids,
            question=body.question,
            idempotency_key=body.idempotency_key,
        )
    except JobTargetNotFoundError as error:
        raise ApiError(404, "contract_not_found", str(error)) from error
    except IntegrityError as error:
        raise ApiError(
            409,
            "contract_job_active",
            "Another technical job is active for this contract",
        ) from error

    return JobResponse.model_validate(result.job)
