"""Scoped persistence for contracts and review units."""

from uuid import UUID

from rag_core.ingestion.models import SourceSpan
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.contracts.db_models import (
    ContractDocumentRecord,
    ReviewUnitRecord,
)
from app.contracts.models import (
    ContractDocument,
    ContractStatus,
    ReviewUnit,
)


def to_contract(record: ContractDocumentRecord) -> ContractDocument:
    return ContractDocument(
        contract_id=record.contract_id,
        role="contract",
        stored_object_id=record.stored_object_id,
        filename=record.filename,
        access_scope=record.access_scope,
        source_version=record.source_version,
        processing_revision=record.processing_revision,
        status=record.status,
        page_count=record.page_count,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def to_review_unit(record: ReviewUnitRecord) -> ReviewUnit:
    return ReviewUnit(
        review_unit_id=record.review_unit_id,
        contract_id=record.contract_id,
        processing_revision=record.processing_revision,
        order_index=record.order_index,
        text=record.text,
        label=record.label,
        source_spans=tuple(
            SourceSpan.model_validate(span)
            for span in record.source_spans
        ),
        page_start=record.page_start,
        page_end=record.page_end,
    )


class ContractRepository:
    """Require trusted scope on every contract lookup."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_record(
        self,
        *,
        contract_id: UUID,
        access_scope: str,
        for_update: bool = False,
    ) -> ContractDocumentRecord | None:
        statement = select(ContractDocumentRecord).where(
            ContractDocumentRecord.contract_id == contract_id,
            ContractDocumentRecord.access_scope == access_scope,
        )

        if for_update:
            statement = statement.with_for_update()

        return self._session.scalar(statement)

    def get_by_identity(
        self,
        *,
        stored_object_id: UUID,
        access_scope: str,
        source_version: str,
    ) -> ContractDocument | None:
        record = self._session.scalar(
            select(ContractDocumentRecord).where(
                ContractDocumentRecord.stored_object_id == stored_object_id,
                ContractDocumentRecord.access_scope == access_scope,
                ContractDocumentRecord.source_version == source_version,
            )
        )
        return None if record is None else to_contract(record)

    def add(self, record: ContractDocumentRecord) -> ContractDocument:
        self._session.add(record)
        self._session.flush()
        self._session.refresh(record)
        return to_contract(record)

    def replace_review_units(
        self,
        *,
        contract: ContractDocumentRecord,
        review_units: tuple[ReviewUnit, ...],
    ) -> None:
        self._session.execute(
            delete(ReviewUnitRecord).where(
                ReviewUnitRecord.contract_id == contract.contract_id,
                ReviewUnitRecord.processing_revision
                == review_units[0].processing_revision,
            )
        )

        for unit in review_units:
            self._session.add(
                ReviewUnitRecord(
                    review_unit_id=unit.review_unit_id,
                    contract_id=unit.contract_id,
                    processing_revision=unit.processing_revision,
                    order_index=unit.order_index,
                    text=unit.text,
                    label=unit.label,
                    source_spans=[
                        span.model_dump(mode="json")
                        for span in unit.source_spans
                    ],
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                )
            )

        self._session.flush()

    def list_review_units(
        self,
        *,
        contract_id: UUID,
        access_scope: str,
        processing_revision: int,
    ) -> tuple[ReviewUnit, ...]:
        contract = self.get_record(
            contract_id=contract_id,
            access_scope=access_scope,
        )

        if contract is None:
            return ()

        records = self._session.scalars(
            select(ReviewUnitRecord)
            .where(
                ReviewUnitRecord.contract_id == contract_id,
                ReviewUnitRecord.processing_revision == processing_revision,
            )
            .order_by(ReviewUnitRecord.order_index)
        ).all()
        return tuple(to_review_unit(record) for record in records)

    def set_status(
        self,
        record: ContractDocumentRecord,
        *,
        status: ContractStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        record.status = status
        record.error_code = error_code
        record.error_message = error_message
        self._session.flush()
