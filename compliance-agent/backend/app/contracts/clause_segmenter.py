"""Generic block fallback for creating review units."""

from uuid import NAMESPACE_URL, UUID, uuid5

from rag_core.ingestion.models import (
    DocumentProcessingResult,
    DocumentRole,
)

from app.contracts.models import ReviewUnit


class ClauseSegmenter:
    """Create review units without inventing clause labels or geometry."""

    def segment(
        self,
        result: DocumentProcessingResult,
        *,
        contract_id: UUID,
        processing_revision: int,
    ) -> tuple[ReviewUnit, ...]:
        if processing_revision < 1:
            raise ValueError("Processing revision must be positive")

        if result.source.role is not DocumentRole.CONTRACT:
            raise ValueError(
                "Contract processing requires contract-role source"
            )

        units: list[ReviewUnit] = []

        for section in result.structured.sections:
            label = None

            if section.structure_path != ("root",):
                label = " / ".join(section.structure_path)

            for block in section.blocks:
                span = block.source_span
                review_unit_id = uuid5(
                    NAMESPACE_URL,
                    f"{contract_id}\x1f{processing_revision}\x1f"
                    f"{block.order_index}\x1f{result.source.checksum_sha256}",
                )
                units.append(
                    ReviewUnit(
                        review_unit_id=review_unit_id,
                        contract_id=contract_id,
                        processing_revision=processing_revision,
                        order_index=len(units),
                        text=block.text,
                        label=label,
                        source_spans=(span,),
                        page_start=span.page_number,
                        page_end=span.page_number,
                    )
                )

        if not units:
            raise ValueError("Contract processing produced no review units")

        return tuple(units)
