"""Provider-independent orchestration for generic grounded RAG."""

from rag_core.generation.citations import validate_answer_citations
from rag_core.generation.context import (
    build_context_sections,
    render_context,
)
from rag_core.generation.errors import GenerationValidationError
from rag_core.generation.models import (
    AnswerStatus,
    GroundedAnswer,
    GroundedCitation,
    GroundedExecution,
    GroundedExecutionTrace,
)
from rag_core.generation.ports import AnswerGenerator
from rag_core.generation.prompt import build_grounded_prompt
from rag_core.retrieval.models import KnowledgeHit, RetrievalRequest
from rag_core.retrieval.ports import Retriever

_ABSTENTION_MESSAGE = (
    "The authorized knowledge sources do not contain enough "
    "evidence to answer this question."
)


def _evidence_identity(hit: KnowledgeHit) -> str:
    """Build one trace identity without relying on chunk ID alone."""

    return (
        f"{hit.access_scope}\x1f"
        f"{hit.source_id}\x1f"
        f"{hit.source_version}\x1f"
        f"{hit.source_checksum_sha256}\x1f"
        f"{hit.index_profile}\x1f"
        f"{hit.chunk_id}"
    )


class GroundedRagService:
    """Coordinate retrieval, generation, and citation linkage."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator

    def execute(
        self,
        request: RetrievalRequest,
    ) -> GroundedExecution:
        """Answer from authorized scopes or abstain without provider use."""

        hits = self._retriever.retrieve(request)

        if not hits:
            return GroundedExecution(
                result=GroundedAnswer(
                    question=request.query,
                    answer=_ABSTENTION_MESSAGE,
                    status=AnswerStatus.ABSTAINED,
                    citations=(),
                    retrieved_count=0,
                ),
                trace=GroundedExecutionTrace(),
            )

        sections = build_context_sections(hits)
        context = render_context(sections)
        prompt = build_grounded_prompt(
            question=request.query,
            context=context,
        )
        raw_answer = self._answer_generator.generate(prompt)
        answer = raw_answer.strip()

        if not answer:
            raise GenerationValidationError(
                "Answer provider returned empty text"
            )

        used_labels = validate_answer_citations(
            answer=answer,
            context=context,
        )
        citations = tuple(
            self._citation_for_label(
                label=label,
                hits=hits,
            )
            for label in used_labels
        )

        return GroundedExecution(
            result=GroundedAnswer(
                question=request.query,
                answer=answer,
                status=AnswerStatus.GROUNDED,
                citations=citations,
                retrieved_count=len(hits),
            ),
            trace=GroundedExecutionTrace(
                retrieved_contexts=sections,
                retrieved_context_ids=tuple(
                    _evidence_identity(hit)
                    for hit in hits
                ),
                rendered_context=context,
                used_citation_ids=used_labels,
            ),
        )

    @staticmethod
    def _citation_for_label(
        *,
        label: str,
        hits: tuple[KnowledgeHit, ...],
    ) -> GroundedCitation:
        """Map a validated source label to exact retrieved evidence."""

        hit_index = int(label.removeprefix("S")) - 1

        if not 0 <= hit_index < len(hits):
            raise GenerationValidationError(
                "Citation does not map to a retrieval result: "
                f"{label}"
            )

        hit = hits[hit_index]

        return GroundedCitation(
            citation_id=label,
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_version=hit.source_version,
            source_checksum_sha256=hit.source_checksum_sha256,
            access_scope=hit.access_scope,
            index_profile=hit.index_profile,
            text=hit.text,
            structure_path=hit.structure_path,
            source_spans=hit.source_spans,
            page_start=hit.page_start,
            page_end=hit.page_end,
            score=hit.score,
            document_format=hit.document_format,
            role=hit.role,
            metadata=dict(hit.metadata),
        )
