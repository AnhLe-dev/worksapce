"""PostgreSQL-backed resolver from KB identity to exact source scopes."""

from uuid import UUID

from app.knowledge.errors import (
    KnowledgeBaseNotFoundError,
    KnowledgeSnapshotNotReadyError,
)
from app.knowledge.models import (
    KnowledgeBaseStatus,
    ResolvedKnowledgeBase,
)
from app.knowledge.repository import KnowledgeRepository, to_base


class KnowledgeRegistry:
    """Resolve authority before any similarity search is attempted."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    def resolve(
        self,
        *,
        knowledge_base_id: UUID,
        trusted_scope: str,
    ) -> ResolvedKnowledgeBase:
        base_record = self._repository.get_base_record(
            knowledge_base_id=knowledge_base_id,
            access_scope=trusted_scope,
        )

        if (
            base_record is None
            or base_record.status is not KnowledgeBaseStatus.ACTIVE
        ):
            raise KnowledgeBaseNotFoundError(
                "Active knowledge base was not found in the trusted scope"
            )

        snapshots = self._repository.list_ready_snapshots(
            knowledge_base_id=knowledge_base_id,
            access_scope=trusted_scope,
        )

        if not snapshots:
            message = "Knowledge base has no READY source snapshots"

            if not self._repository.has_memberships(
                knowledge_base_id=knowledge_base_id,
                access_scope=trusted_scope,
            ):
                message = "Knowledge base has no active memberships"

            raise KnowledgeSnapshotNotReadyError(message)

        return ResolvedKnowledgeBase(
            knowledge_base=to_base(base_record),
            snapshots=snapshots,
            scopes=tuple(
                snapshot.retrieval_scope
                for snapshot in snapshots
            ),
        )
