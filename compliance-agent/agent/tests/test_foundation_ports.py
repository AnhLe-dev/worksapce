"""Verify that agent ports expose only pre-authorized capabilities."""

from inspect import signature
from uuid import UUID

from compliance_agent import (
    ReferenceUnitProvider,
    ReviewUnitProvider,
    SelectedKnowledgeRetriever,
)


class FakeReviewUnitProvider:
    def get_review_units(
        self,
        *,
        review_unit_ids: tuple[UUID, ...],
    ) -> tuple:
        del review_unit_ids
        return ()


class FakeSelectedKnowledgeRetriever:
    def search_selected_knowledge(
        self,
        *,
        review_unit_id: UUID,
        query: str,
        limit: int,
    ) -> tuple:
        del review_unit_id, query, limit
        return ()


class FakeReferenceUnitProvider:
    def get_reference_unit(self, *, reference_unit_id: UUID) -> object:
        del reference_unit_id
        return object()


def test_fake_ports_satisfy_runtime_protocols() -> None:
    assert isinstance(FakeReviewUnitProvider(), ReviewUnitProvider)
    assert isinstance(
        FakeSelectedKnowledgeRetriever(),
        SelectedKnowledgeRetriever,
    )
    assert isinstance(FakeReferenceUnitProvider(), ReferenceUnitProvider)


def test_ports_do_not_accept_authority_identifiers() -> None:
    forbidden = {
        "access_scope",
        "contract_id",
        "knowledge_base_id",
        "snapshot_id",
    }
    methods = (
        ReviewUnitProvider.get_review_units,
        SelectedKnowledgeRetriever.search_selected_knowledge,
        ReferenceUnitProvider.get_reference_unit,
    )

    for method in methods:
        assert forbidden.isdisjoint(signature(method).parameters)
