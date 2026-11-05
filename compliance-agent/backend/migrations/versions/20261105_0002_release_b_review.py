"""Allow durable contract review jobs.

Revision ID: 20261105_0002
Revises: 20261105_0001
Create Date: 2026-11-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261105_0002"
down_revision: str | None = "20261105_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_jobs_kind_matches_target"),
        "jobs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_jobs_kind_matches_target"),
        "jobs",
        "(kind IN ('contract_processing', 'contract_review') "
        "AND contract_id IS NOT NULL) OR "
        "(kind = 'knowledge_indexing' AND snapshot_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_jobs_kind_matches_target"),
        "jobs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_jobs_kind_matches_target"),
        "jobs",
        "(kind = 'contract_processing' AND contract_id IS NOT NULL) OR "
        "(kind = 'knowledge_indexing' AND snapshot_id IS NOT NULL)",
    )
