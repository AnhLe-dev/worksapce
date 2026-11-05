"""Static migration-chain contract for Release B review jobs."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_release_b_review_migration_is_the_single_head() -> None:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(BACKEND_ROOT / "migrations"),
    )
    scripts = ScriptDirectory.from_config(config)

    assert scripts.get_heads() == ["20261105_0002"]
    assert scripts.get_revision("20261105_0002").down_revision == (
        "20261105_0001"
    )
