"""Tests for server-controlled backend configuration."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_use_generic_backend_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.system_access_scope == "local-system"
    assert settings.qdrant_collection_name == "compliance_knowledge_v1"
    assert settings.minio_bucket == "compliance-documents"
    assert settings.openai_api_key is None


def test_legacy_provider_names_are_accepted_without_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "tenant_vectors")
    monkeypatch.setenv("MINIO_BUCKET", "tenant-documents")

    settings = Settings(_env_file=None)

    assert settings.qdrant_collection_name == "tenant_vectors"
    assert settings.minio_bucket == "tenant-documents"


def test_system_scope_rejects_untrusted_path_characters() -> None:
    with pytest.raises(ValidationError, match="unsupported characters"):
        Settings(
            _env_file=None,
            system_access_scope="../../another-tenant",
        )
