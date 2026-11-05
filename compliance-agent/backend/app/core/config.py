"""Validated application configuration and trusted system boundaries."""

from functools import lru_cache
from re import fullmatch
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_CELERY_BROKER_URL,
    DEFAULT_CORS_ALLOWED_ORIGINS,
    DEFAULT_DATABASE_URL,
    DEFAULT_DEPENDENCY_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JOB_LEASE_SECONDS,
    DEFAULT_JOB_MAINTENANCE_INTERVAL_SECONDS,
    DEFAULT_JOB_MAX_ATTEMPTS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CHUNK_CHARACTERS,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_MINIO_BUCKET,
    DEFAULT_MINIO_SECURE,
    DEFAULT_OUTBOX_BATCH_SIZE,
    DEFAULT_OUTBOX_LEASE_SECONDS,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_REVIEW_MAX_KNOWLEDGE_SEARCH_CALLS,
    DEFAULT_REVIEW_MAX_MODEL_CALLS,
    DEFAULT_REVIEW_MAX_REFERENCE_CALLS,
    DEFAULT_REVIEW_MAX_TOTAL_TOOL_CALLS,
    DEFAULT_REVIEW_RETRIEVAL_LIMIT,
    DEFAULT_REVIEW_TIMEOUT_SECONDS,
    DEFAULT_SYSTEM_ACCESS_SCOPE,
    MAX_UPLOAD_BYTES,
    WORKSPACE_ROOT,
)


class Settings(BaseSettings):
    """Load backend settings without making providers import requirements."""

    model_config = SettingsConfigDict(
        env_file=WORKSPACE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = DEFAULT_LOG_LEVEL
    cors_allowed_origins: str = DEFAULT_CORS_ALLOWED_ORIGINS
    system_access_scope: str = DEFAULT_SYSTEM_ACCESS_SCOPE
    max_document_upload_bytes: int = Field(
        default=DEFAULT_MAX_UPLOAD_BYTES,
        gt=0,
        le=MAX_UPLOAD_BYTES,
        validation_alias=AliasChoices(
            "MAX_DOCUMENT_UPLOAD_BYTES",
            "MAX_CONTRACT_UPLOAD_BYTES",
        ),
    )
    dependency_timeout_seconds: float = Field(
        default=DEFAULT_DEPENDENCY_TIMEOUT_SECONDS,
        ge=1,
        le=30,
    )

    database_url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)
    celery_broker_url: SecretStr = SecretStr(
        DEFAULT_CELERY_BROKER_URL
    )
    job_lease_seconds: int = Field(
        default=DEFAULT_JOB_LEASE_SECONDS,
        ge=30,
        le=3600,
    )
    job_max_attempts: int = Field(
        default=DEFAULT_JOB_MAX_ATTEMPTS,
        ge=1,
        le=20,
    )
    outbox_lease_seconds: int = Field(
        default=DEFAULT_OUTBOX_LEASE_SECONDS,
        ge=10,
        le=600,
    )
    outbox_batch_size: int = Field(
        default=DEFAULT_OUTBOX_BATCH_SIZE,
        ge=1,
        le=500,
    )
    job_maintenance_interval_seconds: float = Field(
        default=DEFAULT_JOB_MAINTENANCE_INTERVAL_SECONDS,
        ge=0.25,
        le=60,
    )

    minio_endpoint: str | None = None
    minio_access_key: SecretStr | None = None
    minio_secret_key: SecretStr | None = None
    minio_bucket: str = Field(
        default=DEFAULT_MINIO_BUCKET,
        min_length=3,
        max_length=63,
        validation_alias=AliasChoices(
            "COMPLIANCE_MINIO_BUCKET",
            "MINIO_BUCKET",
        ),
    )
    minio_secure: bool = DEFAULT_MINIO_SECURE

    qdrant_cloud_url: str | None = None
    qdrant_cloud_api_key: SecretStr | None = None
    qdrant_collection_name: str = Field(
        default=DEFAULT_QDRANT_COLLECTION,
        min_length=1,
        validation_alias=AliasChoices(
            "COMPLIANCE_QDRANT_COLLECTION",
            "QDRANT_COLLECTION_NAME",
        ),
    )

    openai_api_key: SecretStr | None = None
    default_llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        min_length=1,
    )
    default_embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
    )
    embedding_dimensions: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        gt=0,
    )
    max_chunk_characters: int = Field(
        default=DEFAULT_MAX_CHUNK_CHARACTERS,
        ge=200,
        le=100_000,
    )
    review_retrieval_limit: int = Field(
        default=DEFAULT_REVIEW_RETRIEVAL_LIMIT,
        ge=1,
        le=100,
    )
    review_max_model_calls: int = Field(
        default=DEFAULT_REVIEW_MAX_MODEL_CALLS,
        ge=1,
        le=50,
    )
    review_max_total_tool_calls: int = Field(
        default=DEFAULT_REVIEW_MAX_TOTAL_TOOL_CALLS,
        ge=1,
        le=500,
    )
    review_max_knowledge_search_calls: int = Field(
        default=DEFAULT_REVIEW_MAX_KNOWLEDGE_SEARCH_CALLS,
        ge=1,
        le=200,
    )
    review_max_reference_calls: int = Field(
        default=DEFAULT_REVIEW_MAX_REFERENCE_CALLS,
        ge=1,
        le=200,
    )
    review_timeout_seconds: float = Field(
        default=DEFAULT_REVIEW_TIMEOUT_SECONDS,
        ge=1,
        le=600,
    )

    @field_validator("system_access_scope")
    @classmethod
    def validate_system_access_scope(cls, value: str) -> str:
        """Accept only a server-controlled storage-safe scope."""

        normalized = value.strip()

        if not fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
            raise ValueError(
                "SYSTEM_ACCESS_SCOPE contains unsupported characters"
            )

        return normalized

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        normalized: list[str] = []

        for item in value.split(","):
            origin = item.strip().rstrip("/")

            if not origin:
                continue

            parsed = urlsplit(origin)

            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("CORS_ALLOWED_ORIGINS is invalid")

            if origin not in normalized:
                normalized.append(origin)

        if not normalized:
            raise ValueError("CORS_ALLOWED_ORIGINS must not be empty")

        return ",".join(normalized)

    @property
    def cors_origins(self) -> tuple[str, ...]:
        return tuple(self.cors_allowed_origins.split(","))

    @field_validator("minio_endpoint", "qdrant_cloud_url")
    @classmethod
    def normalize_optional_endpoint(cls, value: str | None) -> str | None:
        """Turn blank optional endpoints into an explicit absence."""

        if value is None:
            return None

        normalized = value.strip()
        return normalized or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance for runtime composition."""

    return Settings()
