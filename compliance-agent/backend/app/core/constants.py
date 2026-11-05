"""Backend defaults that contain no credentials or domain-specific corpus."""

from pathlib import Path
from typing import Final

BACKEND_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT: Final[Path] = BACKEND_ROOT.parents[1]

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_SYSTEM_ACCESS_SCOPE: Final[str] = "local-system"
DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 25 * 1024 * 1024
MAX_UPLOAD_BYTES: Final[int] = 100 * 1024 * 1024

DEFAULT_DATABASE_URL: Final[str] = (
    "postgresql+psycopg://rag:rag_dev_only@postgres:5432/rag"
)
DEFAULT_CELERY_BROKER_URL: Final[str] = "redis://redis:6379/0"
DEFAULT_MINIO_BUCKET: Final[str] = "compliance-documents"
DEFAULT_MINIO_SECURE: Final[bool] = False
DEFAULT_QDRANT_COLLECTION: Final[str] = "compliance_knowledge_v1"
DEFAULT_JOB_LEASE_SECONDS: Final[int] = 360
DEFAULT_JOB_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_OUTBOX_LEASE_SECONDS: Final[int] = 60
DEFAULT_OUTBOX_BATCH_SIZE: Final[int] = 50
DEFAULT_JOB_MAINTENANCE_INTERVAL_SECONDS: Final[float] = 2.0
DEFAULT_MAX_CHUNK_CHARACTERS: Final[int] = 1200
DEFAULT_DEPENDENCY_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_CORS_ALLOWED_ORIGINS: Final[str] = (
    "http://localhost:3000,http://127.0.0.1:3000"
)

DEFAULT_LLM_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS: Final[int] = 1536
DEFAULT_REVIEW_RETRIEVAL_LIMIT: Final[int] = 5
DEFAULT_REVIEW_MAX_MODEL_CALLS: Final[int] = 8
DEFAULT_REVIEW_MAX_TOTAL_TOOL_CALLS: Final[int] = 40
DEFAULT_REVIEW_MAX_KNOWLEDGE_SEARCH_CALLS: Final[int] = 20
DEFAULT_REVIEW_MAX_REFERENCE_CALLS: Final[int] = 20
DEFAULT_REVIEW_TIMEOUT_SECONDS: Final[float] = 60.0
