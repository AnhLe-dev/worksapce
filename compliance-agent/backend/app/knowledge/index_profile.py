"""Server-owned immutable identity for embedding and chunk configuration."""

from app.core.config import Settings


def build_index_profile(settings: Settings) -> str:
    return (
        f"openai:{settings.default_embedding_model}:"
        f"dimensions={settings.embedding_dimensions}:chunk=block:v1:"
        f"max={settings.max_chunk_characters}"
    )
