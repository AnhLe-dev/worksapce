"""Official OpenAI SDK adapter for the generic embedding port."""

from collections.abc import Sequence

from openai import OpenAI

from rag_core.indexing.errors import IndexingValidationError


class OpenAIEmbeddingProvider:
    """Generate ordered float embeddings through an injected client."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        dimensions: int,
    ) -> None:
        normalized_model = model.strip()

        if not normalized_model:
            raise IndexingValidationError(
                "OpenAI embedding model must not be empty"
            )

        if dimensions <= 0:
            raise IndexingValidationError(
                "OpenAI embedding dimensions must be greater than zero"
            )

        self._client = client
        self._model = normalized_model
        self._dimensions = dimensions

    @property
    def index_profile(self) -> str:
        """Return the stable embedding portion of a snapshot profile."""

        return f"openai:{self._model}:dimensions={self._dimensions}"

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> Sequence[Sequence[float]]:
        """Embed texts and restore the exact input order by response index."""

        if not texts:
            return ()

        if any(not text.strip() for text in texts):
            raise IndexingValidationError(
                "OpenAI embedding input must not contain empty text"
            )

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            encoding_format="float",
            dimensions=self._dimensions,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        indexes = [item.index for item in ordered]

        if indexes != list(range(len(texts))):
            raise IndexingValidationError(
                "OpenAI embedding response indexes do not match input order"
            )

        return tuple(
            tuple(float(value) for value in item.embedding)
            for item in ordered
        )

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed one non-empty retrieval query."""

        vectors = self.embed_documents((text,))

        return vectors[0]
