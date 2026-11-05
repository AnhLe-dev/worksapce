"""Official OpenAI Responses adapter for grounded answer generation."""

from openai import OpenAI

from rag_core.generation.errors import GenerationValidationError
from rag_core.generation.models import GenerationPrompt


class OpenAIAnswerGenerator:
    """Generate answer text through an injected OpenAI client."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        max_output_tokens: int,
    ) -> None:
        normalized_model = model.strip()

        if not normalized_model:
            raise GenerationValidationError(
                "OpenAI generation model must not be empty"
            )

        if max_output_tokens <= 0:
            raise GenerationValidationError(
                "Maximum output tokens must be greater than zero"
            )

        self._client = client
        self._model = normalized_model
        self._max_output_tokens = max_output_tokens

    def generate(self, prompt: GenerationPrompt) -> str:
        """Map the provider-neutral prompt to the Responses API."""

        response = self._client.responses.create(
            model=self._model,
            instructions=prompt.instructions,
            input=prompt.input_text,
            max_output_tokens=self._max_output_tokens,
        )
        answer = response.output_text.strip()

        if not answer:
            raise GenerationValidationError(
                "OpenAI response did not contain output text"
            )

        return answer
