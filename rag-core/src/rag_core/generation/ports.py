"""Small port for provider-specific answer generation."""

from typing import Protocol

from rag_core.generation.models import GenerationPrompt


class AnswerGenerator(Protocol):
    """Generate text from a provider-neutral grounded prompt."""

    def generate(self, prompt: GenerationPrompt) -> str:
        """Return non-empty answer text or propagate provider failure."""

        ...
