"""LLM provider abstractions and factory for llmr."""

import os
import pathlib
from abc import ABC, abstractmethod

from dotenv import find_dotenv, load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Load the .env that lives next to this file first, then fall back to find_dotenv()
_ENV_FILE = pathlib.Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else find_dotenv(), override=True)

_GPT_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")


class LLMs(ABC):
    """Abstract base class for different LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.7, **kwargs: object) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> object:
        """Initialize the specific LLM client."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: object) -> str:
        """Generate a response from the LLM."""

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object):
        """Stream a response from the LLM."""

    @abstractmethod
    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        """Return a version of the client configured for structured output."""


class OllamaLLM(LLMs):
    """Ollama LLM implementation."""

    def _initialize_client(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            reasoning=True,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        response = self.client.invoke(prompt, **kwargs)
        return response.content

    def stream(self, prompt: str, **kwargs: object):
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class OpenAILLM(LLMs):
    """OpenAI GPT LLM implementation."""

    def _initialize_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=_GPT_API_KEY,
            temperature=self.temperature,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        response = self.client.invoke(prompt, **kwargs)
        return response.content

    def stream(self, prompt: str, **kwargs: object):
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class LLMFactory:
    """Factory that creates appropriate LLM instances based on provider name."""

    _PROVIDERS: dict[str, type[LLMs]] = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
    }

    @classmethod
    def create_llm(
        cls,
        provider: str = "ollama",
        model_name: str = "qwen3:14b",
        **kwargs: object,
    ) -> LLMs:
        """Create an LLM instance for the given provider.

        Args:
            provider: "openai" or "ollama".
            model_name: Model identifier string.
            **kwargs: Additional arguments forwarded to the LLM constructor.

        Raises:
            ValueError: If *provider* is not recognised.
        """
        key = provider.lower()
        if key not in cls._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from {list(cls._PROVIDERS)}"
            )
        return cls._PROVIDERS[key](model_name=model_name, **kwargs)


# ── Module-level singletons ────────────────────────────────────────────────────
# These are created once at import time and shared across the workflow.

gpt_llm_small: LLMs = LLMFactory.create_llm(
    provider="openai", model_name="gpt-4o-mini", temperature=0.5
)
gpt_llm_large: LLMs = LLMFactory.create_llm(
    provider="openai", model_name="gpt-4o", temperature=0.5
)
ollama_llm_large: LLMs = LLMFactory.create_llm(
    provider="ollama", model_name="qwen3:14b", temperature=0.5
)

#: Default LLM client used across agent nodes.
default_llm = gpt_llm_small.client


if __name__ == "__main__":
    response = ollama_llm_large.client.invoke("What is Python? Explain in 20 words.")
    print("Ollama response:", response)
