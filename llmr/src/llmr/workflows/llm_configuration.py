
from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod

from typing_extensions import Dict, Iterator, Type

from dotenv import find_dotenv, load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
        """Initialise the provider-specific LLM client."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: object) -> str:
        """Generate a plain-text response from the LLM."""

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Stream a response from the LLM chunk by chunk."""

    @abstractmethod
    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        """Return a client configured to return structured output matching *schema*."""


class OllamaLLM(LLMs):
    """Ollama-hosted LLM (e.g. qwen3:14b)."""

    def _initialize_client(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class OpenAILLM(LLMs):
    """OpenAI GPT LLM."""

    def _initialize_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=_GPT_API_KEY,
            temperature=self.temperature,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class LLMFactory:
    """Creates LLM instances by provider name."""

    _PROVIDERS: Dict[str, Type[LLMs]] = {
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
        key = provider.lower()
        if key not in cls._PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(cls._PROVIDERS)}")
        return cls._PROVIDERS[key](model_name=model_name, **kwargs)


# ── Lazy singletons ────────────────────────────────────────────────────────────
# Declared as type annotations only — actual instances are created on first
# access via module __getattr__ so that importing this module never fails due
# to a missing API key or unreachable model server.

#: Small GPT model (gpt-4o-mini) — fast and cost-effective.
gpt_llm_small: LLMs
#: Large GPT model (gpt-4o) — highest capability.
gpt_llm_large: LLMs
#: Large Ollama-hosted model (qwen3:14b) — local inference.
ollama_llm_large: LLMs
#: Default LLM used across all workflow nodes unless overridden.
default_llm: LLMs

_LAZY_CONFIGS: dict[str, tuple[str, str, float]] = {
    "gpt_llm_small": ("openai", "gpt-4o-mini", 0.5),
    "gpt_llm_large": ("openai", "gpt-4o", 0.5),
    "ollama_llm_large": ("ollama", "qwen3:14b", 0.5),
}


def __getattr__(name: str) -> LLMs:
    """Lazily create and cache LLM singletons on first attribute access."""
    if name in _LAZY_CONFIGS:
        provider, model_name, temperature = _LAZY_CONFIGS[name]
        instance = LLMFactory.create_llm(
            provider=provider, model_name=model_name, temperature=temperature
        )
        globals()[name] = instance
        return instance
    if name == "default_llm":
        instance = __getattr__("gpt_llm_small")
        globals()["default_llm"] = instance
        return instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
