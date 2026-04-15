"""
LLM factory — lightweight, provider-agnostic.

Design:
  - No global singletons (no module-level default_llm, etc.)
  - No hardcoded default model names
  - No abstract base class wrapping LangChain — use BaseChatModel directly
  - Users create their LLM explicitly and inject it into LLMBackend

This keeps the dependency surface minimal: only langchain-core is required.
Provider-specific packages (langchain-openai, langchain-ollama) are optional
extras installed by the user based on their setup.
"""
from __future__ import annotations

import typing
from enum import Enum

from dotenv import load_dotenv
from typing_extensions import Any

from llmr.exceptions import LLMProviderNotSupported

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

load_dotenv()


class LLMProvider(str, Enum):
    """Provider tag passed to ``make_llm()`` to select the LangChain backend."""
    OPENAI = "openai"
    OLLAMA = "ollama"


def make_llm(
    provider: LLMProvider,
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> "BaseChatModel":
    """
    Factory function for creating a LangChain-compatible chat model.

    The returned object is a standard LangChain BaseChatModel — it can be
    passed directly to LLMBackend, nl_plan(), nl_sequential(), or TaskDecomposer.

    :param provider:    LLM provider (OPENAI or OLLAMA).
    :param model:       Model name/identifier (e.g. "gpt-4o", "qwen3:14b").
    :param temperature: Sampling temperature. Use 0.0 for deterministic output.
    :param kwargs:      Additional provider-specific arguments passed to the client.
    :returns: A LangChain BaseChatModel instance.

    Example::

        from llmr.reasoning.llm_config import make_llm, LLMProvider
        from llmr import nl_plan

        llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")
        plan = nl_plan("pick up the milk", context=context, llm=llm)

        # Or with Ollama for local inference:
        llm = make_llm(LLMProvider.OLLAMA, model="qwen3:14b")
    """
    if provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install 'llmr[openai]'"
            ) from e
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Run: pip install 'llmr[ollama]'"
            ) from e
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    raise LLMProviderNotSupported(
        provider=provider,
        valid_providers=[p.value for p in LLMProvider],
    )
