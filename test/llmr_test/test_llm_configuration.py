"""Tests for LLMFactory and LLM provider classes."""

from unittest.mock import MagicMock, patch

import pytest

from llmr.workflows.llm_configuration import LLMFactory, OllamaLLM, OpenAILLM


class TestLLMFactory:
    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.create_llm(provider="nonexistent", model_name="x")

    def test_openai_provider_returns_openai_llm(self) -> None:
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            llm = LLMFactory.create_llm(provider="openai", model_name="gpt-4o-mini")
            assert isinstance(llm, OpenAILLM)

    def test_ollama_provider_returns_ollama_llm(self) -> None:
        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            llm = LLMFactory.create_llm(provider="ollama", model_name="qwen3:14b")
            assert isinstance(llm, OllamaLLM)

    def test_provider_case_insensitive(self) -> None:
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            llm = LLMFactory.create_llm(provider="OpenAI", model_name="gpt-4o-mini")
            assert isinstance(llm, OpenAILLM)

    def test_error_message_lists_valid_providers(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create_llm(provider="foobar", model_name="x")
        msg = str(exc_info.value).lower()
        assert "openai" in msg or "ollama" in msg

    def test_create_llm_stores_model_name(self) -> None:
        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            llm = LLMFactory.create_llm(provider="ollama", model_name="my-model")
            assert llm.model_name == "my-model"

    def test_create_llm_stores_temperature(self) -> None:
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            llm = LLMFactory.create_llm(provider="openai", model_name="gpt-4o-mini", temperature=0.9)
            assert llm.temperature == 0.9
