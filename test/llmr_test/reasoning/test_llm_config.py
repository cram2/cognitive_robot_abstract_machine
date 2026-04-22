"""Tests for LLM factory — make_llm and LLMProvider."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from llmr.exceptions import LLMProviderNotSupported
from llmr.reasoning.llm_config import make_llm, LLMProvider


class TestMakeLlm:
    """make_llm() factory function."""

    def test_openai_provider_raises_import_error_when_not_installed(
        self, monkeypatch
    ) -> None:
        """make_llm with OPENAI raises ImportError if langchain_openai not available."""
        # Monkeypatch the import to fail
        import sys
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langchain_openai" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError):
            make_llm(LLMProvider.OPENAI, model="gpt-4o")

    def test_ollama_provider_raises_import_error_when_not_installed(
        self, monkeypatch
    ) -> None:
        """make_llm with OLLAMA raises ImportError if langchain_ollama not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langchain_ollama" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError):
            make_llm(LLMProvider.OLLAMA, model="qwen3:14b")

    def test_unknown_provider_raises_llm_provider_not_supported(self) -> None:
        """make_llm with unknown provider raises LLMProviderNotSupported."""
        # Create a fake provider value that's not in LLMProvider
        with pytest.raises(LLMProviderNotSupported):
            make_llm("unknown_provider", model="test")  # type: ignore

    def test_default_does_not_import_dotenv(self, monkeypatch) -> None:
        """make_llm does not load .env files unless explicitly requested."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "dotenv":
                raise AssertionError("dotenv should not be imported by default")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(LLMProviderNotSupported):
            make_llm("unknown_provider", model="test")  # type: ignore

    def test_load_env_true_loads_dotenv(self, monkeypatch) -> None:
        """make_llm can explicitly load .env files at call time."""
        calls = []
        fake_dotenv = SimpleNamespace(load_dotenv=lambda: calls.append("loaded"))
        monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

        with pytest.raises(LLMProviderNotSupported):
            make_llm("unknown_provider", model="test", load_env=True)  # type: ignore

        assert calls == ["loaded"]

    def test_model_name_passed_through(self, monkeypatch) -> None:
        """make_llm passes model name to the provider client."""

        # Mock ChatOpenAI to capture the model argument
        class FakeChatOpenAI:
            def __init__(self, model: str, temperature: float = 0.0, **kwargs):
                self.model = model
                self.temperature = temperature

        monkeypatch.setattr(
            "langchain_openai.ChatOpenAI", FakeChatOpenAI, raising=False
        )

        try:
            llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")
            assert llm.model == "gpt-4o"
        except ImportError:
            pytest.skip("langchain-openai not installed")

    def test_temperature_passed_through(self, monkeypatch) -> None:
        """make_llm passes temperature to the provider client."""

        class FakeChatOpenAI:
            def __init__(self, model: str, temperature: float = 0.0, **kwargs):
                self.model = model
                self.temperature = temperature

        monkeypatch.setattr(
            "langchain_openai.ChatOpenAI", FakeChatOpenAI, raising=False
        )

        try:
            llm = make_llm(LLMProvider.OPENAI, model="gpt-4o", temperature=0.7)
            assert llm.temperature == 0.7
        except ImportError:
            pytest.skip("langchain-openai not installed")


class TestLLMProvider:
    """LLMProvider enum."""

    def test_openai_provider_value(self) -> None:
        """LLMProvider.OPENAI has correct string value."""
        assert LLMProvider.OPENAI.value == "openai"

    def test_ollama_provider_value(self) -> None:
        """LLMProvider.OLLAMA has correct string value."""
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_provider_is_enum(self) -> None:
        """LLMProvider is an Enum."""
        assert isinstance(LLMProvider.OPENAI, LLMProvider)
        assert isinstance(LLMProvider.OLLAMA, LLMProvider)
