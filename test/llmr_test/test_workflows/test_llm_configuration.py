from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── LLMFactory ────────────────────────────────────────────────────────────────


class TestLLMFactory:
    def test_create_openai_llm(self):
        from llmr.workflows.llm_configuration import LLMFactory, OpenAILLM

        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = LLMFactory.create_llm("openai", "gpt-4o", temperature=0.5)
        assert isinstance(llm, OpenAILLM)

    def test_create_ollama_llm(self):
        from llmr.workflows.llm_configuration import LLMFactory, OllamaLLM

        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = LLMFactory.create_llm("ollama", "qwen3:14b", temperature=0.5)
        assert isinstance(llm, OllamaLLM)

    def test_unknown_provider_raises_value_error(self):
        from llmr.workflows.llm_configuration import LLMFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.create_llm("anthropic", "claude-3")

    def test_provider_lookup_is_case_insensitive(self):
        from llmr.workflows.llm_configuration import LLMFactory, OpenAILLM

        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = LLMFactory.create_llm("OpenAI", "gpt-4o")
        assert isinstance(llm, OpenAILLM)


# ── Lazy __getattr__ ─────────────────────────────────────────────────────────


class TestLazyGetattr:
    def _clean_cache(self, *names: str) -> None:
        """Remove cached singleton entries from the module globals."""
        import llmr.workflows.llm_configuration as mod

        for name in names:
            mod.__dict__.pop(name, None)

    def test_gpt_llm_small_is_openai_instance(self):
        import llmr.workflows.llm_configuration as mod
        from llmr.workflows.llm_configuration import OpenAILLM

        self._clean_cache("gpt_llm_small")
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            inst = mod.gpt_llm_small
        assert isinstance(inst, OpenAILLM)

    def test_gpt_llm_small_is_cached_on_second_access(self):
        import llmr.workflows.llm_configuration as mod

        self._clean_cache("gpt_llm_small")
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            inst1 = mod.gpt_llm_small
            inst2 = mod.gpt_llm_small
        assert inst1 is inst2

    def test_default_llm_resolves_to_gpt_llm_small(self):
        """default_llm and gpt_llm_small refer to the same cached instance.

        __getattr__("default_llm") calls __getattr__("gpt_llm_small") internally
        which (re-)creates and caches the singleton.  Accessing default_llm first
        populates globals()["gpt_llm_small"] so a subsequent access returns the
        same object.
        """
        import llmr.workflows.llm_configuration as mod

        self._clean_cache("gpt_llm_small", "default_llm")
        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            # Access default_llm FIRST — this also sets globals()["gpt_llm_small"]
            default = mod.default_llm
            # Now gpt_llm_small returns the already-cached instance
            small = mod.gpt_llm_small
        assert default is small

    def test_unknown_attribute_raises_attribute_error(self):
        import llmr.workflows.llm_configuration as mod

        with pytest.raises(AttributeError, match="nonexistent_llm_xyz"):
            _ = mod.nonexistent_llm_xyz

    def test_ollama_llm_large_is_ollama_instance(self):
        import llmr.workflows.llm_configuration as mod
        from llmr.workflows.llm_configuration import OllamaLLM

        self._clean_cache("ollama_llm_large")
        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            inst = mod.ollama_llm_large
        assert isinstance(inst, OllamaLLM)


# ── LLMs base class and subclass behaviour ────────────────────────────────────


class TestOpenAILLM:
    def test_with_structured_output_delegates_to_client(self):
        from llmr.workflows.llm_configuration import OpenAILLM

        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            llm = OpenAILLM(model_name="gpt-4o", temperature=0.5)

        schema = MagicMock()
        llm.with_structured_output(schema, method="function_calling")
        mock_client.with_structured_output.assert_called_once_with(
            schema, method="function_calling"
        )

    def test_invoke_returns_content(self):
        from llmr.workflows.llm_configuration import OpenAILLM

        with patch("llmr.workflows.llm_configuration.ChatOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.invoke.return_value.content = "hello"
            mock_cls.return_value = mock_client
            llm = OpenAILLM(model_name="gpt-4o", temperature=0.0)

        assert llm.invoke("prompt") == "hello"


class TestOllamaLLM:
    def test_with_structured_output_delegates_to_client(self):
        from llmr.workflows.llm_configuration import OllamaLLM

        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            llm = OllamaLLM(model_name="qwen3:14b", temperature=0.5)

        schema = MagicMock()
        llm.with_structured_output(schema)
        mock_client.with_structured_output.assert_called_once_with(schema)

    def test_stream_yields_content(self):
        from llmr.workflows.llm_configuration import OllamaLLM

        with patch("llmr.workflows.llm_configuration.ChatOllama") as mock_cls:
            chunk1 = MagicMock()
            chunk1.content = "hello"
            chunk2 = MagicMock()
            chunk2.content = " world"
            mock_client = MagicMock()
            mock_client.stream.return_value = iter([chunk1, chunk2])
            mock_cls.return_value = mock_client
            llm = OllamaLLM(model_name="qwen3:14b")

        tokens = list(llm.stream("prompt"))
        assert tokens == ["hello", " world"]
