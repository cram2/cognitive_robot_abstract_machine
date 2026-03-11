"""Tests for the EmbeddingHelper classes."""

from llmr.workflows.lg_memory.utils.embeddings import NoOpEmbeddingHelper


class TestNoOpEmbeddingHelper:
    def test_embed_returns_none(self) -> None:
        h = NoOpEmbeddingHelper()
        assert h.embed("hello") is None

    def test_embed_many_returns_list_of_none(self) -> None:
        h = NoOpEmbeddingHelper()
        result = h.embed_many(["a", "b", "c"])
        assert result == [None, None, None]

    def test_dimensions_zero(self) -> None:
        h = NoOpEmbeddingHelper()
        assert h.dimensions == 0

    def test_embed_empty_string_returns_none(self) -> None:
        h = NoOpEmbeddingHelper()
        assert h.embed("") is None

    def test_embed_many_empty_list(self) -> None:
        h = NoOpEmbeddingHelper()
        assert h.embed_many([]) == []

    def test_embed_many_single_element(self) -> None:
        h = NoOpEmbeddingHelper()
        result = h.embed_many(["test"])
        assert result == [None]
