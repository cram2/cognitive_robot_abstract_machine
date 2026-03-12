"""
utils/embeddings.py
Embedding helper — wraps OpenAI or local embedding models.
Used for semantic search in the memory retriever.
"""

from ..config import EMBEDDING_CONFIG


class EmbeddingHelper:
    """Thin wrapper around embedding models.

    Swap providers by changing EMBEDDING_PROVIDER env var.

    Providers:
    - "openai" -> OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
    - "local"  -> sentence-transformers (free, runs locally)
    """

    def __init__(self) -> None:
        self.provider = EMBEDDING_CONFIG.provider
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            self._model = OpenAIEmbeddings(model=EMBEDDING_CONFIG.model)
        elif self.provider == "local":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return a list of floats."""
        self._load_model()
        return self._model.embed_query(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one batch call."""
        self._load_model()
        return self._model.embed_documents(texts)

    @property
    def dimensions(self) -> int:
        return EMBEDDING_CONFIG.dimensions


class NoOpEmbeddingHelper:
    """Drop-in replacement when embeddings are disabled."""

    def embed(self, text: str) -> None:
        return None

    def embed_many(self, texts: list[str]) -> list[None]:
        return [None] * len(texts)

    @property
    def dimensions(self) -> int:
        return 0
