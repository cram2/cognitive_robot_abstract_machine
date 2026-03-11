"""
config.py — Central configuration for the entire project.
All env vars and constants live here.
"""

import os
from dataclasses import dataclass

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the workflows directory — works regardless of CWD
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass(frozen=True)
class MongoConfig:
    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name: str = os.getenv("MONGODB_DB", "langgraph_memory")

    # Collection names
    col_checkpoints: str = "checkpoints"
    col_long_term: str = "long_term_memories"
    col_conversations: str = "conversation_summaries"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai")  # anthropic | openai
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    max_tokens: int = 1024


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai | local
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dimensions: int = 1536


@dataclass(frozen=True)
class MemoryConfig:
    # How many long-term memories to inject into each prompt
    max_facts_to_inject: int = 5
    max_semantic_results: int = 3
    # Auto-summarize after N messages
    summarize_after_turns: int = 10


# Singleton instances — import these throughout the project
MONGO_CONFIG = MongoConfig()
LLM_CONFIG = LLMConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
MEMORY_CONFIG = MemoryConfig()
