"""Retrieves relevant long-term memories (tag-based + semantic) to inject into prompts."""

import logging

from langchain_core.messages import BaseMessage

from ..config import MEMORY_CONFIG
from ..utils.embeddings import EmbeddingHelper
from .long_term_memory_store import LongTermMemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Fetches the most relevant memories for a given user and query context."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embedder: EmbeddingHelper | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        user_id: str,
        current_message: str | None = None,
        tags: list[str] | None = None,
        max_results: int = MEMORY_CONFIG.max_facts_to_inject,
    ) -> list[dict]:
        """Return de-duplicated memories: semantic matches → tag matches → recent facts."""
        seen_contents: set[str] = set()
        memories: list[dict] = []

        def add(items: list[dict]) -> None:
            for memory_item in items:
                content_key = memory_item.get("content", "")
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    memories.append(memory_item)

        if current_message and self.embedder:
            query_vec = self.embedder.embed(current_message)
            if query_vec is not None:
                try:
                    semantic = self.store.semantic_search(
                        query_embedding=query_vec,
                        user_id=user_id,
                        top_k=MEMORY_CONFIG.max_semantic_results,
                    )
                    add(semantic)
                except Exception as e:
                    if "SearchNotEnabled" in str(e) or "vectorSearch" in str(e):
                        logger.warning(
                            "Semantic search unavailable (requires Atlas) — using tag search only."
                        )
                    else:
                        raise

        if tags:
            tagged = self.store.get_by_tags(user_id, tags, limit=max_results)
            add(tagged)

        facts = self.store.get_all(user_id, memory_type="fact", limit=max_results)
        preferences = self.store.get_all(user_id, memory_type="preference", limit=3)
        summary = self.store.get_summary(user_id)

        add(facts)
        add(preferences)
        if summary:
            add([summary])

        return memories[:max_results]

    def format_for_prompt(
        self,
        user_id: str,
        current_message: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Return a ready-to-inject string block for use in system prompts."""
        memories = self.retrieve(user_id, current_message, tags)
        if not memories:
            return ""

        grouped: dict[str, list[str]] = {}
        for mem in memories:
            memory_type = mem.get("memory_type", "fact")
            grouped.setdefault(memory_type, []).append(mem["content"])

        type_labels: dict[str, str] = {
            "fact": "Known facts",
            "preference": "User preferences",
            "summary": "Previous conversation summary",
            "entity": "Relevant entities",
        }

        formatted_lines: list[str] = []
        for memory_type, contents in grouped.items():
            label = type_labels.get(memory_type, memory_type.title())
            formatted_lines.append(f"{label}:")
            for content_item in contents:
                formatted_lines.append(f"  - {content_item}")
            formatted_lines.append("")

        return "\n".join(formatted_lines).strip()

    def get_relevant_tags_from_message(self, message: str) -> list[str]:
        """Simple keyword-to-tag mapping for common support topics."""
        keyword_map: dict[str, list[str]] = {
            "billing": ["billing", "payment", "invoice", "charge"],
            "technical": ["technical", "bug", "error", "issue", "broken"],
            "account": ["account", "login", "password", "access"],
            "shipping": ["shipping", "delivery", "order", "package"],
            "refund": ["refund", "return", "money back"],
        }
        message_lower = message.lower()
        matched_tags: list[str] = []
        for tag, keywords in keyword_map.items():
            if any(keyword in message_lower for keyword in keywords):
                matched_tags.append(tag)
        return matched_tags
