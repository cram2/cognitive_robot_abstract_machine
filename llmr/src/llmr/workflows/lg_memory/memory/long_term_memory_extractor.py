"""Uses an LLM to extract and persist structured memories from conversations."""

import json
import logging
import re
from typing_extensions import Any

logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage, HumanMessage

from ..config import LLM_CONFIG, MEMORY_CONFIG
from ..utils.embeddings import EmbeddingHelper
from .long_term_memory_store import LongTermMemoryStore, MemoryType


def _get_llm():
    """Instantiate the configured LLM provider for memory extraction."""
    if LLM_CONFIG.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=LLM_CONFIG.model, max_tokens=LLM_CONFIG.max_tokens)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_CONFIG.model, max_tokens=LLM_CONFIG.max_tokens)


class MemoryExtractor:
    """Extracts structured memories from conversations using an LLM and saves them to the store."""

    EXTRACTION_PROMPT = """You are a memory extraction system. Analyze the conversation below and extract important long-term information about the user.

Extract ONLY genuinely useful long-term information. Ignore small talk, pleasantries, and one-time context.

Return a JSON array of memory objects. Each object must have:
- "content": clear, concise statement about the user (1-2 sentences)
- "memory_type": one of "fact" | "preference" | "summary" | "entity"
- "tags": list of relevant keyword tags (2-4 tags)

Memory type guide:
- "fact"       -> objective info: name, location, job, situation
- "preference" -> how they like things: communication style, format, tools
- "summary"    -> recap of what happened in this conversation
- "entity"     -> important things they mentioned: products, companies, projects

Return ONLY valid JSON. If nothing worth saving, return [].

Conversation:
{conversation}
"""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embedder: EmbeddingHelper | None = None,
    ) -> None:
        self.store = store
        self._llm = None
        self.embedder = embedder

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _get_llm()
        return self._llm

    def _format_conversation(self, messages: list[BaseMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _parse_llm_response(self, raw: str) -> list[dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code fences."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            result = json.loads(cleaned)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    def extract_and_save(
        self,
        messages: list[BaseMessage],
        user_id: str,
        source_thread_id: str | None = None,
    ) -> list[str]:
        """Extract memories from a list of messages and save them.

        Returns list of inserted MongoDB IDs.
        """
        if not messages:
            return []

        conversation = self._format_conversation(messages)
        prompt = self.EXTRACTION_PROMPT.format(conversation=conversation)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        memories = self._parse_llm_response(response.content)

        saved_ids: list[str] = []
        for mem in memories:
            content = mem.get("content", "").strip()
            memory_type: MemoryType = mem.get("memory_type", "fact")
            tags: list[str] = mem.get("tags", [])

            if not content:
                continue

            embedding = self.embedder.embed(content) if self.embedder else None

            doc_id = self.store.upsert(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                tags=tags,
                embedding=embedding,
                metadata={"source": "auto_extracted"},
            )
            saved_ids.append(doc_id)
            logger.debug("Memory saved (%s): %s...", memory_type, content[:70])

        return saved_ids

    def generate_summary(
        self,
        messages: list[BaseMessage],
        user_id: str,
        source_thread_id: str | None = None,
    ) -> str | None:
        """Generate and save a conversation summary. Returns the MongoDB doc ID, or None if too short."""
        if len(messages) < MEMORY_CONFIG.summarize_after_turns:
            return None

        conversation = self._format_conversation(messages)
        summary_prompt = (
            "Summarize this support conversation in 2-3 sentences.\n"
            "Focus on: what the user needed, what actions were taken, and the outcome.\n\n"
            f"Conversation:\n{conversation}"
        )

        response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()

        embedding = self.embedder.embed(summary) if self.embedder else None

        doc_id = self.store.save(
            user_id=user_id,
            content=summary,
            memory_type="summary",
            tags=["conversation_summary", "auto_generated"],
            embedding=embedding,
            source_thread_id=source_thread_id,
        )
        logger.debug("Summary saved for user '%s'", user_id)
        return doc_id
