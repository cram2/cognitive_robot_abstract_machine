"""ScriptedLLM — deterministic BaseChatModel for testing without API keys.

Cycles through pre-built Pydantic instances (responses) in order.
Overrides with_structured_output() to return instances directly, bypassing JSON parsing.
Zero API key, zero network, fully reproducible.
"""
from __future__ import annotations

from typing_extensions import Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, PrivateAttr


class ScriptedLLM(BaseChatModel):
    """Deterministic BaseChatModel for tests — no API key, no network.

    Cycles through `responses` (pre-built Pydantic instances) in order.
    `with_structured_output` returns the next response directly, bypassing JSON.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: List[BaseModel]
    """Ordered list of Pydantic instances to return, one per invoke call."""

    _call_index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-llm"

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a response by returning the next pre-built instance as JSON."""
        obj = self.responses[self._call_index % len(self.responses)]
        self._call_index += 1
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=obj.model_dump_json()))
            ]
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        """Return a Runnable that yields pre-built instances directly.

        Overrides BaseChatModel.with_structured_output to bypass JSON parsing.
        The returned Runnable ignores the schema and messages, cycling through responses.
        """
        responses = self.responses
        idx = [self._call_index]

        def _invoke(messages: Any, **kw: Any) -> BaseModel:
            obj = responses[idx[0] % len(responses)]
            idx[0] += 1
            return obj

        return RunnableLambda(_invoke)
