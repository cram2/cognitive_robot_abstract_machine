"""Tests for :mod:`llmr.backend` — LLMBackend._evaluate pipeline with scripted LLM."""

from __future__ import annotations

from typing_extensions import Any, Dict

import pytest

from llmr.backend import LLMBackend, _UNRESOLVED, _Unresolved
from llmr.bridge.match_reader import required_match
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.schemas import ActionReasoningOutput, EntityDescriptionSchema, SlotValue

from ._fixtures.actions import (
    GraspType,
    MockNavigateAction,
    MockPickUpAction,
    MockRequiredNestedAction,
)
from ._fixtures.symbols import WorldBody
from ._fixtures.worlds import symbol_world  # noqa: F401
from .scripted_llm import ScriptedLLM


class TestUnresolvedSentinel:
    """The module-level ``_UNRESOLVED`` sentinel."""

    def test_repr(self) -> None:
        assert repr(_UNRESOLVED) == "<UNRESOLVED>"

    def test_is_unique(self) -> None:
        """Two ``_Unresolved()`` instances are not equal — the module exposes only one."""
        assert _UNRESOLVED is not _Unresolved()


class TestEvaluateFastPath:
    """Fully-resolved Match expressions bypass the LLM call."""

    def test_no_free_slots_skips_llm(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """When every required field is already bound, ``_evaluate`` yields directly."""
        milk = symbol_world["milk_on_table"]
        match = required_match(MockNavigateAction)
        slot = next(iter(match.matches_with_variables))
        slot.assigned_variable._value_ = milk

        # An LLM that would crash if called — verifies the fast path is taken.
        crashing_llm = ScriptedLLM(responses=[])
        backend = LLMBackend(llm=crashing_llm)
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockNavigateAction)
        assert result.target_location is milk


class TestEvaluateHappyPath:
    """LLM-driven slot filling with a scripted entity_description."""

    def test_resolves_single_entity_slot(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        match = required_match(MockPickUpAction)
        response = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            groundable_type=WorldBody,
        )
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockPickUpAction)
        assert result.object_designator is symbol_world["milk_on_table"]


class TestEvaluateErrorPaths:
    """Error behaviours: LLM returning nothing and strict_required unresolved."""

    def test_llm_failure_raises_slot_filling_failed(self) -> None:
        """When the LLM returns ``None``, the backend raises :class:`LLMSlotFillingFailed`."""
        match = required_match(MockPickUpAction)

        class NullLLM(ScriptedLLM):
            def with_structured_output(self, schema: Any, **kwargs: Any):
                from langchain_core.runnables import RunnableLambda

                def _broken(messages: Any, **kw: Any) -> Any:
                    raise RuntimeError("LLM is down")

                return RunnableLambda(_broken)

        backend = LLMBackend(llm=NullLLM(responses=[]), groundable_type=WorldBody)
        with pytest.raises(LLMSlotFillingFailed):
            next(iter(backend.evaluate(match)))

    def test_strict_required_raises_when_unresolved(self) -> None:
        """Empty slot output with ``strict_required=True`` raises :class:`LLMUnresolvedRequiredFields`."""
        match = required_match(MockPickUpAction)
        response = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="does_not_exist"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            groundable_type=WorldBody,
            strict_required=True,
        )
        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            next(iter(backend.evaluate(match)))
        assert "object_designator" in exc_info.value.unresolved_fields


class TestWorldContextProvider:
    """Custom ``world_context_provider`` is used and falls back on exception."""

    def test_provider_injects_custom_text(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        captured: Dict[str, str] = {}

        class RecordingScripted(ScriptedLLM):
            def with_structured_output(self, schema: Any, **kwargs: Any):
                runnable = super().with_structured_output(schema, **kwargs)
                original_invoke = runnable.invoke

                def _invoke(messages: Any, **kw: Any) -> Any:
                    captured["user"] = next(
                        msg["content"] for msg in messages if msg["role"] == "user"
                    )
                    return original_invoke(messages, **kw)

                runnable.invoke = _invoke  # type: ignore[method-assign]
                return runnable

        match = required_match(MockPickUpAction)
        response = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=RecordingScripted(responses=[response]),
            groundable_type=WorldBody,
            world_context_provider=lambda: "## Custom Context\nSPECIAL_WORLD",
        )
        next(iter(backend.evaluate(match)))
        assert "SPECIAL_WORLD" in captured["user"]

    def test_provider_exception_falls_back_to_symbol_graph(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """If the custom provider raises, the backend still serialises the SymbolGraph."""

        def _boom() -> str:
            raise RuntimeError("provider down")

        backend = LLMBackend(
            llm=ScriptedLLM(responses=[]),
            groundable_type=WorldBody,
            world_context_provider=_boom,
        )
        ctx = backend._get_world_context()
        assert "World State Summary" in ctx
