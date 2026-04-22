"""Tests for :mod:`llmr.factory` — user-facing NL-driven entry points.

``execute_single`` is patched per-test because it wraps PyCRAM's real factory,
which the pure-llmr tests must not invoke.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import Any, Dict, List

import pytest

from llmr.exceptions import LLMActionClassificationFailed
from llmr.factory import nl_plan, nl_sequential, resolve_match, resolve_params
from llmr.schemas import (
    ActionClassification,
    ActionReasoningOutput,
    EntityDescriptionSchema,
    SlotValue,
)

from ._fixtures.actions import MockNavigateAction, MockPickUpAction
from ._fixtures.symbols import WorldBody
from ._fixtures.worlds import symbol_world  # noqa: F401
from .scripted_llm import ScriptedLLM


@pytest.fixture
def fake_execute_single(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Replace ``execute_single`` in factory with a no-op recorder.

    Returns the list of ``(match, context)`` tuples that the factory forwarded.
    """
    calls: List[Any] = []

    def _fake(match: Any, context: Any) -> Any:
        calls.append((match, context))
        return SimpleNamespace(perform=lambda: None, match=match, context=context)

    monkeypatch.setattr("llmr.factory.execute_single", _fake)
    return calls


class TestNlPlan:
    """:func:`nl_plan` — classify → build match → backend → plan node."""

    def test_classifies_and_returns_plan_node(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        llm = ScriptedLLM(
            responses=[
                ActionClassification(action_type="MockPickUpAction"),
                ActionReasoningOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescriptionSchema(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        context = SimpleNamespace(query_backend=None)
        plan = nl_plan(
            instruction="pick up the milk",
            context=context,
            llm=llm,
            groundable_type=WorldBody,
            action_registry={"MockPickUpAction": MockPickUpAction},
        )
        # Factory forwarded (match, context) exactly once.
        assert len(fake_execute_single) == 1
        forwarded_match, forwarded_context = fake_execute_single[0]
        assert forwarded_context is context
        assert forwarded_match.type is MockPickUpAction
        # Backend was attached to the context.
        assert context.query_backend is not None
        # PlanNode stub is returned.
        assert hasattr(plan, "perform")

    def test_raises_when_classifier_returns_none(
        self, fake_execute_single: List[Any]
    ) -> None:
        """An unknown action_type from the classifier yields ``LLMActionClassificationFailed``."""
        llm = ScriptedLLM(
            responses=[ActionClassification(action_type="NotARegisteredAction")]
        )
        context = SimpleNamespace(query_backend=None)
        with pytest.raises(LLMActionClassificationFailed):
            nl_plan(
                instruction="do something weird",
                context=context,
                llm=llm,
                action_registry={"MockPickUpAction": MockPickUpAction},
            )
        assert fake_execute_single == []


class TestNlSequential:
    """:func:`nl_sequential` — decompose then plan each step."""

    def test_returns_one_plan_per_decomposed_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        # Stub the decomposer so the test does not invoke the LLM twice.
        from llmr.reasoning import decomposer as decomposer_mod

        def _stub_decompose(self: Any, instruction: str) -> Any:
            return decomposer_mod.DecomposedPlan(
                steps=["navigate to the table", "pick up the milk"],
                dependencies={},
            )

        monkeypatch.setattr(decomposer_mod.TaskDecomposer, "decompose", _stub_decompose)

        llm = ScriptedLLM(
            responses=[
                # Step 1: classification + slot filler for MockNavigateAction
                ActionClassification(action_type="MockNavigateAction"),
                ActionReasoningOutput(
                    action_type="MockNavigateAction",
                    slots=[
                        SlotValue(
                            field_name="target_location",
                            entity_description=EntityDescriptionSchema(name="table"),
                        )
                    ],
                ),
                # Step 2: classification + slot filler for MockPickUpAction
                ActionClassification(action_type="MockPickUpAction"),
                ActionReasoningOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescriptionSchema(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        registry = {
            "MockNavigateAction": MockNavigateAction,
            "MockPickUpAction": MockPickUpAction,
        }
        context = SimpleNamespace(query_backend=None)
        plans = nl_sequential(
            instruction="go to the table and pick up the milk",
            context=context,
            llm=llm,
            groundable_type=WorldBody,
            action_registry=registry,
        )
        assert len(plans) == 2
        assert len(fake_execute_single) == 2


class TestResolveMatch:
    """:func:`resolve_match` — backend wiring for an already-built Match."""

    def test_attaches_backend_and_delegates_to_execute_single(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import required_match

        match = required_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                ActionReasoningOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescriptionSchema(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )
        context = SimpleNamespace(query_backend=None)
        plan = resolve_match(
            match,
            context=context,
            llm=llm,
            groundable_type=WorldBody,
            instruction="pick up the milk",
        )
        assert hasattr(plan, "perform")
        assert context.query_backend is not None
        assert fake_execute_single == [(match, context)]


class TestResolveParams:
    """:func:`resolve_params` — non-executing variant that returns the action instance."""

    def test_returns_constructed_action_instance(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import required_match

        match = required_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                ActionReasoningOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescriptionSchema(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )
        result = resolve_params(
            match,
            llm=llm,
            groundable_type=WorldBody,
            instruction="pick up the milk",
        )
        assert isinstance(result, MockPickUpAction)
        assert result.object_designator is symbol_world["milk_on_table"]
