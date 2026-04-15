"""Opt-in live LLM smoke tests.

These tests verify provider/schema integration without asserting fragile prose.
The deterministic ScriptedLLM suite remains the default production test path.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from llmr.reasoning.slot_filler import classify_action, run_slot_filler
from test.llmr_test.test_actions import MockPickUpAction


load_dotenv("llmr/.env", override=True)

pytestmark = pytest.mark.skipif(
    os.getenv("LLMR_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Live LLM tests require LLMR_LIVE_TESTS=1 and OPENAI_API_KEY.",
)


def test_live_classify_action_returns_registered_action(live_llm) -> None:
    result = classify_action(
        "pick up the milk",
        live_llm,
        action_registry={"MockPickUpAction": MockPickUpAction},
    )

    assert result is MockPickUpAction


def test_live_slot_filler_returns_required_slot(live_llm) -> None:
    result = run_slot_filler(
        instruction="pick up the milk",
        action_cls=MockPickUpAction,
        free_slot_names=["object_designator"],
        fixed_slots={},
        world_context=(
            "## World State Summary\n"
            "Scene objects and surfaces: milk, table\n"
            "## Semantic annotations\n"
            "Available types: FoodItem, SupportSurface\n"
            "Per body:\n"
            "  milk: FoodItem\n"
            "  table: SupportSurface\n"
        ),
        llm=live_llm,
    )

    assert result is not None
    assert result.action_type == "MockPickUpAction"
    assert {slot.field_name for slot in result.slots} == {"object_designator"}
    slot = result.slots[0]
    assert slot.entity_description is not None
