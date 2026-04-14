"""Tests for llmr exception classes.

All exceptions inherit from krrood.utils.DataclassException and provide
structured error messages via __post_init__.

Coverage target: 100% (5 tests for 5 exception classes).
"""
from __future__ import annotations

import pytest
from llmr.exceptions import (
    LLMProviderNotSupported,
    LLMActionClassificationFailed,
    LLMSlotFillingFailed,
    LLMUnresolvedRequiredFields,
    LLMActionRegistryEmpty,
)


class TestLLMProviderNotSupported:
    """LLMProviderNotSupported exception."""

    def test_message_contains_provider_and_valid_options(self) -> None:
        """Message should include the invalid provider and list valid options."""
        exc = LLMProviderNotSupported(
            provider="bad_provider", valid_providers=["openai", "ollama"]
        )
        assert "bad_provider" in exc.message
        assert "openai" in exc.message
        assert "ollama" in exc.message
        assert "Unknown LLM provider" in exc.message


class TestLLMActionClassificationFailed:
    """LLMActionClassificationFailed exception."""

    def test_message_contains_instruction(self) -> None:
        """Message should include the instruction that failed to classify."""
        instruction = "pick up the milk from the table"
        exc = LLMActionClassificationFailed(instruction=instruction)
        assert instruction in exc.message
        assert "Could not classify" in exc.message


class TestLLMSlotFillingFailed:
    """LLMSlotFillingFailed exception."""

    def test_message_contains_action_name(self) -> None:
        """Message should include the action name that failed to fill."""
        exc = LLMSlotFillingFailed(action_name="PickUpAction")
        assert "PickUpAction" in exc.message
        assert "slot filler" in exc.message.lower()


class TestLLMUnresolvedRequiredFields:
    """LLMUnresolvedRequiredFields exception."""

    def test_message_lists_all_unresolved_field_names(self) -> None:
        """Message should list all unresolved fields."""
        exc = LLMUnresolvedRequiredFields(
            action_name="PickUpAction",
            unresolved_fields=["object_designator", "grasp_description"],
        )
        assert "PickUpAction" in exc.message
        assert "object_designator" in exc.message
        assert "grasp_description" in exc.message


class TestLLMActionRegistryEmpty:
    """LLMActionRegistryEmpty exception."""

    def test_has_non_empty_message(self) -> None:
        """Message should be descriptive."""
        exc = LLMActionRegistryEmpty()
        assert len(exc.message) > 0
        assert "registry" in exc.message.lower() or "action" in exc.message.lower()
