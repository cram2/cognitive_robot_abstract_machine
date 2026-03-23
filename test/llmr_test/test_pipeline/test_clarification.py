from __future__ import annotations

import pytest

from llmr.pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)


class TestClarificationRequest:
    def test_fields_stored(self):
        req = ClarificationRequest(
            entity_name="milk",
            entity_role="object",
            available_names=["apple", "mug"],
            message="Cannot find milk.",
        )
        assert req.entity_name == "milk"
        assert req.entity_role == "object"
        assert req.available_names == ["apple", "mug"]
        assert req.message == "Cannot find milk."

    def test_default_available_names(self):
        req = ClarificationRequest(entity_name="x", entity_role="object")
        assert req.available_names == []

    def test_default_message_empty(self):
        req = ClarificationRequest(entity_name="x", entity_role="object")
        assert req.message == ""


class TestClarificationNeededError:
    def test_carries_request(self):
        req = ClarificationRequest(
            entity_name="cup", entity_role="object", message="Cup not found."
        )
        err = ClarificationNeededError(req)
        assert err.request is req

    def test_message_from_request(self):
        req = ClarificationRequest(entity_name="cup", entity_role="object", message="Cup not found.")
        err = ClarificationNeededError(req)
        assert str(err) == "Cup not found."

    def test_fallback_message_uses_entity_name(self):
        req = ClarificationRequest(entity_name="cup", entity_role="object", message="")
        err = ClarificationNeededError(req)
        assert "cup" in str(err)

    def test_is_exception(self):
        req = ClarificationRequest(entity_name="x", entity_role="object")
        with pytest.raises(ClarificationNeededError):
            raise ClarificationNeededError(req)

    def test_caught_with_request_intact(self):
        req = ClarificationRequest(entity_name="x", entity_role="object")
        caught = None
        try:
            raise ClarificationNeededError(req)
        except ClarificationNeededError as e:
            caught = e
        assert caught is not None
        assert caught.request is req


class TestArmCapacityRequest:
    def test_fields_stored(self):
        req = ArmCapacityRequest(
            occupied_arms=["LEFT", "RIGHT"],
            held_object_names=["milk", "cup"],
            message="Both arms occupied.",
        )
        assert req.occupied_arms == ["LEFT", "RIGHT"]
        assert req.held_object_names == ["milk", "cup"]
        assert req.message == "Both arms occupied."

    def test_defaults(self):
        req = ArmCapacityRequest()
        assert req.occupied_arms == []
        assert req.held_object_names == []
        assert req.message == ""


class TestArmCapacityError:
    def test_carries_request(self):
        req = ArmCapacityRequest(held_object_names=["milk"], message="All arms occupied.")
        err = ArmCapacityError(req)
        assert err.request is req

    def test_message_from_request(self):
        req = ArmCapacityRequest(message="All arms occupied.")
        err = ArmCapacityError(req)
        assert str(err) == "All arms occupied."

    def test_fallback_message_uses_held_names(self):
        req = ArmCapacityRequest(held_object_names=["milk"])
        err = ArmCapacityError(req)
        assert "milk" in str(err)

    def test_is_exception(self):
        req = ArmCapacityRequest()
        with pytest.raises(ArmCapacityError):
            raise ArmCapacityError(req)
