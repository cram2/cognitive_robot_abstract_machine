from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pycram.datastructures.enums import Arms

from llmr.pipeline.action_dispatcher import (
    ActionDispatcher,
    ActionHandler,
    PickUpActionHandler,
    WorldContext,
)
from llmr.pipeline.clarification import ClarificationNeededError
from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import GraspParamsSchema


# ── ActionDispatcher ──────────────────────────────────────────────────────────


class TestActionDispatcher:
    def test_dispatch_pickup_calls_handler(self, entity_description, mock_world):
        """dispatch() routes to the registered PickUpAction handler."""
        ctx = WorldContext()
        dispatcher = ActionDispatcher(world=mock_world, world_context=ctx)
        mock_handler = MagicMock()
        mock_handler.execute.return_value = MagicMock()
        dispatcher._handlers["PickUpAction"] = mock_handler

        from llmr.workflows.schemas.pick_up import PickUpSlotSchema

        schema = PickUpSlotSchema(object_description=entity_description)
        dispatcher.dispatch(schema)
        mock_handler.execute.assert_called_once_with(schema)

    def test_dispatch_unknown_action_raises_key_error(self, mock_world):
        ctx = WorldContext()
        dispatcher = ActionDispatcher(world=mock_world, world_context=ctx)

        fake_schema = MagicMock()
        fake_schema.action_type = "UnknownAction_xyz_not_real"
        dispatcher._handlers.pop("UnknownAction_xyz_not_real", None)

        with pytest.raises(KeyError, match="UnknownAction_xyz_not_real"):
            dispatcher.dispatch(fake_schema)

    def test_register_adds_to_registry(self, mock_world):
        """Registering a new handler makes it available in the registry."""
        sentinel_type = "TestOnlyAction_abc123"

        class _FakeHandler(ActionHandler):
            def execute(self, schema):
                return None

        ActionDispatcher.register(sentinel_type, _FakeHandler)
        assert sentinel_type in ActionDispatcher._registry
        # Clean up to avoid polluting other tests
        del ActionDispatcher._registry[sentinel_type]


# ── PickUpActionHandler._parse_arm ───────────────────────────────────────────


class TestParseArm:
    def _make_handler(self, mock_world):
        ctx = WorldContext()
        return PickUpActionHandler(world=mock_world, world_context=ctx)

    def test_left(self, mock_world):
        h = self._make_handler(mock_world)
        assert h._parse_arm("LEFT") is Arms.LEFT

    def test_right(self, mock_world):
        h = self._make_handler(mock_world)
        assert h._parse_arm("RIGHT") is Arms.RIGHT

    def test_none_returns_none(self, mock_world):
        h = self._make_handler(mock_world)
        assert h._parse_arm(None) is None

    def test_invalid_returns_none(self, mock_world):
        h = self._make_handler(mock_world)
        # Unknown arm string — logs a warning and returns None
        assert h._parse_arm("DIAGONAL") is None


# ── PickUpActionHandler._parse_grasp ─────────────────────────────────────────


class TestParseGrasp:
    def _make_handler(self, mock_world, manipulator=None):
        ctx = WorldContext(manipulator=manipulator)
        return PickUpActionHandler(world=mock_world, world_context=ctx)

    def test_none_params_returns_none(self, mock_world):
        h = self._make_handler(mock_world)
        assert h._parse_grasp(None) is None

    def test_partial_params_returns_none(self, mock_world):
        h = self._make_handler(mock_world)
        params = GraspParamsSchema(approach_direction="FRONT")  # vertical_alignment missing
        assert h._parse_grasp(params) is None

    def test_full_params_but_no_manipulator_returns_none(self, mock_world):
        h = self._make_handler(mock_world, manipulator=None)
        params = GraspParamsSchema(
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
        )
        assert h._parse_grasp(params) is None


# ── PickUpActionHandler._known_params ────────────────────────────────────────


class TestKnownParams:
    def test_no_arm_no_grasp_returns_none_message(self):
        partial = MagicMock()
        partial.kwargs = {"arm": None, "grasp_description": None}
        result = PickUpActionHandler._known_params(partial)
        assert "None" in result

    def test_arm_present_shows_arm(self):
        partial = MagicMock()
        partial.kwargs = {"arm": Arms.LEFT, "grasp_description": None}
        result = PickUpActionHandler._known_params(partial)
        assert "arm" in result
        assert "LEFT" in result

    def test_arm_and_grasp_present(self):
        partial = MagicMock()
        grasp = MagicMock()
        grasp.approach_direction.name = "FRONT"
        grasp.vertical_alignment.name = "TOP"
        grasp.rotate_gripper = True
        partial.kwargs = {"arm": Arms.RIGHT, "grasp_description": grasp}
        result = PickUpActionHandler._known_params(partial)
        assert "arm" in result
        assert "FRONT" in result
        assert "TOP" in result


# ── PickUpActionHandler._missing_params ──────────────────────────────────────


class TestMissingParams:
    def test_all_missing(self):
        partial = MagicMock()
        partial.kwargs = {"arm": None, "grasp_description": None}
        result = PickUpActionHandler._missing_params(partial)
        assert "arm" in result
        assert "approach_direction" in result
        assert "vertical_alignment" in result
        assert "rotate_gripper" in result

    def test_arm_only_missing(self):
        partial = MagicMock()
        grasp = MagicMock()
        partial.kwargs = {"arm": None, "grasp_description": grasp}
        result = PickUpActionHandler._missing_params(partial)
        assert "arm" in result
        assert "approach_direction" not in result

    def test_nothing_missing(self):
        partial = MagicMock()
        partial.kwargs = {"arm": Arms.LEFT, "grasp_description": MagicMock()}
        result = PickUpActionHandler._missing_params(partial)
        assert "All parameters already specified" in result


# ── execute() — grounding failure raises ClarificationNeededError ─────────────


class TestPickUpHandlerExecuteGroundingFailure:
    def test_no_bodies_raises_clarification_error(self, entity_description, mock_world):
        mock_world.bodies = []
        ctx = WorldContext()
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)
        handler._grounder = MagicMock()
        handler._grounder.ground.return_value = MagicMock(bodies=[], warning=None)

        from llmr.workflows.schemas.pick_up import PickUpSlotSchema

        schema = PickUpSlotSchema(object_description=entity_description)
        with pytest.raises(ClarificationNeededError) as exc_info:
            handler.execute(schema)
        assert exc_info.value.request.entity_name == "milk"
