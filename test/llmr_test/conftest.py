from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Stub missing ROS packages so llmr can be imported without a ROS install ──
# pycram's composite actions import semantic_digital_twin.adapters.ros which
# needs rclpy_message_converter — a ROS 2 package not available in the test
# environment.  Inserting a MagicMock entry before any llmr import prevents
# ModuleNotFoundError without affecting the llmr code under test.
for _ros_mod in (
    "rclpy_message_converter",
    "rclpy_message_converter.message_converter",
):
    if _ros_mod not in sys.modules:
        sys.modules[_ros_mod] = MagicMock()

from llmr.pipeline.action_dispatcher import WorldContext
from llmr.planning.motion_precondition_planner import ExecutionState
from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import (
    GraspParamsSchema,
    PickUpDiscreteResolutionSchema,
    PickUpSlotSchema,
)
from llmr.workflows.schemas.place import PlaceDiscreteResolutionSchema, PlaceSlotSchema


@pytest.fixture
def mock_body():
    """MagicMock Body with .name.name and .global_pose attributes."""
    body = MagicMock()
    body.name.name = "test_object"
    body.global_pose = MagicMock()
    return body


@pytest.fixture
def mock_world():
    """MagicMock World with .bodies, .semantic_annotations, .get_semantic_annotations_by_type."""
    world = MagicMock()
    world.bodies = []
    world.semantic_annotations = []
    world.get_semantic_annotations_by_type.return_value = []
    return world


@pytest.fixture
def world_context():
    return WorldContext(manipulator=MagicMock())


@pytest.fixture
def exec_state():
    return ExecutionState()


@pytest.fixture
def entity_description():
    return EntityDescriptionSchema(name="milk", semantic_type="Milk")


@pytest.fixture
def pickup_slot_schema(entity_description):
    return PickUpSlotSchema(object_description=entity_description)


@pytest.fixture
def place_slot_schema(entity_description):
    return PlaceSlotSchema(
        object_description=entity_description,
        target_description=EntityDescriptionSchema(name="counter"),
    )


@pytest.fixture
def pickup_resolution():
    return PickUpDiscreteResolutionSchema(
        arm="LEFT",
        approach_direction="FRONT",
        vertical_alignment="TOP",
        rotate_gripper=False,
        reasoning="Object is to the left of the robot.",
    )


@pytest.fixture
def place_resolution():
    return PlaceDiscreteResolutionSchema(
        arm="LEFT",
        reasoning="Left arm is holding the object.",
    )
