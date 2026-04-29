from __future__ import annotations

import math
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from dataclasses import dataclass
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    FixedConnection,
)


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@pytest.fixture
def world_with_cup():
    """World containing a single pourable container with a tilt joint, filled to 100%."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            scale=Scale(0.4, 0.4, 1.0),
            initial_fill=1.0,
            k=1,
        )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return world, cup


class TestPouringTask:
    """Test suite for the PouringTask in Giskardpy."""

    def test_pouring_task_achieves_goal(self, world_with_cup, rclpy_node):
        """
        Test that PouringTask successfully tilts the cup and reduces fill level
        to the target value.
        """
        world, cup = world_with_cup

        goal_fill = 0.6
        tolerance = 0.05

        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            tolerance=tolerance,
            reference_velocity=0.05,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            MotionStatechartContext(world=world),
        )
        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=1000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level <= goal_fill + tolerance
        assert cup.fill_level >= goal_fill - tolerance
        assert cup.root.parent_connection.position > 0.1

    def test_pr2_pouring_from_gripper(self, pr2_world_setup, rclpy_node):
        """
        Test that PouringTask works when the cup is held by the PR2 robot.
        """
        world = pr2_world_setup

        # 2. Create a cup setup
        gripper_frame = world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )

        with world.modify_world():
            cup_body = Body(name=PrefixedName("cup"))
            world.add_body(cup_body)
            gripper_C_tilt = FixedConnection.create_with_dofs(
                world=world,
                parent=gripper_frame,
                child=cup_body,
                name=PrefixedName("gripper_T_cup_tilt"),
            )
            world.add_connection(gripper_C_tilt)

            _cup_height = 0.12
            _cup_half_width = 0.04
            cup_shape = Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=_cup_height / 2,
                    reference_frame=cup_body,
                ),
                scale=Scale(
                    2 * _cup_half_width,
                    2 * _cup_half_width,
                    _cup_height,
                ),
            )
            cup_body.visual = ShapeCollection(shapes=[cup_shape])
            cup_body.collision = ShapeCollection(shapes=[cup_shape])

        cup = PourableContainer(name=PrefixedName("cup"), root=cup_body)
        with world.modify_world():
            world.add_semantic_annotation(cup)

        # 3. Initialize fill level and equation
        cup.initialize_fill_level(
            world=world,
            initial_fill=1.0,
            k=1.0,
        )

        # 4. Run PouringTask
        goal_fill = 0.6
        tolerance = 0.05
        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup_body,
            goal_value=goal_fill,
            tolerance=tolerance,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            MotionStatechartContext(world=world),
        )
        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=900)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level == pytest.approx(goal_fill, abs=tolerance)
        assert cup.fill_equation.symbolic_velocity(
            cup.fill_connection.tilt_expression,
            cup.fill_connection.dof.variables.position,
        ).evaluate()[0] == pytest.approx(0.0, abs=1e-2)
