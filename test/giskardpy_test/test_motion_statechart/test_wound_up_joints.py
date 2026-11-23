"""
Tests of pose goals on Tracy, whose arm joints turn through more than a full turn.

A revolute joint turned by a full turn leaves the robot in the same geometric
configuration, but from a wound up start a local controller drives the joint into its
position limit instead of turning it the other way round.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from enum import Enum

import numpy as np
import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import PositionLimitBlockedError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import (
    BlockedPositionLimit,
    PositionLimitSide,
    StillProgressing,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointState
from krrood_spatial_math.spatial_types import Pose, Vector3
from semantic_digital_twin.robots.tracy import TracyJoint
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

from .test_translation_rotation_balance import TRACY_ARM_POSE

# %% starts of the wrist


class WristWinding(float, Enum):
    """
    Where the last wrist joint of Tracy's right arm starts, in radians.

    Both members put the arm in the same geometric configuration; they differ by a full
    turn of the joint, whose position limits are a full turn either side of zero.
    """

    UNWOUND = -0.5
    """
    Close to zero, a full turn away from either position limit.
    """

    WOUND_UP = 2 * np.pi - 0.5
    """
    Just short of the upper position limit.
    """


@pytest.fixture()
def tracy_world_copy(tracy_world: World) -> World:
    """
    A copy of Tracy's world that a test may move the robot in.
    """
    return deepcopy(tracy_world)


def wrist_turn_watched_for_a_stall(
    *, world: World, winding: WristWinding
) -> tuple[StatechartExecutor, CartesianPose]:
    """
    Compile a quarter turn of Tracy's right tool about its own axis, started from
    `winding`, together with a stall monitor that cancels it.

    From :attr:`WristWinding.WOUND_UP` the turn takes the last wrist joint past its
    upper position limit.

    :return: The compiled executor and the pose goal it runs.
    """
    start = TRACY_ARM_POSE | {TracyJoint.RIGHT_WRIST_3: winding.value}
    JointState.from_str_dict(start, world=world).apply_to(world)
    tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    goal = CartesianPose(
        kinematic_chain=KinematicChain(root=world.root, tip=tip),
        goal_pose=Pose.from_xyz_axis_angle(
            axis=Vector3(z=1), angle=np.pi / 2, reference_frame=tip
        ),
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    motion_statechart = Statechart(context=executor.context)
    progressing = StillProgressing(monitored_node=goal, timeout=timedelta(seconds=1))
    motion_statechart.add_nodes(nodes=[goal, progressing])
    motion_statechart.add_node(node=EndMotion.when_true(node=goal))
    motion_statechart.add_node(node=progressing.cancel_motion())
    executor.compile(statechart=motion_statechart)
    return executor, goal


# %% pose goals from a wound up wrist


def test_wrist_turn_from_an_unwound_wrist_is_reached(tracy_world_copy: World) -> None:
    """
    From :attr:`WristWinding.UNWOUND`, the wrist turns to the goal without being cancelled.
    """
    executor, _ = wrist_turn_watched_for_a_stall(
        world=tracy_world_copy, winding=WristWinding.UNWOUND
    )

    executor.tick_until_end(timeout=1_000)


def test_wrist_turn_from_a_wound_up_wrist_names_the_blocked_limit(
    tracy_world_copy: World,
) -> None:
    """
    From :attr:`WristWinding.WOUND_UP`, the wrist stalls at its upper position limit,
    and the motion is cancelled naming that limit and the orientation task driving the
    joint into it.
    """
    executor, goal = wrist_turn_watched_for_a_stall(
        world=tracy_world_copy, winding=WristWinding.WOUND_UP
    )
    wrist = tracy_world_copy.get_connection_by_name(TracyJoint.RIGHT_WRIST_3.value)
    (orientation_task,) = [
        node for node in goal.nodes if isinstance(node, CartesianOrientation)
    ]

    with pytest.raises(PositionLimitBlockedError) as exception_info:
        executor.tick_until_end(timeout=1_000)

    assert exception_info.value.blocked_position_limits == [
        BlockedPositionLimit(
            degree_of_freedom=wrist.dof,
            side=PositionLimitSide.UPPER,
            task=orientation_task,
        )
    ]
