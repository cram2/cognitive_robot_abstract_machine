"""
Tests of pose goals on Tracy, whose arm joints turn through more than a full turn.

A revolute joint turned by a full turn leaves the robot in the same geometric
configuration, but from a wound up start a local controller drives the joint into its
position limit instead of turning it the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import (
    NoProgressError,
    PositionLimitBlockedError,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import (
    BlockedPositionLimit,
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
from semantic_digital_twin.world_description.degree_of_freedom import PositionLimitSide
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

from ...dataset.wound_up_wrist import WristWinding

# %% a wrist turn watched for a stall


@dataclass(kw_only=True)
class WatchedWristTurn:
    """
    A compiled quarter turn of Tracy's right tool about its own axis, together with a
    stall monitor that cancels it.
    """

    executor: StatechartExecutor
    """
    The executor the turn is compiled in.
    """

    goal: CartesianPose
    """
    The pose goal that turns the tool.
    """

    progressing: StillProgressing
    """
    The stall monitor watching :attr:`goal`.
    """

    @classmethod
    def from_winding(
        cls,
        *,
        world: World,
        winding: WristWinding,
        arm_pose: dict[TracyJoint, float],
    ) -> WatchedWristTurn:
        """
        :param world: Tracy's world, which the arm is moved in.
        :param winding: Where the last wrist joint starts. From
            :attr:`WristWinding.WOUND_UP` the turn takes it past its upper position
            limit.
        :param arm_pose: The pose of the rest of the right arm.
        """
        start = arm_pose | {TracyJoint.RIGHT_WRIST_3: winding.value}
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
        progressing = StillProgressing(
            monitored_node=goal, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_nodes(nodes=[goal, progressing])
        motion_statechart.add_node(node=EndMotion.when_true(node=goal))
        motion_statechart.add_node(node=progressing.cancel_motion())
        executor.compile(statechart=motion_statechart)
        return cls(executor=executor, goal=goal, progressing=progressing)


# %% pose goals from a wound up wrist


def test_wrist_turn_from_an_unwound_wrist_is_reached(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    From :attr:`WristWinding.UNWOUND`, the wrist turns to the goal without being
    cancelled.
    """
    wrist_turn = WatchedWristTurn.from_winding(
        world=tracy_world_copy, winding=WristWinding.UNWOUND, arm_pose=tracy_arm_pose
    )

    wrist_turn.executor.tick_until_end(timeout=1_000)


def test_wrist_turn_from_a_wound_up_wrist_names_the_blocked_limit(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    From :attr:`WristWinding.WOUND_UP`, the wrist stalls at its upper position limit,
    and the motion is cancelled naming that limit and the orientation task driving the
    joint into it.
    """
    wrist_turn = WatchedWristTurn.from_winding(
        world=tracy_world_copy, winding=WristWinding.WOUND_UP, arm_pose=tracy_arm_pose
    )
    wrist = tracy_world_copy.get_connection_by_name(TracyJoint.RIGHT_WRIST_3.value)
    (orientation_task,) = [
        node for node in wrist_turn.goal.nodes if isinstance(node, CartesianOrientation)
    ]

    with pytest.raises(PositionLimitBlockedError) as exception_info:
        wrist_turn.executor.tick_until_end(timeout=1_000)

    assert exception_info.value.blocked_position_limits == [
        BlockedPositionLimit(
            degree_of_freedom=wrist.raw_dof,
            side=PositionLimitSide.UPPER,
            task=orientation_task,
        )
    ]


def test_a_slope_no_steeper_than_the_negligible_fraction_does_not_block(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    At the stall from :attr:`WristWinding.WOUND_UP`, a slope counts as zero up to the
    given fraction of the task's steepest slope, so a fraction of one blames no limit.
    """
    wrist_turn = WatchedWristTurn.from_winding(
        world=tracy_world_copy, winding=WristWinding.WOUND_UP, arm_pose=tracy_arm_pose
    )
    with pytest.raises(PositionLimitBlockedError):
        wrist_turn.executor.tick_until_end(timeout=1_000)

    assert (
        wrist_turn.progressing.find_blocked_position_limits(
            negligible_slope_fraction=1.0
        )
        == []
    )


def test_a_limit_of_a_joint_without_hardware_interface_is_not_blamed(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    A degree of freedom without a hardware interface, such as a drawer of the
    environment, is never named as blocking, even when a stalled task drives it into a
    position limit it lies at.
    """
    wrist = tracy_world_copy.get_connection_by_name(TracyJoint.RIGHT_WRIST_3.value)
    with tracy_world_copy.modify_world():
        tracy_world_copy.set_dofs_has_hardware_interface([wrist.raw_dof], False)
    wrist_turn = WatchedWristTurn.from_winding(
        world=tracy_world_copy, winding=WristWinding.WOUND_UP, arm_pose=tracy_arm_pose
    )

    with pytest.raises(NoProgressError) as exception_info:
        wrist_turn.executor.tick_until_end(timeout=1_000)

    assert type(exception_info.value) is NoProgressError
