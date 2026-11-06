import numpy as np
import pytest

from cramph.composites import Parallel
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.goals.gripper import (
    MoveFingers,
    MoveFingersToJointState,
    MoveGripper,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.cartesian_tasks import DistanceAlongAxis
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.world import World

from ..motion_control_context import create_context_with_motion_control

# %% reading the gripper out of a world


def _left_gripper(world: World) -> HasTwoFingers:
    """
    :return: The end effector of the left arm of the robot in ``world``.
    """
    return world.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector


def _expanded_nodes(goal: MoveFingers, world: World):
    """
    :return: The children the goal is made of, once it joined a statechart.
    """
    Statechart(context=create_context_with_motion_control(world)).add_node(goal)
    return goal.nodes


def _half_open(gripper: HasTwoFingers, **parameters) -> MoveGripper:
    """
    :return: A goal standing the fingertips of `gripper` half as wide apart as it opens.
    """
    return MoveGripper(
        gripper=gripper, width=gripper.separation_when_open / 2, **parameters
    )


def _descendants(nodes):
    """
    :return: ``nodes`` and, recursively, the nodes any composite among them holds.
    """
    found = []
    for node in nodes:
        found.append(node)
        if isinstance(node, Parallel):
            found.extend(_descendants(node.nodes))
    return found


# %% commanding the fingers


def test_gripper_is_given_the_width_between_its_facing_points(pr2_world_copy) -> None:
    """
    The distance between the fingertips is what Giskard is given, so it finds the joint
    positions that realise it rather than being told them.
    """
    gripper = _left_gripper(pr2_world_copy)
    goal = _half_open(gripper)

    nodes = _expanded_nodes(goal, pr2_world_copy)

    [width_task] = nodes
    assert isinstance(width_task, DistanceAlongAxis)
    assert width_task.reference_point.reference_frame is gripper.finger.tip
    assert width_task.tip_point.reference_frame is gripper.thumb.tip
    assert width_task.distance == goal.width
    assert width_task.threshold == goal.threshold


def test_fingers_driven_to_a_joint_state_are_given_that_state(pr2_world_copy) -> None:
    goal_state = _left_gripper(pr2_world_copy).get_joint_state_by_type(
        GripperState.CLOSE
    )
    goal = MoveFingersToJointState(goal_state=goal_state)

    nodes = _expanded_nodes(goal, pr2_world_copy)

    [joint_goal] = nodes
    assert isinstance(joint_goal, JointPositionList)
    assert joint_goal.goal_state == goal_state


def test_gripper_commands_nothing_besides_the_fingers_by_default(
    pr2_world_copy,
) -> None:
    """
    A caller that asks for nothing in particular gets the finger task alone: no speed
    cap and no stall alternative.
    """
    goal = _half_open(_left_gripper(pr2_world_copy))

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    assert [type(node) for node in nodes] == [DistanceAlongAxis]


# %% settling for stalled fingers


def test_gripper_without_stall_tolerance_only_ends_on_the_finger_task(
    pr2_world_copy,
) -> None:
    """
    Stall tolerance is off unless asked for, so nothing watches the fingers for standing
    still.
    """
    goal = _half_open(_left_gripper(pr2_world_copy))

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    assert not any(isinstance(node, LocalMinimumReached) for node in nodes)


def test_gripper_tolerating_stall_settles_for_either_outcome(pr2_world_copy) -> None:
    """
    Fingers that close on an object stop short of their goal, so with stall tolerance
    the finger task and the stall monitor sit under one goal that either of them can
    satisfy.
    """
    goal = _half_open(_left_gripper(pr2_world_copy), tolerate_stall=True)

    nodes = _expanded_nodes(goal, pr2_world_copy)

    alternative = next(node for node in nodes if isinstance(node, Parallel))
    assert alternative.minimum_success == 1
    assert {type(node) for node in alternative.nodes} == {
        DistanceAlongAxis,
        LocalMinimumReached,
    }


def test_gripper_watches_only_the_fingers_for_stalling(pr2_world_copy) -> None:
    """
    The stall monitor judges the fingers, not the whole robot, so the rest of the body
    moving cannot keep it from reporting settled fingers.
    """
    gripper = _left_gripper(pr2_world_copy)
    goal = _half_open(gripper, tolerate_stall=True)

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    monitor = next(node for node in nodes if isinstance(node, LocalMinimumReached))
    assert monitor.degrees_of_freedom == [
        connection.raw_dof for connection in gripper.finger_connections
    ]
    assert monitor.measure_from_own_start


def test_gripper_keeps_the_monitors_dwell_time_when_none_is_given(
    pr2_world_copy,
) -> None:
    """
    Leaving the dwell time unset keeps whatever the monitor itself defaults to, so the
    default lives in one place.
    """
    goal = _half_open(_left_gripper(pr2_world_copy), tolerate_stall=True)

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    monitor = next(node for node in nodes if isinstance(node, LocalMinimumReached))
    assert monitor.minimum_time == LocalMinimumReached.minimum_time


def test_gripper_dwells_as_long_as_asked_before_accepting_a_stall(
    pr2_world_copy,
) -> None:
    """
    A caller that wants the fingers to stand still longer before the stall counts gets
    that time on the monitor.
    """
    goal = _half_open(
        _left_gripper(pr2_world_copy), tolerate_stall=True, stall_minimum_time=2.5
    )

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    monitor = next(node for node in nodes if isinstance(node, LocalMinimumReached))
    assert monitor.minimum_time == goal.stall_minimum_time


# %% capping the finger speed


def test_gripper_speed_cap_limits_the_finger_joints(pr2_world_copy) -> None:
    """
    A finger speed applies to the joints that move the fingers, and to no others.
    """
    gripper = _left_gripper(pr2_world_copy)
    goal = _half_open(gripper, finger_velocity=0.01)

    nodes = _descendants(_expanded_nodes(goal, pr2_world_copy))

    limit = next(node for node in nodes if isinstance(node, JointVelocityLimit))
    assert limit.max_velocity == goal.finger_velocity
    assert limit.connections == gripper.finger_connections


def test_gripper_speed_cap_holds_whichever_way_the_goal_ends(pr2_world_copy) -> None:
    """
    The stall alternative decides when the goal is done; the speed cap is not one of the
    things that can satisfy it, so it stays outside that alternative and keeps holding
    while the fingers move.
    """
    goal = _half_open(
        _left_gripper(pr2_world_copy), tolerate_stall=True, finger_velocity=0.01
    )

    nodes = _expanded_nodes(goal, pr2_world_copy)

    alternative = next(node for node in nodes if isinstance(node, Parallel))
    assert not any(isinstance(node, JointVelocityLimit) for node in alternative.nodes)
    assert any(isinstance(node, JointVelocityLimit) for node in nodes)


# %% running it


def _run(goal: MoveFingers, world: World) -> None:
    """
    Tick `goal` in `world` until it reached its goal.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(goal)
    statechart.add_node(EndMotion.when_true(goal))

    executor.compile(statechart)
    executor.tick_until_end(timeout=500)


def test_gripper_stands_its_fingertips_the_width_apart(pr2_world_state_reset) -> None:
    gripper = _left_gripper(pr2_world_state_reset)
    goal = _half_open(gripper)

    _run(goal, pr2_world_state_reset)

    assert gripper.fingertips.separation == pytest.approx(
        goal.width, abs=goal.threshold
    )


def test_fingers_reach_the_commanded_joint_state(pr2_world_state_reset) -> None:
    """
    Ticking the goal drives the fingers to the positions the commanded state holds.
    """
    goal_state = _left_gripper(pr2_world_state_reset).get_joint_state_by_type(
        GripperState.OPEN
    )

    _run(MoveFingersToJointState(goal_state=goal_state), pr2_world_state_reset)

    reached = [connection.position for connection in goal_state.connections]
    np.testing.assert_allclose(reached, goal_state.target_values, atol=1e-3)
