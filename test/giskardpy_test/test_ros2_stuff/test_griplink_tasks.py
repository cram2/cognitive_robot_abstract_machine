"""
The griplink tasks a gripper configuration is reached with, skipped where
``griplink_interfaces`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pytest

pytest.importorskip("griplink_interfaces")

from griplink_interfaces.action import Flexgrip, Grip
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from typing_extensions import Any

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import NoGriplinkEndpoint
from giskardpy.motion_statechart.goals.gripper import MoveFingers
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.griplink import (
    GriplinkActionServerTask,
    GriplinkFlexActionServerTask,
    GriplinkPresetActionServerTask,
    GriplinkStatus,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperStateConfiguration,
)
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyLeftGripper
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkFlexConfiguration,
    GriplinkGripPreset,
    GriplinkPresetConfiguration,
)
from semantic_digital_twin.world import World

# %% gripper stand-in


class GriplinkActionName(StrEnum):
    """
    The action names the griplink stand-ins serve under.
    """

    GRIP = "gripper/grip"
    FLEXGRIP = "gripper/flexgrip"


@dataclass
class RecordingGriplinkServer:
    """
    Answers every goal of one griplink action with a fixed status and records the goals.
    """

    action: type
    """
    The griplink action the server executes.
    """

    status: GriplinkStatus
    """
    The status every result reports.
    """

    received_goals: list[Any] = field(default_factory=list)
    """
    The goals the server received, in order.
    """

    def execute(self, goal: ServerGoalHandle) -> Any:
        """
        Records the goal and finishes it with :attr:`status`.
        """
        self.received_goals.append(goal.request)
        goal.succeed()
        return self.action.Result(status=self.status)

    def serve(self, node: Node, action_topic: str) -> ActionServer:
        """
        :return: The action server answering goals on `action_topic` through
            :meth:`execute`.
        """
        return ActionServer(node, self.action, action_topic, self.execute)


def run_until_observed(
    task: GriplinkActionServerTask,
    observation: ObservationStateValues,
    rclpy_node: Node,
) -> None:
    """
    Ticks a statechart holding `task` until `task` observes `observation`.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(task)
    if observation == ObservationStateValues.TRUE:
        statechart.add_node(EndMotion.when_true(task))
    else:
        statechart.add_node(EndMotion.when_false(task))
    executor = StatechartExecutor(
        context=statechart.context,
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
        pacer=RealTimePacer(),
    )
    executor.compile(statechart=statechart)
    executor.tick_until_end()


# %% helpers


def left_gripper_of(world: World) -> DAiSyLeftGripper:
    """
    :return: The left gripper of the DAiSy in `world`.
    """
    return world.get_semantic_annotations_by_type(DAiSy)[0].left_arm.end_effector


# %% reaching a griplink configuration


def test_a_preset_configuration_is_reached_through_the_endpoint_of_its_state(
    daisy_world,
) -> None:
    gripper = left_gripper_of(daisy_world)
    configuration = GriplinkPresetConfiguration.from_state_type(
        gripper, GripperState.CLOSE, grip_preset=GriplinkGripPreset.PRESET_3
    )
    endpoint = gripper.griplink_endpoint(GripperState.CLOSE)

    task = GriplinkActionServerTask.moving_gripper_to(configuration)

    assert type(task) is GriplinkPresetActionServerTask
    assert (task.action_topic, task.message_type, task.grip_preset) == (
        endpoint.action_topic,
        endpoint.message_type,
        configuration.grip_preset,
    )


def test_a_flex_configuration_is_reached_with_its_hardware_parameters(
    daisy_world,
) -> None:
    gripper = left_gripper_of(daisy_world)
    configuration = GriplinkFlexConfiguration.from_state_type(
        gripper,
        GripperState.FLEXCLOSE,
        grip_position=0.05,
        grip_force=120.0,
        grip_velocity=0.1,
        grip_acceleration=1.0,
    )
    endpoint = gripper.griplink_endpoint(GripperState.FLEXCLOSE)

    task = GriplinkActionServerTask.moving_gripper_to(configuration)

    assert type(task) is GriplinkFlexActionServerTask
    assert (
        task.action_topic,
        task.message_type,
        task.grip_position,
        task.grip_force,
        task.grip_velocity,
        task.grip_acceleration,
    ) == (
        endpoint.action_topic,
        endpoint.message_type,
        configuration.grip_position,
        configuration.grip_force,
        configuration.grip_velocity,
        configuration.grip_acceleration,
    )


def test_a_configuration_of_no_griplink_kind_is_reached_with_a_finger_goal(
    daisy_world,
) -> None:
    configuration = GripperStateConfiguration.from_state_type(
        left_gripper_of(daisy_world), GripperState.CLOSE
    )

    goal = GriplinkActionServerTask.moving_gripper_to(configuration)

    assert type(goal) is type(MoveFingers.moving_gripper_to(configuration))


def test_a_state_served_by_an_action_of_the_other_kind_has_no_endpoint(
    daisy_world,
) -> None:
    gripper = left_gripper_of(daisy_world)
    flex_close_state = GriplinkFlexConfiguration.from_state_type(
        gripper, GripperState.FLEXCLOSE
    ).joint_state

    with pytest.raises(NoGriplinkEndpoint):
        GriplinkActionServerTask.moving_gripper_to(
            GriplinkPresetConfiguration(
                end_effector=gripper, joint_state=flex_close_state
            )
        )


# %% talking to the griplink action server


@pytest.mark.parametrize(
    "status, observation",
    [
        (GriplinkStatus.SUCCESS, ObservationStateValues.TRUE),
        (GriplinkStatus.AXIS_BLOCKED, ObservationStateValues.FALSE),
    ],
    ids=["succeeding", "failing"],
)
def test_task_observes_whether_the_server_reports_success(
    status: GriplinkStatus, observation: ObservationStateValues, rclpy_node: Node
) -> None:
    server = RecordingGriplinkServer(action=Grip, status=status).serve(
        rclpy_node, GriplinkActionName.GRIP
    )
    task = GriplinkPresetActionServerTask(
        action_topic=GriplinkActionName.GRIP, message_type=Grip
    )

    run_until_observed(task, observation, rclpy_node)
    server.destroy()

    assert task.on_tick() == observation


def test_preset_task_sends_its_preset(rclpy_node: Node) -> None:
    gripper = RecordingGriplinkServer(action=Grip, status=GriplinkStatus.SUCCESS)
    server = gripper.serve(rclpy_node, GriplinkActionName.GRIP)
    task = GriplinkPresetActionServerTask(
        action_topic=GriplinkActionName.GRIP,
        message_type=Grip,
        grip_preset=GriplinkGripPreset.PRESET_2,
    )

    run_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert [goal.index for goal in gripper.received_goals] == [task.grip_preset.value]


def test_flex_task_sends_its_parameters_in_micro_units(rclpy_node: Node) -> None:
    gripper = RecordingGriplinkServer(action=Flexgrip, status=GriplinkStatus.SUCCESS)
    server = gripper.serve(rclpy_node, GriplinkActionName.FLEXGRIP)
    task = GriplinkFlexActionServerTask(
        action_topic=GriplinkActionName.FLEXGRIP,
        message_type=Flexgrip,
        grip_position=0.05,
        grip_force=120.0,
        grip_velocity=0.1,
        grip_acceleration=1.0,
    )

    run_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert [
        (goal.position, goal.force, goal.speed, goal.acceleration)
        for goal in gripper.received_goals
    ] == [(50_000, 120_000, 100_000, 1_000_000)]
