"""
Test suite for RobotiqGripperActionServerTask using in-process ROS 2 action servers and
Statechart execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("control_msgs")

from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.goals.gripper import MoveFingers
from giskardpy.motion_statechart.ros2_nodes.robotiq import (
    RobotiqGripperActionServerTask,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy


def unique_topic(prefix: str = "robotiq/gripper_cmd") -> str:
    """
    Generate a unique topic name per test to prevent DDS crosstalk.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# %% Mock In-Process Action Server


@dataclass
class MockRobotiqActionServer:
    """
    Mock ROS 2 Action Server for control_msgs/ParallelGripperCommand.

    Records incoming goals and returns configurable results.
    """

    reached_goal: bool = True
    stalled: bool = False
    accept_goals: bool = True
    received_goals: list[Any] = field(default_factory=list)

    def goal_callback(self, goal_request):
        from rclpy.action import GoalResponse

        return GoalResponse.ACCEPT if self.accept_goals else GoalResponse.REJECT

    def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> ParallelGripperCommand.Result:
        self.received_goals.append(goal_handle.request.command)
        goal_handle.succeed()

        result = ParallelGripperCommand.Result()
        result.reached_goal = self.reached_goal
        result.stalled = self.stalled
        return result

    def serve(self, node: Node, topic: str) -> ActionServer:
        return ActionServer(
            node,
            ParallelGripperCommand,
            topic,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
        )


def run_statechart_until_observed(
    task: RobotiqGripperActionServerTask,
    target_observation: ObservationStateValues,
    rclpy_node: Node,
) -> None:
    """
    Mounts task into a Statechart and executes it until target_observation is observed.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(task)

    if target_observation == ObservationStateValues.TRUE:
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


# %% reaching a gripper configuration


@pytest.mark.parametrize("state", [GripperState.OPEN, GripperState.CLOSE])
def test_a_configuration_is_reached_through_the_parallel_gripper_command_server_of_its_gripper(
    tracy_world: World, state: GripperState
) -> None:
    gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector
    configuration = gripper.default_configuration(state)

    task = RobotiqGripperActionServerTask.moving_gripper_to(configuration)

    assert type(task) is RobotiqGripperActionServerTask
    assert (task.action_topic, task.target_position) == (
        gripper.parallel_gripper_command_topic,
        gripper.separation_at(configuration.joint_state),
    )


def test_a_gripper_without_a_parallel_gripper_command_server_is_reached_with_a_finger_goal(
    pr2_world_state_reset: World,
) -> None:
    configuration = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[
        0
    ].left_arm.end_effector.default_configuration(GripperState.CLOSE)

    goal = RobotiqGripperActionServerTask.moving_gripper_to(configuration)

    assert type(goal) is type(MoveFingers.moving_gripper_to(configuration))


# %% Tests


@pytest.mark.parametrize(
    "reached_goal, stalled, expected_observation",
    [
        (True, False, ObservationStateValues.TRUE),
        (False, True, ObservationStateValues.TRUE),
        (False, False, ObservationStateValues.FALSE),
    ],
    ids=["reached_goal", "stalled_on_contact", "failed_movement"],
)
def test_robotiq_task_observes_result(
    reached_goal: bool,
    stalled: bool,
    expected_observation: ObservationStateValues,
    rclpy_node: Node,
) -> None:
    topic = unique_topic()
    mock_server = MockRobotiqActionServer(
        reached_goal=reached_goal,
        stalled=stalled,
        accept_goals=True,
    )
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        target_position=0.03,
    )

    try:
        run_statechart_until_observed(task, expected_observation, rclpy_node)
        assert task.on_tick() == expected_observation
    finally:
        server.destroy()


def test_robotiq_task_handles_goal_rejection(rclpy_node: Node) -> None:
    topic = unique_topic()
    mock_server = MockRobotiqActionServer(accept_goals=False)
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        target_position=0.0,
    )

    try:
        run_statechart_until_observed(task, ObservationStateValues.FALSE, rclpy_node)
        assert task._done is True
        assert task._goal_accepted is False
        assert task.on_tick() == ObservationStateValues.FALSE
    finally:
        server.destroy()


def test_robotiq_task_builds_and_sends_correct_goal_message(rclpy_node: Node) -> None:
    topic = unique_topic()
    mock_server = MockRobotiqActionServer(reached_goal=True)
    server = mock_server.serve(rclpy_node, topic)

    pos = 0.045
    vel = 12.0
    eff = 85.0

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        target_position=pos,
        target_velocity=vel,
        target_effort=eff,
    )

    try:
        run_statechart_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
        assert len(mock_server.received_goals) == 1
        received_cmd = mock_server.received_goals[0]
        # Convert array.array to list for comparison
        assert list(received_cmd.position) == [pos]
        assert list(received_cmd.velocity) == [vel]
        assert list(received_cmd.effort) == [eff]
    finally:
        server.destroy()


def test_robotiq_task_uses_default_values(rclpy_node: Node) -> None:
    topic = unique_topic()
    mock_server = MockRobotiqActionServer(reached_goal=True)
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        target_position=0.07,
    )

    try:
        run_statechart_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
        assert len(mock_server.received_goals) == 1
        received_cmd = mock_server.received_goals[0]
        assert list(received_cmd.position) == [0.07]
        assert list(received_cmd.velocity) == [10.0]
        assert list(received_cmd.effort) == [50.0]
    finally:
        server.destroy()
