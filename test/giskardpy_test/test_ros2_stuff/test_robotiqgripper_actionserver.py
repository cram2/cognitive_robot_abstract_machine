"""
Test suite for RobotiqGripperActionServerTask using in-process ROS 2 action servers and
Statechart execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    RobotiqGripperActionServerTask,
)


# %% In-process Mock Action Server


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
        result.position = goal_handle.request.command.position
        result.effort = goal_handle.request.command.effort
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
) -> MotionStatechartContext:
    """
    Executes the task within a compiled statechart until it evaluates to
    target_observation.
    """
    context = StatechartContext(world=World())
    statechart = Statechart(context=context)
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
    return context


# %% Test Cases


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
    topic = "robotiq/gripper_cmd"
    mock_server = MockRobotiqActionServer(
        reached_goal=reached_goal,
        stalled=stalled,
        accept_goals=True,
    )
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        message_type=ParallelGripperCommand,
        target_position=0.03,
    )

    ctx = run_statechart_until_observed(task, expected_observation, rclpy_node)
    server.destroy()

    assert task.on_tick(ctx) == expected_observation


def test_robotiq_task_handles_goal_rejection(rclpy_node: Node) -> None:
    topic = "robotiq/gripper_cmd"
    mock_server = MockRobotiqActionServer(accept_goals=False)
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        message_type=ParallelGripperCommand,
        target_position=0.0,
    )

    ctx = run_statechart_until_observed(task, ObservationStateValues.FALSE, rclpy_node)
    server.destroy()

    assert task._done is True
    assert task._goal_accepted is False
    assert task.on_tick(ctx) == ObservationStateValues.FALSE


def test_robotiq_task_builds_and_sends_correct_goal_message(rclpy_node: Node) -> None:
    topic = "robotiq/gripper_cmd"
    mock_server = MockRobotiqActionServer(reached_goal=True)
    server = mock_server.serve(rclpy_node, topic)

    pos = 0.045
    vel = 12.0
    eff = 85.0

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        message_type=ParallelGripperCommand,
        target_position=pos,
        target_velocity=vel,
        target_effort=eff,
    )

    run_statechart_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert len(mock_server.received_goals) == 1
    received_cmd = mock_server.received_goals[0]
    assert received_cmd.position == [pos]
    assert received_cmd.velocity == [vel]
    assert received_cmd.effort == [eff]


def test_robotiq_task_uses_default_values(rclpy_node: Node) -> None:
    topic = "robotiq/gripper_cmd"
    mock_server = MockRobotiqActionServer(reached_goal=True)
    server = mock_server.serve(rclpy_node, topic)

    task = RobotiqGripperActionServerTask(
        action_topic=topic,
        message_type=ParallelGripperCommand,
        target_position=0.07,
    )

    run_statechart_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert len(mock_server.received_goals) == 1
    received_cmd = mock_server.received_goals[0]
    assert received_cmd.position == [0.07]
    assert received_cmd.velocity == [10.0]
    assert received_cmd.effort == [50.0]
