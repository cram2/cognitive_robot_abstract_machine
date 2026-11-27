"""
Test suite for PR2GripperActionServerTask using in-process ROS 2 action servers and
Statechart execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

pytest.importorskip("pr2_controllers_msgs")

from pr2_controllers_msgs.action import Pr2GripperCommand
from pr2_controllers_msgs.msg import Pr2GripperCommand as Pr2GripperCommandMessage
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.pr2_gripper import (
    PR2GripperActionServerTask,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from giskardpy.motion_statechart.tasks.gripper import MoveFingers
from krrood.patterns.called_positionally import called_positionally
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World

# %% helpers


def unique_topic() -> str:
    """
    :return: A topic name of its own per test, so the tests' servers do not meet.
    """
    return f"pr2_gripper/gripper_action_{uuid.uuid4().hex[:8]}"


@dataclass(kw_only=True)
class RecordingGripperServer:
    """
    Serves ``pr2_controllers_msgs/Pr2GripperCommand``, records the commands it receives
    and answers each with the configured result.
    """

    reached_goal: bool = True
    """
    Whether the result reports the gap reached.
    """

    stalled: bool = False
    """
    Whether the result reports the fingers stalled.
    """

    accept_goals: bool = True
    """
    Whether the server accepts goals.
    """

    received_commands: list[Pr2GripperCommandMessage] = field(default_factory=list)
    """
    The commands of the goals the server executed, in order.
    """

    @called_positionally
    def respond_to_goal(self, goal_request: Pr2GripperCommand.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT if self.accept_goals else GoalResponse.REJECT

    @called_positionally
    def execute(self, goal_handle: ServerGoalHandle) -> Pr2GripperCommand.Result:
        self.received_commands.append(goal_handle.request.command)
        goal_handle.succeed()
        result = Pr2GripperCommand.Result()
        result.reached_goal = self.reached_goal
        result.stalled = self.stalled
        return result

    def serve(self, *, node: Node, topic: str) -> ActionServer:
        """
        :return: The action server serving `topic` on `node`.
        """
        return ActionServer(
            node,
            Pr2GripperCommand,
            topic,
            execute_callback=self.execute,
            goal_callback=self.respond_to_goal,
        )


def run_statechart_until_observed(
    *,
    task: PR2GripperActionServerTask,
    target_observation: ObservationStateValue,
    rclpy_node: Node,
) -> None:
    """
    Runs `task` in a statechart until it observes `target_observation`.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(node=task)
    if target_observation == ObservationStateValue.TRUE:
        statechart.add_node(node=EndMotion.when_true(node=task))
    else:
        statechart.add_node(node=EndMotion.when_false(node=task))
    executor = StatechartExecutor(
        context=statechart.context,
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl()],
        pacer=RealTimePacer(),
    )
    executor.compile(statechart=statechart)
    executor.tick_until_end()


# %% reaching a gripper configuration


@pytest.mark.parametrize("state", [GripperState.OPEN, GripperState.CLOSE])
def test_a_configuration_is_reached_through_the_action_of_its_gripper(
    pr2_world_state_reset: World, state: GripperState
) -> None:
    [pr2] = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)
    for arm in pr2.all_arms:
        gripper = arm.end_effector
        configuration = gripper.default_configuration(state)

        task = PR2GripperActionServerTask.create_gripper_goal(
            configuration=configuration
        )

        assert type(task) is PR2GripperActionServerTask
        assert (task.action_topic, task.target_gap) == (
            gripper.gripper_action_name,
            gripper.target_gap(joint_state=configuration.joint_state),
        )


def test_a_gripper_of_another_robot_is_reached_with_a_finger_goal(
    tracy_world: World,
) -> None:
    configuration = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector.default_configuration(GripperState.CLOSE)

    goal = PR2GripperActionServerTask.create_gripper_goal(configuration=configuration)

    assert type(goal) is type(
        MoveFingers.create_gripper_goal(configuration=configuration)
    )


# %% the answers of the server


@pytest.mark.parametrize(
    "reached_goal, stalled, expected_observation",
    [
        (True, False, ObservationStateValue.TRUE),
        (False, True, ObservationStateValue.TRUE),
        (False, False, ObservationStateValue.FALSE),
    ],
    ids=["reached_goal", "stalled_on_contact", "failed_movement"],
)
def test_the_task_observes_the_result(
    reached_goal: bool,
    stalled: bool,
    expected_observation: ObservationStateValue,
    rclpy_node: Node,
) -> None:
    topic = unique_topic()
    server = RecordingGripperServer(reached_goal=reached_goal, stalled=stalled)
    action_server = server.serve(node=rclpy_node, topic=topic)
    task = PR2GripperActionServerTask(action_topic=topic, target_gap=0.03)

    try:
        run_statechart_until_observed(
            task=task, target_observation=expected_observation, rclpy_node=rclpy_node
        )
        assert task.on_tick() == expected_observation
    finally:
        action_server.destroy()


def test_a_rejected_goal_is_observed_as_false(rclpy_node: Node) -> None:
    topic = unique_topic()
    server = RecordingGripperServer(accept_goals=False)
    action_server = server.serve(node=rclpy_node, topic=topic)
    task = PR2GripperActionServerTask(action_topic=topic, target_gap=0.0)

    try:
        run_statechart_until_observed(
            task=task,
            target_observation=ObservationStateValue.FALSE,
            rclpy_node=rclpy_node,
        )
        assert task.on_tick() == ObservationStateValue.FALSE
        assert server.received_commands == []
    finally:
        action_server.destroy()


# %% the goal sent


def test_the_goal_carries_the_gap_and_the_effort(rclpy_node: Node) -> None:
    topic = unique_topic()
    server = RecordingGripperServer()
    action_server = server.serve(node=rclpy_node, topic=topic)
    task = PR2GripperActionServerTask(
        action_topic=topic, target_gap=0.045, max_effort=25.0
    )

    try:
        run_statechart_until_observed(
            task=task,
            target_observation=ObservationStateValue.TRUE,
            rclpy_node=rclpy_node,
        )
        [command] = server.received_commands
        assert (command.position, command.max_effort) == (
            task.target_gap,
            task.max_effort,
        )
    finally:
        action_server.destroy()
