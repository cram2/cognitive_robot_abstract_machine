"""
The Franka gripper tasks against action servers standing in for the gripper, skipped
where ``franka_msgs`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pytest

pytest.importorskip("franka_msgs")

from franka_msgs.action import Grasp, Homing, Move
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from typing_extensions import Any, Iterator

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.middleware.ros2 import franka_gripper_tasks
from giskardpy.middleware.ros2.franka_gripper_tasks import (
    FrankaGripperTask,
    GripperGraspTask,
    GripperHomingTask,
    GripperMoveTask,
)
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.world import World

from .test_node_json_round_trips import receiving_executor, sending_context
from ...statechart_json_round_trip import (
    NodeRoundTripCase,
    NodeRoundTripTests,
    concrete_node_types,
)

# %% gripper stand-in


class GripperActionName(StrEnum):
    """
    The action names the gripper stand-ins serve under.
    """

    MOVE = "gripper/move"
    GRASP = "gripper/grasp"
    HOMING = "gripper/homing"


@dataclass
class RecordingGripper:
    """
    Answers every goal of one gripper action with a fixed success and records the goals.
    """

    action: type
    """
    The franka_msgs action the gripper serves.
    """

    succeeds: bool
    """
    The success every result reports.
    """

    received_goals: list[Any] = field(default_factory=list)
    """
    The goals the gripper received, in order.
    """

    def execute(self, goal: ServerGoalHandle) -> Any:
        """
        Records the goal and finishes it with :attr:`succeeds`.
        """
        self.received_goals.append(goal.request)
        goal.succeed()
        return self.action.Result(success=self.succeeds)

    def serve(self, node: Node, action_topic: str) -> ActionServer:
        """
        :return: The action server answering goals on `action_topic` through
            :meth:`execute`.
        """
        return ActionServer(node, self.action, action_topic, self.execute)


def run_until_observed(
    task: FrankaGripperTask,
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


@pytest.fixture(params=[True, False], ids=["succeeding", "failing"])
def homing_gripper(
    request: pytest.FixtureRequest, rclpy_node: Node
) -> Iterator[RecordingGripper]:
    gripper = RecordingGripper(action=Homing, succeeds=request.param)
    server = gripper.serve(rclpy_node, GripperActionName.HOMING)
    yield gripper
    server.destroy()


# %% observation


def test_task_observes_the_success_the_gripper_reports(
    homing_gripper: RecordingGripper, rclpy_node: Node
) -> None:
    task = GripperHomingTask(action_topic=GripperActionName.HOMING)
    expected = (
        ObservationStateValues.TRUE
        if homing_gripper.succeeds
        else ObservationStateValues.FALSE
    )

    run_until_observed(task, expected, rclpy_node)

    assert task.on_tick() == expected


# %% goals


def test_move_task_sends_its_width_and_speed(rclpy_node: Node) -> None:
    gripper = RecordingGripper(action=Move, succeeds=True)
    server = gripper.serve(rclpy_node, GripperActionName.MOVE)
    task = GripperMoveTask(action_topic=GripperActionName.MOVE, width=0.04, speed=0.1)

    run_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert [(goal.width, goal.speed) for goal in gripper.received_goals] == [
        (task.width, task.speed)
    ]


def test_grasp_task_sends_its_parameters(rclpy_node: Node) -> None:
    gripper = RecordingGripper(action=Grasp, succeeds=True)
    server = gripper.serve(rclpy_node, GripperActionName.GRASP)
    task = GripperGraspTask(
        action_topic=GripperActionName.GRASP,
        width=0.02,
        speed=0.1,
        force=20.0,
        inner_tolerance=0.005,
        outer_tolerance=0.01,
    )

    run_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert [
        (
            goal.width,
            goal.speed,
            goal.force,
            goal.epsilon.inner,
            goal.epsilon.outer,
        )
        for goal in gripper.received_goals
    ] == [
        (
            task.width,
            task.speed,
            task.force,
            task.inner_tolerance,
            task.outer_tolerance,
        )
    ]


# %% round trips through JSON


def _cases() -> list[NodeRoundTripCase]:
    """
    :return: One case per Franka gripper task.
    """
    return [
        NodeRoundTripCase(
            GripperMoveTask,
            lambda world: [
                GripperMoveTask(
                    action_topic=GripperActionName.MOVE, width=0.04, speed=0.1
                )
            ],
        ),
        NodeRoundTripCase(
            GripperGraspTask,
            lambda world: [
                GripperGraspTask(
                    action_topic=GripperActionName.GRASP,
                    width=0.02,
                    speed=0.1,
                    force=20.0,
                    inner_tolerance=0.005,
                    outer_tolerance=0.01,
                )
            ],
        ),
        NodeRoundTripCase(
            GripperHomingTask,
            lambda world: [GripperHomingTask(action_topic=GripperActionName.HOMING)],
        ),
    ]


@pytest.fixture()
def gripper_action_servers(rclpy_node: Node) -> Iterator[list[ActionServer]]:
    """
    :return: The action servers the gripper tasks wait for before they are built.
    """
    servers = [
        RecordingGripper(action=Move, succeeds=True).serve(
            rclpy_node, GripperActionName.MOVE
        ),
        RecordingGripper(action=Grasp, succeeds=True).serve(
            rclpy_node, GripperActionName.GRASP
        ),
        RecordingGripper(action=Homing, succeeds=True).serve(
            rclpy_node, GripperActionName.HOMING
        ),
    ]
    yield servers
    for server in servers:
        server.destroy()


@pytest.fixture(params=_cases(), ids=lambda case: case.node_type.__name__)
def case(request: pytest.FixtureRequest) -> NodeRoundTripCase:
    return request.param


@pytest.fixture()
def cases() -> list[NodeRoundTripCase]:
    return _cases()


@pytest.fixture()
def node_types_needing_a_case() -> set[type[StatechartNode]]:
    return concrete_node_types(franka_gripper_tasks)


@pytest.mark.usefixtures("gripper_action_servers")
class TestFrankaGripperTaskRoundTrips(NodeRoundTripTests):
    pass
