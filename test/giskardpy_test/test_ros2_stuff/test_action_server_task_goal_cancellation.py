"""
The goal of an :class:`ActionServerTask` is cancelled when the task ends, pauses or is
reset, tested against an in-process action server that holds goals until they are
cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto

import pytest
from json_msgs.action import JsonAction
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from typing_extensions import Any, Callable, Iterator

import krrood.symbolic_math.symbolic_math as sm
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.monitors import CountTicks, Pulse
from cramph.node import EndedByOwner, StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.action_servers import (
    ActionServerTask,
    SentGoal,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from krrood.patterns.called_positionally import called_positionally
from semantic_digital_twin.world import World

# %% goal-holding stand-in


class GoalTopic(StrEnum):
    """
    The action topics the stand-in servers serve under.
    """

    HOLDING = "goal_holding_server"
    ANSWERING = "first_goal_answering_server"


class UnansweredGoalTreatment(Enum):
    """
    What a :class:`GoalHoldingServer` does with a goal it does not answer.
    """

    HELD = auto()
    """
    The server accepts the goal and keeps it active until it is cancelled.
    """

    LEFT_UNACCEPTED = auto()
    """
    The server neither accepts nor rejects the goal.
    """


@dataclass(kw_only=True)
class GoalHoldingServer:
    """
    Stands in for an action server: answers the first :attr:`answered_goal_count` goals
    with an empty result at once and treats every later one as
    :attr:`unanswered_goal_treatment` says.
    """

    action: type
    """
    The action the server serves.
    """

    answered_goal_count: int = 0
    """
    How many goals, counted from the first, the server answers.
    """

    unanswered_goal_treatment: UnansweredGoalTreatment = UnansweredGoalTreatment.HELD
    """
    What the server does with the goals after the answered ones.
    """

    received_goal_count: int = field(init=False, default=0)
    """
    How many goals the server received.
    """

    cancel_request_count: int = field(init=False, default=0)
    """
    How many cancel requests the server received for its goals.
    """

    @called_positionally
    async def decide_on_goal(self, goal_request: Any) -> GoalResponse:
        """
        Counts the goal and accepts it, except for an unanswered goal that is to stay
        unaccepted, whose decision never comes.
        """
        self.received_goal_count += 1
        if (
            self.received_goal_count > self.answered_goal_count
            and self.unanswered_goal_treatment
            is UnansweredGoalTreatment.LEFT_UNACCEPTED
        ):
            await Future()
        return GoalResponse.ACCEPT

    @called_positionally
    def answer_or_hold(self, goal_handle: ServerGoalHandle) -> None:
        """
        Answers the accepted goal if it is among the answered ones. A held goal is never
        executed, so it stays active.
        """
        if self.received_goal_count <= self.answered_goal_count:
            goal_handle.execute()

    @called_positionally
    def answer(self, goal_handle: ServerGoalHandle) -> Any:
        """
        Finishes the goal successfully with an empty result.
        """
        goal_handle.succeed()
        return self.action.Result()

    @called_positionally
    def count_cancel_request(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """
        Counts the cancel request and accepts it.
        """
        self.cancel_request_count += 1
        return CancelResponse.ACCEPT

    def serve(self, *, node: Node, action_topic: str) -> ActionServer:
        """
        :return: The action server answering goals on `action_topic`.
        """
        return ActionServer(
            node,
            self.action,
            action_topic,
            execute_callback=self.answer,
            goal_callback=self.decide_on_goal,
            handle_accepted_callback=self.answer_or_hold,
            cancel_callback=self.count_cancel_request,
        )


@pytest.fixture
def holding_server(rclpy_node: Node) -> Iterator[GoalHoldingServer]:
    """
    :return: A server holding every goal it receives.
    """
    server = GoalHoldingServer(action=JsonAction)
    action_server = server.serve(node=rclpy_node, action_topic=GoalTopic.HOLDING)
    yield server
    action_server.destroy()


@pytest.fixture
def answering_server(rclpy_node: Node) -> Iterator[GoalHoldingServer]:
    """
    :return: A server answering the first goal it receives and holding every later one.
    """
    server = GoalHoldingServer(action=JsonAction, answered_goal_count=1)
    action_server = server.serve(node=rclpy_node, action_topic=GoalTopic.ANSWERING)
    yield server
    action_server.destroy()


# %% statechart nodes of the scenarios


@dataclass(eq=False, repr=False, kw_only=True)
class ServerStateMonitor(EndedByOwner, StatechartNode):
    """
    Observes whether :attr:`condition` holds for a stand-in server.
    """

    condition: Callable[[], bool]
    """
    The condition on the server's counts.
    """

    def on_tick(self) -> ObservationStateValue:
        """
        :return: Whether :attr:`condition` holds.
        """
        if self.condition():
            return ObservationStateValue.TRUE
        return ObservationStateValue.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class ResultObservingTask(
    ActionServerTask[
        JsonAction, JsonAction.Goal, JsonAction.Result, JsonAction.Feedback
    ]
):
    """
    Sends an empty goal and observes True once any result arrived.
    """

    message_type: type[JsonAction] = field(default=JsonAction, kw_only=True)
    """
    The action the task sends its goal through.
    """

    def build_action_goal(self) -> None:
        self._action_goal = JsonAction.Goal()

    def on_tick(self) -> ObservationStateValue:
        """
        :return: True once a result arrived, Unknown until then.
        """
        if self._result is None:
            return ObservationStateValue.UNKNOWN
        return ObservationStateValue.TRUE


@dataclass(kw_only=True)
class GoalCancellationScenario:
    """
    A statechart holding one :class:`ActionServerTask` that sends its goals to a
    :class:`GoalHoldingServer`.
    """

    task: ActionServerTask
    """
    The task under test.
    """

    server: GoalHoldingServer
    """
    The server the task sends its goals to.
    """

    rclpy_node: Node
    """
    The ROS node the task's action client and the server live on.
    """

    statechart: Statechart = field(
        init=False,
        default_factory=lambda: Statechart(context=StatechartContext(world=World())),
    )
    """
    The statechart the task runs in.
    """

    def __post_init__(self) -> None:
        """
        Adds :attr:`task` to :attr:`statechart`.
        """
        self.statechart.add_node(node=self.task)

    def add_server_monitor(
        self, *, condition: Callable[[], bool]
    ) -> ServerStateMonitor:
        """
        :return: A new node of the statechart observing whether `condition` holds.
        """
        monitor = ServerStateMonitor(condition=condition)
        self.statechart.add_node(node=monitor)
        return monitor

    def add_one_tick_trigger(self, *, start_condition: sm.Scalar) -> Pulse:
        """
        :return: A new node of the statechart that observes True for one tick once
            `start_condition` holds, and False ever after.
        """
        trigger = Pulse()
        self.statechart.add_node(node=trigger)
        trigger.start_condition = start_condition
        return trigger

    def add_trigger_on_first_received_goal(self) -> Pulse:
        """
        :return: A new node of the statechart that observes True for one tick once the
            server received the task's first goal.
        """
        first_goal_received = self.add_server_monitor(
            condition=lambda: self.server.received_goal_count >= 1
        )
        return self.add_one_tick_trigger(
            start_condition=first_goal_received.observes_true
        )

    def run_until(self, *, condition: Callable[[], bool]) -> None:
        """
        Ticks the statechart until `condition` holds.
        """
        done = self.add_server_monitor(condition=condition)
        self.statechart.add_node(node=EndMotion.when_true(node=done))
        executor = StatechartExecutor(
            context=self.statechart.context,
            extensions=[RosNodeAccess(ros_node=self.rclpy_node), MotionControl()],
            pacer=RealTimePacer(),
        )
        executor.compile(statechart=self.statechart)
        executor.tick_until_end(timeout=200)


# %% cancelling the goal when the task ends, pauses or is reset


def test_interrupting_the_task_cancels_its_goal(
    holding_server: GoalHoldingServer, rclpy_node: Node
) -> None:
    """
    An interrupted task asks the server to cancel the goal it still pursues.
    """
    task = ResultObservingTask(action_topic=GoalTopic.HOLDING)
    scenario = GoalCancellationScenario(
        task=task, server=holding_server, rclpy_node=rclpy_node
    )
    task.interrupt_condition = (
        scenario.add_trigger_on_first_received_goal().observes_true
    )

    scenario.run_until(condition=lambda: holding_server.cancel_request_count >= 1)

    assert holding_server.received_goal_count == 1
    assert holding_server.cancel_request_count == 1


def test_pausing_the_task_cancels_its_goal_and_unpausing_sends_it_again(
    holding_server: GoalHoldingServer, rclpy_node: Node
) -> None:
    """
    A paused task cancels its goal, and sends it again once it continues.
    """
    task = ResultObservingTask(action_topic=GoalTopic.HOLDING)
    scenario = GoalCancellationScenario(
        task=task, server=holding_server, rclpy_node=rclpy_node
    )
    task.pause_condition = scenario.add_trigger_on_first_received_goal().observes_true

    scenario.run_until(
        condition=lambda: (
            holding_server.received_goal_count >= 2
            and holding_server.cancel_request_count >= 1
        )
    )

    assert holding_server.received_goal_count == 2
    assert holding_server.cancel_request_count == 1


def test_resetting_the_task_cancels_its_goal_before_it_sends_a_new_one(
    holding_server: GoalHoldingServer, rclpy_node: Node
) -> None:
    """
    A reset task cancels its goal, and sends a new one once it starts again.
    """
    task = ResultObservingTask(action_topic=GoalTopic.HOLDING)
    scenario = GoalCancellationScenario(
        task=task, server=holding_server, rclpy_node=rclpy_node
    )
    task.reset_condition = scenario.add_trigger_on_first_received_goal().observes_true

    scenario.run_until(
        condition=lambda: (
            holding_server.received_goal_count >= 2
            and holding_server.cancel_request_count >= 1
        )
    )

    assert holding_server.received_goal_count == 2
    assert holding_server.cancel_request_count == 1


def test_a_task_started_paused_sends_its_goal_only_once_it_continues(
    holding_server: GoalHoldingServer, rclpy_node: Node
) -> None:
    """
    A task paused from its start sends no goal while paused, and one goal once it
    continues.
    """
    task = ResultObservingTask(action_topic=GoalTopic.HOLDING)
    scenario = GoalCancellationScenario(
        task=task, server=holding_server, rclpy_node=rclpy_node
    )
    pause_over = CountTicks(ticks=10)
    scenario.statechart.add_node(node=pause_over)
    task.pause_condition = sm.logic_not(pause_over.observes_true)

    scenario.run_until(
        condition=lambda: (
            task.life_cycle_state == LifeCycleValue.RUNNING
            and holding_server.received_goal_count >= 1
        )
    )

    assert holding_server.received_goal_count == 1
    assert holding_server.cancel_request_count == 0


def test_a_restarted_task_waits_for_the_result_of_its_new_goal(
    answering_server: GoalHoldingServer, rclpy_node: Node
) -> None:
    """
    A task started again does not observe the result of the goal it sent before.
    """
    task = ResultObservingTask(action_topic=GoalTopic.ANSWERING)
    scenario = GoalCancellationScenario(
        task=task, server=answering_server, rclpy_node=rclpy_node
    )
    task.reset_condition = scenario.add_one_tick_trigger(
        start_condition=task.observes_true
    ).observes_true

    scenario.run_until(condition=lambda: answering_server.received_goal_count >= 2)

    assert task.observation_state == ObservationStateValue.UNKNOWN


# %% the goal's lifetime between sending and the server's answer


@dataclass(kw_only=True)
class GoalHandleCountingCancels:
    """
    Stands in for a client's handle of a goal, counting the cancel requests sent
    through it.
    """

    accepted: bool = True
    """
    Whether the server accepted the goal.
    """

    cancel_request_count: int = field(init=False, default=0)
    """
    How many cancel requests were sent through the handle.
    """

    def cancel_goal_async(self) -> None:
        """
        Counts one cancel request.
        """
        self.cancel_request_count += 1


@dataclass(kw_only=True)
class ResponseCounter:
    """
    Counts the server's responses a :class:`SentGoal` passed on.
    """

    passed_on_count: int = field(init=False, default=0)
    """
    How many responses were passed on.
    """

    def count(self) -> None:
        """
        Counts one response passed on.
        """
        self.passed_on_count += 1


def test_a_goal_cancelled_before_the_server_accepts_it_is_cancelled_once_accepted() -> (
    None
):
    """
    A goal cancelled while its acceptance is on the way is cancelled on the server once accepted.
    """
    goal = SentGoal()
    goal.cancel()
    goal_handle = GoalHandleCountingCancels()

    goal.receive_goal_response(goal_handle=goal_handle, record_response=lambda: None)

    assert goal_handle.cancel_request_count == 1


def test_the_responses_to_a_cancelled_goal_are_not_passed_on() -> None:
    """
    Neither the acceptance nor the result of a cancelled goal reaches the task.
    """
    goal = SentGoal()
    counter = ResponseCounter()
    goal.cancel()

    goal.receive_goal_response(
        goal_handle=GoalHandleCountingCancels(), record_response=counter.count
    )
    goal.receive_result(store_result=counter.count)

    assert counter.passed_on_count == 0


def test_cancelling_a_goal_twice_sends_one_cancel_request() -> None:
    """
    Cancelling a goal again, as resetting an ended task does, sends no second cancel
    request.
    """
    goal = SentGoal()
    goal_handle = GoalHandleCountingCancels()
    goal.receive_goal_response(goal_handle=goal_handle, record_response=lambda: None)

    goal.cancel()
    goal.cancel()

    assert goal_handle.cancel_request_count == 1


def test_a_goal_whose_result_arrived_is_not_cancelled_on_the_server() -> None:
    """
    A goal the server already answered is not cancelled there.
    """
    goal = SentGoal()
    goal_handle = GoalHandleCountingCancels()
    goal.receive_goal_response(goal_handle=goal_handle, record_response=lambda: None)
    goal.receive_result(store_result=lambda: None)

    goal.cancel()

    assert goal_handle.cancel_request_count == 0


def test_a_rejected_goal_is_not_cancelled_on_the_server() -> None:
    """
    A goal the server rejected is not cancelled there.
    """
    goal = SentGoal()
    goal.cancel()
    goal_handle = GoalHandleCountingCancels(accepted=False)

    goal.receive_goal_response(goal_handle=goal_handle, record_response=lambda: None)

    assert goal_handle.cancel_request_count == 0
