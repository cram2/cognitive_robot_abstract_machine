"""
Tests for a running goal whose nodes choose their child on the client, see
:mod:`giskardpy.middleware.ros2.child_choices`.
"""

from cramph.data_types import StatechartPayloadKey
import json
from dataclasses import dataclass, field

import pytest

from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
)
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.monitors import CountSimulationTimeSeconds
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.middleware.ros2.action_server import GoalOutcome
from giskardpy.middleware.ros2.child_choices import (
    ChildChoiceClient,
    ChildChoiceMessage,
    ChildSentByClient,
)
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from giskardpy.middleware.ros2.exceptions import StatechartOutOfStepError
from giskardpy.middleware.ros2.feedback_publisher import MotionStatechartPayloadKey
from semantic_digital_twin.input_synchronization import InputSynchronizer
from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.middleware.ros2.world_updates import ClientWorldUpdates
from giskardpy.motion_statechart.graph_node import EndMotion
from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.ros.messages import MetaData
from semantic_digital_twin.world import World

from .test_world_updates import PublishingSynchronizerMimic
from .test_motion_server import (
    CycleWatchingGoalCanceler,
    GoalQueueMimic,
    MotionServerFixture,
    feedback_data,
    motion_server,
)

# %% mimics


@dataclass
class ChooserAnsweringInTurn(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Gives the prepared children one after another, then says no child is left.
    """

    children: list[StatechartNode | None]
    """
    The children still to give, in order, None for no child left.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        if not self.children:
            return None
        return self.children.pop(0)


@dataclass
class ClientAnsweringFromTheControlLoop(InputSynchronizer):
    """
    Stands in for a client that reads the feedback of the running goal and sends the
    children it chooses, answering from inside the control loop.
    """

    client: ChildChoiceClient | None = None
    """
    Decides what to send for the feedback.
    """

    action_server: GoalQueueMimic | None = None
    """
    The action server whose feedback is read.
    """

    server_chooser: ChildSentByClient | None = None
    """
    The server side the choices are sent to.
    """

    sent_messages: list[ChildChoiceMessage] = field(default_factory=list)
    """
    Every message the client sent.
    """

    def apply(self) -> bool:
        if not self.action_server.feedback_messages:
            return False
        feedback = feedback_data(self.action_server.feedback_messages[-1])
        for message in self.client.answer(feedback):
            sent = ChildChoiceMessage.from_json(
                json.loads(json.dumps(message.to_json()))
            )
            self.sent_messages.append(sent)
            self.server_chooser.receive(sent)
        return False


# %% helpers


def client_world_updates() -> ClientWorldUpdates:
    """
    :return: The world updates of a client whose world published no change and applied
        none of Giskard's.
    """
    return ClientWorldUpdates(world_synchronizer=PublishingSynchronizerMimic())


@dataclass
class ChoosingGoal:
    """
    A goal whose root chooses its child, as the client holds it.
    """

    statechart: Statechart
    """
    The client's statechart the goal was built from.
    """

    choosing_node: CompositeNodeChoosingItsChild
    """
    The client's node choosing its child.
    """

    @classmethod
    def ending_when(cls, end_node_factory) -> "ChoosingGoal":
        """
        :param end_node_factory: Builds the node ending the statechart from the
            choosing node.
        :return: A goal running one choosing node until `end_node_factory` ends it.
        """
        context = StatechartContext(world=World())
        context.add_extension(ChildChooserAccess(choosers=[ChooserAnsweringInTurn([])]))
        statechart = Statechart(context=context)
        choosing_node = CompositeNodeChoosingItsChild(name="choosing")
        statechart.add_node(choosing_node)
        statechart.add_node(end_node_factory(choosing_node))
        return cls(statechart=statechart, choosing_node=choosing_node)

    def goal_json(self) -> str:
        return json.dumps(
            MotionGoal.for_motion_statechart(
                self.statechart,
                client=MetaData(node_name="unwatched_client", process_id=0),
            ).to_json()
        )


def _server_chooser(motion_server: MotionServerFixture) -> ChildSentByClient:
    """
    :return: The chooser of the server's statechart context.
    """
    [chooser] = motion_server.executor.context.require_extension(
        ChildChooserAccess
    ).choosers
    return chooser


def _client_answering(
    motion_server: MotionServerFixture,
    goal: ChoosingGoal,
    children: list[StatechartNode | None],
) -> ClientAnsweringFromTheControlLoop:
    """
    Let a client answer the running goal from inside the control loop.
    """
    answering = ClientAnsweringFromTheControlLoop(
        world=motion_server.executor.context.world,
        client=ChildChoiceClient(
            statechart=goal.statechart,
            child_chooser_access=ChildChooserAccess(
                choosers=[ChooserAnsweringInTurn(children)]
            ),
            world_updates=client_world_updates(),
        ),
        action_server=motion_server.action_server,
        server_chooser=_server_chooser(motion_server),
    )
    motion_server.control_loop.inputs.synchronizers.append(answering)
    return answering


def _cancel_after(motion_server: MotionServerFixture, ticks: int) -> None:
    """
    Cancel the running goal once `ticks` control cycles passed.
    """
    motion_server.control_loop.inputs.synchronizers.append(
        CycleWatchingGoalCanceler(
            world=motion_server.executor.context.world,
            cycle_counter=motion_server.cycle_counter,
            action_server=motion_server.action_server,
            ticks_until_cancel=ticks,
        )
    )


def _server_choosing_node(
    motion_server: MotionServerFixture, goal: ChoosingGoal
) -> CompositeNodeChoosingItsChild:
    return motion_server.executor.statechart.get_node_by_index(goal.choosing_node.index)


@pytest.fixture()
def choosing_motion_server(motion_server: MotionServerFixture) -> MotionServerFixture:
    """
    :return: A motion server whose nodes choose the children a client sends.
    """
    motion_server.executor.context.add_extension(
        ChildChooserAccess(
            choosers=[
                ChildSentByClient(
                    world_updates=motion_server.world_updates,
                    action_server=motion_server.action_server,
                )
            ]
        )
    )
    return motion_server


# %% the server runs what the client chooses


def test_a_child_the_client_chooses_runs_and_the_goal_succeeds(
    choosing_motion_server: MotionServerFixture,
) -> None:
    server = choosing_motion_server
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    child = CountSimulationTimeSeconds(seconds=0.1)
    _client_answering(server, goal, [child])
    server.action_server.goal_json = goal.goal_json()

    server.motion_server.run_idle_cycle()

    assert server.action_server.outcome == GoalOutcome.SUCCEEDED
    server_node = _server_choosing_node(server, goal)
    assert [type(node) for node in server_node.children] == [type(child)]
    assert server_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_the_feedback_names_the_nodes_waiting_for_a_child(
    choosing_motion_server: MotionServerFixture,
) -> None:
    server = choosing_motion_server
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    _cancel_after(server, ticks=3)
    server.action_server.goal_json = goal.goal_json()

    server.motion_server.run_idle_cycle()

    waiting = feedback_data(server.action_server.feedback_messages[-1])[
        MotionStatechartPayloadKey.WAITING_FOR_CHILD
    ]
    assert waiting == {str(goal.choosing_node.index): 0}


def test_no_child_left_fails_the_node(
    choosing_motion_server: MotionServerFixture,
) -> None:
    server = choosing_motion_server
    goal = ChoosingGoal.ending_when(EndMotion.when_failed)
    _client_answering(server, goal, [None])
    server.action_server.goal_json = goal.goal_json()

    server.motion_server.run_idle_cycle()

    assert server.action_server.outcome == GoalOutcome.SUCCEEDED
    server_node = _server_choosing_node(server, goal)
    assert server_node.children == []
    assert server_node.life_cycle_state == LifeCycleValues.FAILED


def test_a_child_out_of_step_with_the_statechart_aborts_the_goal(
    choosing_motion_server: MotionServerFixture,
) -> None:
    server = choosing_motion_server
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    _cancel_after(server, ticks=10)
    server.action_server.goal_json = goal.goal_json()
    chooser = _server_chooser(server)
    chooser.receive(
        ChildChoiceMessage(
            goal_id=0,
            node_index=goal.choosing_node.index,
            first_node_index=len(goal.statechart.nodes) + 1,
            nodes={},
        )
    )

    server.motion_server.run_idle_cycle()

    assert server.action_server.outcome == GoalOutcome.ABORTED
    error = from_json(json.loads(server.action_server.sent_results[0].result)["error"])
    assert isinstance(error, StatechartOutOfStepError)


def test_a_choice_for_another_goal_is_not_taken(
    choosing_motion_server: MotionServerFixture,
) -> None:
    server = choosing_motion_server
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    _cancel_after(server, ticks=3)
    server.action_server.goal_json = goal.goal_json()
    _server_chooser(server).receive(
        ChildChoiceMessage(
            goal_id=7,
            node_index=goal.choosing_node.index,
            first_node_index=len(goal.statechart.nodes),
            nodes=None,
        )
    )

    server.motion_server.run_idle_cycle()

    assert server.action_server.outcome == GoalOutcome.CANCELED
    assert _server_choosing_node(server, goal).life_cycle_state == (
        LifeCycleValues.RUNNING
    )


# %% the client answers each request once


def test_the_client_answers_a_waiting_node_once_per_child_it_holds() -> None:
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    client = ChildChoiceClient(
        statechart=goal.statechart,
        child_chooser_access=ChildChooserAccess(
            choosers=[ChooserAnsweringInTurn([CountSimulationTimeSeconds(seconds=0.1)])]
        ),
        world_updates=client_world_updates(),
    )
    feedback = {
        StatechartPayloadKey.GOAL_ID: 3,
        MotionStatechartPayloadKey.WAITING_FOR_CHILD: {
            str(goal.choosing_node.index): 0
        },
    }

    first_answers = client.answer(feedback)
    repeated_answers = client.answer(feedback)

    [message] = first_answers
    assert repeated_answers == []
    assert message.goal_id == 3
    assert message.node_index == goal.choosing_node.index
    assert message.first_node_index == len(goal.statechart.nodes) - 1
    assert goal.choosing_node.children == [goal.statechart.nodes[-1]]
