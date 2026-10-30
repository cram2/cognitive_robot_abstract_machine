"""
Tests for the exchange of child choices between a running statechart and the client's
copy of it, see :mod:`giskardpy.middleware.ros2.child_choices`, without a ROS graph.
"""

from __future__ import annotations

from cramph.data_types import StatechartPayloadKey
import json
from dataclasses import dataclass, field

import pytest
from typing_extensions import Any

from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
)
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import StatechartNode
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.reported_statechart import ReportedStatechart
from cramph.statechart import Statechart
from giskardpy.middleware.ros2.child_choices import (
    ChildChoiceClient,
    ChildChoiceMessage,
    ChildSentByClient,
)
from giskardpy.middleware.ros2.exceptions import (
    GiskardWorldUpdateNotReceivedError,
    StatechartOutOfStepError,
)
from giskardpy.middleware.ros2.feedback_publisher import (
    ActionFeedbackPublisher,
)
from giskardpy.middleware.ros2.world_updates import (
    ClientWorldUpdates,
    OutgoingWorldUpdates,
)
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition
from semantic_digital_twin.world import World

from .test_motion_server import PublicationProgressMimic
from .test_world_updates import PublishingSynchronizerMimic

# %% mimics


@dataclass
class ActionServerRunningOneGoal:
    """
    Stands in for the action server while one goal runs.
    """

    goal_id: int = 0
    """
    The goal that is running.
    """

    sent_feedback: list[Any] = field(default_factory=list)
    """
    Every feedback message that was sent.
    """

    def send_feedback(self, message: Any) -> None:
        self.sent_feedback.append(message)


@dataclass
class WorldUpdatesCaughtUpOnDemand:
    """
    Stands in for the incoming world updates, catching up with a client when told to.
    """

    caught_up: bool = True
    """
    Whether every position asked about counts as applied.
    """

    def has_applied(self, position: StreamPosition) -> bool:
        return self.caught_up


@dataclass
class ChooserGivingAFreshChild(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Gives every node that asks a new child that succeeds on the tick after it starts.
    """

    given_children: int = 0
    """
    How many children were given so far.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        self.given_children += 1
        return _node_succeeding_at_once(f"child {self.given_children}")


@dataclass
class ChooserGivingUpOnItsFirstChild(ChooserGivingAFreshChild):
    """
    Gives a child that never ends on its own first, with a failure monitor giving up on
    it after a few ticks, and a fresh child that succeeds at once after that.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        if self.given_children > 0:
            return super().choose_child(node)
        self.given_children += 1
        return NodeSucceedingOnObservingTrue(
            name="undecided", observation=ObservationStateValues.UNKNOWN
        )

    def failure_monitors_for(
        self,
        node: CompositeNodeChoosingItsChild,
        child: StatechartNode,
    ) -> list[StatechartNode]:
        if child.name != "undecided":
            return []
        return [CountTicks(name="timeout", ticks=2)]


# %% helpers


def _node_succeeding_at_once(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


@dataclass
class TwoChoosingNodes:
    """
    A statechart in which the choosing node with the higher index starts first.
    """

    statechart: Statechart
    """
    The statechart holding the nodes.
    """

    starting_late: CompositeNodeChoosingItsChild
    """
    The choosing node with the lower index, starting once :attr:`delay` succeeded.
    """

    starting_first: CompositeNodeChoosingItsChild
    """
    The choosing node with the higher index, starting with the statechart.
    """

    delay: NodeSucceedingOnObservingTrue
    """
    Succeeds on the tick after the statechart started.
    """

    @classmethod
    def in_context(cls, context: StatechartContext) -> TwoChoosingNodes:
        """
        :param context: The context the statechart is built in.
        """
        statechart = Statechart(context=context)
        starting_late = CompositeNodeChoosingItsChild(name="starting late")
        starting_first = CompositeNodeChoosingItsChild(name="starting first")
        delay = _node_succeeding_at_once("delay")
        statechart.add_nodes([starting_late, starting_first, delay])
        starting_late.start_condition = delay.is_succeeded
        return cls(
            statechart=statechart,
            starting_late=starting_late,
            starting_first=starting_first,
            delay=delay,
        )


@dataclass
class ClientAndRunningStatechart:
    """
    A client holding a copy of a statechart that runs elsewhere, and the two ends of the
    exchange of child choices between them.
    """

    world_updates: WorldUpdatesCaughtUpOnDemand = field(
        default_factory=WorldUpdatesCaughtUpOnDemand
    )
    """
    Tells the running side whether its world contains what the client chose on.
    """

    action_server: ActionServerRunningOneGoal = field(
        default_factory=ActionServerRunningOneGoal
    )
    """
    Tells both ends which goal is running.
    """

    client_chooser: ChildChooser = field(default_factory=ChooserGivingAFreshChild)
    """
    Chooses the children on the client's copy.
    """

    running_side_publications: PublicationProgressMimic = field(
        default_factory=PublicationProgressMimic
    )
    """
    Publishes the changes of the running side's world.
    """

    client_world_updates: ClientWorldUpdates = field(
        default_factory=lambda: ClientWorldUpdates(
            world_synchronizer=PublishingSynchronizerMimic()
        )
    )
    """
    Tells how far the client's world caught up with the changes of the running side.
    """

    outgoing_world_updates: OutgoingWorldUpdates = field(init=False)
    """
    Tells which changes of the running side's world its goal published.
    """

    feedback_publisher: ActionFeedbackPublisher = field(init=False)
    """
    Reports the running statechart to the client.
    """

    executor: StatechartExecutor = field(init=False)
    """
    Ticks the running statechart.
    """

    running: TwoChoosingNodes = field(init=False)
    """
    The statechart that runs.
    """

    copy: TwoChoosingNodes = field(init=False)
    """
    The client's copy of the running statechart.
    """

    receiving_chooser: ChildSentByClient = field(init=False)
    """
    Takes the client's choices on the running side.
    """

    client: ChildChoiceClient = field(init=False)
    """
    Chooses on the client's copy.
    """

    def __post_init__(self) -> None:
        self.receiving_chooser = ChildSentByClient(
            world_updates=self.world_updates, action_server=self.action_server
        )
        self.executor = StatechartExecutor(context=StatechartContext(world=World()))
        self.executor.context.add_extension(
            ChildChooserAccess(choosers=[self.receiving_chooser])
        )
        self.running = TwoChoosingNodes.in_context(self.executor.context)
        self.outgoing_world_updates = OutgoingWorldUpdates(
            world_synchronizer=self.running_side_publications
        )
        self.feedback_publisher = ActionFeedbackPublisher(
            executor=self.executor,
            action_server=self.action_server,
            outgoing_world_updates=self.outgoing_world_updates,
        )

        client_context = StatechartContext(world=World())
        client_context.add_extension(ChildChooserAccess(choosers=[self.client_chooser]))
        self.copy = TwoChoosingNodes.in_context(client_context)
        self.client = ChildChoiceClient(
            statechart=self.copy.statechart,
            child_chooser_access=ChildChooserAccess(choosers=[self.client_chooser]),
            world_updates=self.client_world_updates,
            required_position=StreamPosition(
                origin=MetaData(node_name="client", process_id=0), sequence_number=1
            ),
        )

    def start(self) -> None:
        """
        Accept the running statechart as a goal and compile it, which starts it.
        """
        self.outgoing_world_updates.start_goal()
        self.executor.compile(self.running.statechart)

    def let_the_client_catch_up(self) -> None:
        """
        Let every change the running side published so far reach the client's world.
        """
        self.client_world_updates.world_synchronizer.applied_sequence_number = (
            self.running_side_publications.published_sequence_number
        )

    def let_the_client_answer(self) -> list[ChildChoiceMessage]:
        """
        Hand the feedback of the running statechart to the client and the client's
        choices back, each as the JSON that travels between them.

        :return: The choices the client sent.
        """
        feedback = self.feedback_publisher.create_states()
        feedback[StatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        sent_choices = [
            ChildChoiceMessage.from_json(json.loads(json.dumps(choice.to_json())))
            for choice in self.client.answer(json.loads(json.dumps(feedback)))
        ]
        for choice in sent_choices:
            self.receiving_chooser.receive(choice)
        return sent_choices

    def let_both_nodes_be_answered_before_any_choice_is_taken(self) -> None:
        """
        Let the client answer both choosing nodes, the one starting first before the
        other, while the running side cannot take a choice yet.
        """
        self.world_updates.caught_up = False
        choices = self.let_the_client_answer()
        self.executor.tick()
        choices += self.let_the_client_answer()
        answered_nodes = [choice.node_index for choice in choices]
        assert answered_nodes == [
            self.copy.starting_first.index,
            self.copy.starting_late.index,
        ]
        self.world_updates.caught_up = True


@pytest.fixture()
def exchange_before_start() -> ClientAndRunningStatechart:
    """
    :return: A statechart that is about to run and a client answering its nodes waiting
        for a child.
    """
    return ClientAndRunningStatechart()


@pytest.fixture()
def exchange(
    exchange_before_start: ClientAndRunningStatechart,
) -> ClientAndRunningStatechart:
    """
    :return: A running statechart and a client answering its nodes waiting for a child.
    """
    exchange_before_start.start()
    return exchange_before_start


# %% choices are taken in the order the client made them


def test_choices_are_taken_in_the_order_the_client_made_them(
    exchange: ClientAndRunningStatechart,
) -> None:
    exchange.let_both_nodes_be_answered_before_any_choice_is_taken()

    exchange.executor.tick()
    exchange.executor.tick()

    assert [node.name for node in exchange.running.statechart.nodes] == [
        node.name for node in exchange.copy.statechart.nodes
    ]
    assert [child.name for child in exchange.running.starting_first.children] == [
        child.name for child in exchange.copy.starting_first.children
    ]
    assert [child.name for child in exchange.running.starting_late.children] == [
        child.name for child in exchange.copy.starting_late.children
    ]


def test_a_choice_made_later_waits_for_the_one_made_before_it(
    exchange: ClientAndRunningStatechart,
) -> None:
    exchange.let_both_nodes_be_answered_before_any_choice_is_taken()

    assert not exchange.receiving_chooser.has_choice_for(exchange.running.starting_late)


# %% a choice out of step is still rejected


def test_a_choice_skipping_nodes_the_statechart_never_got_is_rejected(
    exchange: ClientAndRunningStatechart,
) -> None:
    waiting_node = exchange.running.starting_first
    exchange.receiving_chooser.receive(
        ChildChoiceMessage(
            goal_id=exchange.action_server.goal_id,
            node_index=waiting_node.index,
            first_node_index=len(exchange.running.statechart.nodes) + 1,
            nodes={},
        )
    )

    with pytest.raises(StatechartOutOfStepError):
        exchange.executor.tick()


def test_a_choice_made_after_one_whose_node_ended_untaken_is_rejected(
    exchange_before_start: ClientAndRunningStatechart,
) -> None:
    exchange = exchange_before_start
    ended_node = exchange.running.starting_first
    ended_node.interrupt_condition = exchange.running.delay.is_succeeded
    exchange.start()
    exchange.let_both_nodes_be_answered_before_any_choice_is_taken()
    assert ended_node.life_cycle_state == LifeCycleValues.INTERRUPTED

    with pytest.raises(StatechartOutOfStepError):
        exchange.executor.tick()


# %% the failure monitors of a child travel with it


def test_the_running_side_gives_up_on_a_child_the_client_handed_failure_monitors() -> (
    None
):
    exchange = ClientAndRunningStatechart(
        client_chooser=ChooserGivingUpOnItsFirstChild()
    )
    exchange.start()

    for _ in range(10):
        exchange.let_the_client_answer()
        exchange.executor.tick()

    choosing_node = exchange.running.starting_first
    undecided, succeeding = choosing_node.chosen_children
    assert undecided.life_cycle_state == LifeCycleValues.FAILED
    assert [monitor.name for monitor in choosing_node.failure_reasons] == ["timeout"]
    assert succeeding.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% the client chooses on the changes the running side published


def test_the_client_chooses_only_once_its_world_holds_the_changes_of_the_goal(
    exchange: ClientAndRunningStatechart,
) -> None:
    """
    A choice reads the client's world, which learns of a change of the running side, a
    body attached to a gripper, say, on another channel than the request to choose.
    """
    exchange.running_side_publications.publish_one_update()

    answered_before_catching_up = exchange.let_the_client_answer()
    exchange.let_the_client_catch_up()
    [answer] = exchange.let_the_client_answer()

    assert answered_before_catching_up == []
    assert answer.node_index == exchange.copy.starting_first.index


def test_the_client_does_not_wait_for_changes_published_before_the_goal(
    exchange_before_start: ClientAndRunningStatechart,
) -> None:
    """
    A client that fetched the world instead of receiving these changes would never catch
    up with them.
    """
    exchange = exchange_before_start
    exchange.running_side_publications.publish_one_update()
    exchange.start()

    [answer] = exchange.let_the_client_answer()

    assert answer.node_index == exchange.copy.starting_first.index


def test_a_client_whose_world_does_not_catch_up_reports_it(
    exchange: ClientAndRunningStatechart,
) -> None:
    exchange.client_world_updates.timeout = 0.0
    exchange.running_side_publications.publish_one_update()

    with pytest.raises(GiskardWorldUpdateNotReceivedError) as raised:
        exchange.let_the_client_answer()

    assert raised.value.awaited_sequence_number == (
        exchange.running_side_publications.published_sequence_number
    )


# %% a node waiting for a child is asked about again


def test_the_running_side_asks_again_while_a_node_waits_for_a_child(
    exchange: ClientAndRunningStatechart,
) -> None:
    """
    A client that could not answer yet is only asked again by another feedback, which is
    otherwise sent only when the states change.
    """
    exchange.feedback_publisher.child_request_repeat_period = 0.0

    exchange.feedback_publisher.publish_if_changed()
    exchange.feedback_publisher.publish_if_changed()

    assert len(exchange.action_server.sent_feedback) == 2


def test_the_running_side_asks_again_only_once_the_repeat_period_passed(
    exchange: ClientAndRunningStatechart,
) -> None:
    exchange.feedback_publisher.child_request_repeat_period = 60.0

    exchange.feedback_publisher.publish_if_changed()
    exchange.feedback_publisher.publish_if_changed()

    assert len(exchange.action_server.sent_feedback) == 1


# %% the statechart rebuilt from the feedback


def test_the_statechart_rebuilt_from_the_feedback_holds_the_children_that_joined(
    exchange: ClientAndRunningStatechart,
) -> None:
    exchange.feedback_publisher.publish_structure()
    exchange.let_both_nodes_be_answered_before_any_choice_is_taken()
    exchange.executor.tick()
    exchange.executor.tick()
    exchange.feedback_publisher.publish()

    reported_statechart = ReportedStatechart()
    for message in exchange.action_server.sent_feedback:
        reported_statechart.take_over(json.loads(message.feedback))

    assert [node.name for node in reported_statechart.statechart.nodes] == [
        node.name for node in exchange.running.statechart.nodes
    ]


def test_the_running_side_repeats_nothing_while_no_node_waits_for_a_child() -> None:
    action_server = ActionServerRunningOneGoal()
    executor = StatechartExecutor(context=StatechartContext(world=World()))
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        NodeSucceedingOnObservingTrue(
            name="undecided", observation=ObservationStateValues.UNKNOWN
        )
    )
    executor.compile(statechart)
    feedback_publisher = ActionFeedbackPublisher(
        executor=executor,
        action_server=action_server,
        outgoing_world_updates=OutgoingWorldUpdates(
            world_synchronizer=PublicationProgressMimic()
        ),
        child_request_repeat_period=0.0,
    )

    feedback_publisher.publish_if_changed()
    feedback_publisher.publish_if_changed()

    assert len(action_server.sent_feedback) == 1
