import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from cramph.composites import ChildChooserAccess
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import NoPacing, StatechartExecutor
from cramph.monitors import CountSimulationTimeSeconds, CountTicks
from cramph.node import StatechartNode
from cramph.state_history import StateHistory, StateHistoryObserver
from cramph.statechart import Statechart
from giskardpy.middleware.ros2.action_server import GoalOutcome
from giskardpy.middleware.ros2.child_choices import (
    ChildChoiceClient,
    ChildChoiceMessage,
    ChildSentByClient,
)
from giskardpy.middleware.ros2.exceptions import StateUpdateAheadOfStatechartError
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.middleware.ros2.state_updates import StatechartStateUpdate
from giskardpy.middleware.ros2.world_updates import OutgoingWorldUpdates
from giskardpy.motion_statechart.graph_node import EndMotion
from semantic_digital_twin.adapters.ros.messages import MetaData
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.world import World

from .test_child_choices import (
    ChooserAnsweringInTurn,
    ChoosingGoal,
    client_world_updates,
)
from .test_motion_server import (
    GoalQueueMimic,
    PublicationProgressMimic,
    MotionServerFixture,
    feedback_data,
    motion_server,
)

# %% mimics


@dataclass
class ObserverRecordingTicks(StateHistoryObserver):
    """
    Records the tick of every snapshot a history tells it about.
    """

    ticks: list[int] = field(default_factory=list)
    """
    The ticks of the snapshots, in the order they were recorded.
    """

    def on_state_change(self, history: StateHistory) -> None:
        self.ticks.append(history.latest_item.tick_count)


@dataclass
class ObserverRecordingLifeCycles(StateHistoryObserver):
    """
    Records the life cycle state one node has in every snapshot a history tells it
    about.
    """

    node: StatechartNode
    """
    The node whose life cycle state is recorded.
    """

    life_cycle_states: list[LifeCycleValues] = field(default_factory=list)
    """
    The life cycle state of :attr:`node` in every snapshot, in the order they were
    recorded.
    """

    statechart_ended: list[bool] = field(default_factory=list)
    """
    Whether the statechart of :attr:`node` had ended in every snapshot, in the order
    they were recorded.
    """

    def on_state_change(self, history: StateHistory) -> None:
        self.life_cycle_states.append(history.latest_item.life_cycle_state[self.node])
        self.statechart_ended.append(self.node.statechart.is_ended())


@dataclass
class ClientFollowingTheStates(InputSynchronizer):
    """
    Stands in for a client that follows the states of the running goal in every
    feedback, and answers the nodes waiting for a child after that, as the client of
    Giskard does, from inside the control loop.
    """

    statechart: Statechart | None = None
    """
    The client's copy of the statechart sent as the goal.
    """

    action_server: GoalQueueMimic | None = None
    """
    The action server whose feedback is read.
    """

    child_choice_client: ChildChoiceClient | None = None
    """
    Decides the children to send, None for a goal without nodes choosing a child.
    """

    server_chooser: ChildSentByClient | None = None
    """
    The server side the choices are sent to.
    """

    _read_feedback_count: int = field(default=0, init=False)
    """
    How many feedback messages were followed so far.
    """

    def apply(self) -> bool:
        messages = self.action_server.feedback_messages
        for message in messages[self._read_feedback_count :]:
            self._follow(feedback_data(message))
        self._read_feedback_count = len(messages)
        return False

    def _follow(self, feedback: dict) -> None:
        """
        :param feedback: One feedback of the running goal.
        """
        StatechartStateUpdate.from_json(feedback, statechart=self.statechart).apply()
        if self.child_choice_client is None:
            return
        for choice in self.child_choice_client.answer(feedback):
            self.server_chooser.receive(
                ChildChoiceMessage.from_json(json.loads(json.dumps(choice.to_json())))
            )


# %% helpers


def _statechart_counting_ticks(context: StatechartContext) -> Statechart:
    """
    :return: A statechart in `context` whose two nodes succeed after different numbers
        of ticks, built the same way for Giskard's copy and the client's.
    """
    statechart = Statechart(context=context)
    statechart.add_nodes([CountTicks(ticks=1), CountTicks(ticks=5)])
    return statechart


def _reported_states(ticks: int) -> dict:
    """
    :param ticks: How often Giskard's copy is ticked before its states are reported.
    :return: The states Giskard reports after `ticks` ticks.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=World()), pacer=NoPacing()
    )
    executor.compile(_statechart_counting_ticks(executor.context))
    for _ in range(ticks):
        executor.tick()
    return ActionFeedbackPublisher(
        executor=executor,
        action_server=GoalQueueMimic(),
        outgoing_world_updates=OutgoingWorldUpdates(
            world_synchronizer=PublicationProgressMimic()
        ),
    ).create_states()


@pytest.fixture()
def client_statechart() -> Statechart:
    """
    :return: The client's copy of the statechart Giskard runs.
    """
    return _statechart_counting_ticks(StatechartContext(world=World()))


# %% applying reported states


def test_the_reported_states_are_written_into_the_client_statechart(
    client_statechart: Statechart,
) -> None:
    payload = _reported_states(ticks=2)
    update = StatechartStateUpdate.from_json(payload, statechart=client_statechart)

    update.apply()

    assert np.array_equal(
        client_statechart.life_cycle_state.data, update.life_cycle_state.data
    )
    assert np.array_equal(
        client_statechart.observation_state.data, update.observation_state.data
    )
    assert np.array_equal(
        client_statechart.last_observation_state.data,
        update.last_observation_state.data,
    )


def test_a_node_giskard_has_not_added_yet_keeps_its_state(
    client_statechart: Statechart,
) -> None:
    payload = _reported_states(ticks=2)
    client_statechart.add_node(chosen := CountTicks(ticks=1))

    StatechartStateUpdate.from_json(payload, statechart=client_statechart).apply()

    assert chosen.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_states_of_more_nodes_than_the_client_holds_are_rejected() -> None:
    payload = _reported_states(ticks=2)
    client_statechart = Statechart(context=StatechartContext(world=World()))
    client_statechart.add_node(CountTicks(ticks=1))

    with pytest.raises(StateUpdateAheadOfStatechartError):
        StatechartStateUpdate.from_json(payload, statechart=client_statechart).apply()


def test_the_observers_of_the_client_statechart_learn_the_reported_tick(
    client_statechart: Statechart,
) -> None:
    observer = ObserverRecordingTicks()
    client_statechart.state_history.add_observer(observer)
    update = StatechartStateUpdate.from_json(
        _reported_states(ticks=3), statechart=client_statechart
    )

    update.apply()

    assert observer.ticks == [update.tick_count]


def test_the_same_states_reported_twice_are_recorded_once(
    client_statechart: Statechart,
) -> None:
    payload = _reported_states(ticks=3)

    StatechartStateUpdate.from_json(payload, statechart=client_statechart).apply()
    StatechartStateUpdate.from_json(payload, statechart=client_statechart).apply()

    assert len(client_statechart.state_history.items) == 1


# %% following a running goal


def test_the_client_sees_a_node_run_before_the_goal_ends(
    motion_server: MotionServerFixture,
) -> None:
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(counter := CountSimulationTimeSeconds(seconds=0.1))
    statechart.add_node(EndMotion.when_true(counter))
    observer = ObserverRecordingLifeCycles(node=counter)
    statechart.state_history.add_observer(observer)
    following = ClientFollowingTheStates(
        world=motion_server.executor.context.world,
        statechart=statechart,
        action_server=motion_server.action_server,
    )
    motion_server.control_loop.inputs.synchronizers.append(following)
    motion_server.action_server.goal_json = json.dumps(
        MotionGoal.for_motion_statechart(
            statechart, client=MetaData(node_name="unwatched_client", process_id=0)
        ).to_json()
    )

    motion_server.motion_server.run_idle_cycle()
    following.apply()

    assert (LifeCycleValues.RUNNING, False) in zip(
        observer.life_cycle_states, observer.statechart_ended
    )
    assert observer.statechart_ended[-1]


def test_the_client_follows_a_child_it_chose_once_giskard_runs_it(
    motion_server: MotionServerFixture,
) -> None:
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
    goal = ChoosingGoal.ending_when(EndMotion.when_true)
    child = CountSimulationTimeSeconds(seconds=0.1)
    following = ClientFollowingTheStates(
        world=motion_server.executor.context.world,
        statechart=goal.statechart,
        action_server=motion_server.action_server,
        child_choice_client=ChildChoiceClient(
            statechart=goal.statechart,
            child_chooser_access=ChildChooserAccess(
                choosers=[ChooserAnsweringInTurn([child])]
            ),
            world_updates=client_world_updates(),
        ),
        server_chooser=motion_server.executor.context.require_extension(
            ChildChooserAccess
        ).choosers[0],
    )
    motion_server.control_loop.inputs.synchronizers.append(following)
    motion_server.action_server.goal_json = goal.goal_json()

    motion_server.motion_server.run_idle_cycle()
    following.apply()

    assert motion_server.action_server.outcome == GoalOutcome.SUCCEEDED
    assert np.array_equal(
        goal.statechart.life_cycle_state.data,
        motion_server.executor.statechart.life_cycle_state.data,
    )
