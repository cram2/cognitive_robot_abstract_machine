"""
Tests for the motion statechart templates that try alternatives, ``TryAll`` and
``TryInOrder``, and for the goals that run a node under a monitor.

The templates are exercised by compiling them into a real :class:`Statechart` and
ticking the executor, asserting the resulting observation and life cycle states.
Alternatives observe a fixed value and fail on observing False, so they are
deterministic children that always arrive / fail.
"""

import pytest
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    ChildCannotFailError,
    CompositeNodeWithoutChildrenError,
)
from cramph.composites import Sequence, TryAll, TryInOrder
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.statechart import Statechart
from cramph.monitors import CountTicks, Pulse
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeFailingOnObservingFalse,
    NodeObservingNothingYet,
)
from cramph.composites import PausedUntilTrue, PausedWhileTrue, StoppedWhenTrue
from semantic_digital_twin.world import World

from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

# Number of ticks after which the templates below have settled into their final observation.
SETTLE_TICKS = 6


def _compile_and_tick(
    goal: MotionStatechartNode, ticks: int = SETTLE_TICKS
) -> StatechartExecutor:
    """
    Add the goal to a fresh statechart, compile it and tick the executor.

    :param goal: The template under test.
    :param ticks: Control cycles to run.
    :return: The executor, so a caller can keep ticking and inspect intermediate states.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=World()), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(goal)
    executor.compile(statechart=statechart)
    for _ in range(ticks):
        executor.tick()
    return executor


def _failing(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: An alternative that fails on the first tick it runs.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.FALSE
    )


def _arriving(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: An alternative that is at its goal from the first tick it runs, and could
        fail, so a template accepts it anywhere.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.TRUE
    )


def _ticks_until_observed_true(
    goal: MotionStatechartNode, node: MotionStatechartNode, max_ticks: int
) -> int:
    """
    Compile `goal` and tick until `node` observes True.

    :return: The number of ticks that took.
    """
    executor = _compile_and_tick(goal, ticks=0)
    for tick in range(1, max_ticks + 1):
        executor.tick()
        if node.observation_state == ObservationStateValues.TRUE:
            return tick
    raise AssertionError(f"{node.name} never observed True within {max_ticks} ticks")


# %% TryAll, parallel and succeeding if any child succeeds


def test_try_all_succeeds_if_any_child_succeeds():
    stuck = _failing("a")
    arriving = _arriving("b")
    goal = TryAll(nodes=[stuck, arriving])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE
    assert arriving.life_cycle_state == LifeCycleValues.SUCCEEDED
    # Alternatives run side by side, so the other one is not cut short by the success.
    assert stuck.life_cycle_state != LifeCycleValues.NOT_STARTED


def test_try_all_fails_only_if_all_children_fail():
    goal = TryAll(
        nodes=[
            _failing("a"),
            _failing("b"),
        ]
    )
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.FALSE


def test_try_all_single_child():
    goal = TryAll(nodes=[_arriving("only")])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE


# %% TryInOrder, sequential and short-circuiting on the first success


def test_try_in_order_short_circuits_on_first_success():
    first = _arriving("first")
    second = _failing("second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE
    # First child succeeded and finished...
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    # ...so the second child is never started (short-circuit).
    assert second.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_try_in_order_advances_after_failure():
    first = _failing("first")
    second = _arriving("second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE
    # Both children ran: the first failed, the second was started and succeeded.
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_try_in_order_fails_only_if_all_children_fail():
    first = _failing("first")
    second = _failing("second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.FALSE
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.FAILED


def test_try_in_order_single_child():
    goal = TryInOrder(nodes=[_arriving("only")])
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE


# %% when the next alternative takes over


def test_the_next_alternative_starts_on_the_cycle_the_previous_one_fails():
    """
    An alternative waits for its predecessor's outcome, which it reads on the cycle that
    outcome is reached, so no control cycle passes with neither of them running.
    """
    first = _failing("first")
    second = _arriving("second")
    goal = TryInOrder(nodes=[first, second])
    executor = _compile_and_tick(goal, ticks=0)

    for _ in range(SETTLE_TICKS):
        executor.tick()
        if first.life_cycle_state == LifeCycleValues.FAILED:
            break

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.RUNNING


# %% goals built without children


def test_a_try_all_without_nodes_is_rejected() -> None:
    executor = StatechartExecutor(
        context=StatechartContext(world=World()), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(TryAll(nodes=[]))

    with pytest.raises(CompositeNodeWithoutChildrenError):
        executor.compile(statechart=statechart)


def test_a_try_in_order_without_nodes_is_rejected() -> None:
    executor = StatechartExecutor(
        context=StatechartContext(world=World()), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(TryInOrder(nodes=[]))

    with pytest.raises(CompositeNodeWithoutChildrenError):
        executor.compile(statechart=statechart)


# %% alternatives that cannot fail


def test_a_try_in_order_rejects_a_plain_task_before_its_last_alternative() -> None:
    """
    A plain task has no way of failing, so the alternatives after it could never start.
    """
    task = ConstFalseNode(name="first")
    goal = TryInOrder(nodes=[task, _arriving("second")])

    with pytest.raises(ChildCannotFailError) as error:
        _compile_and_tick(goal, ticks=0)

    assert error.value.node is goal
    assert error.value.child is task


def test_a_try_in_order_accepts_a_plain_task_as_its_last_alternative() -> None:
    goal = TryInOrder(
        nodes=[
            _failing("first"),
            ConstTrueNode(name="last"),
        ]
    )
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.TRUE


def test_a_try_in_order_accepts_a_plain_task_that_fails_on_its_own() -> None:
    """
    A task declaring its own failure has a way to fail, so the next alternative is
    reachable.
    """
    first = ConstTrueNode(name="first")
    first.fail_condition = first.observes_true
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert goal.last_observation_state == ObservationStateValues.TRUE


def test_a_try_in_order_accepts_a_monitored_node_that_is_stopped_before_its_last_alternative() -> (
    None
):
    """
    A monitored template fails once its monitor stopped the monitored node, which is a
    way for it to fail.
    """
    first = StoppedWhenTrue(
        name="first",
        monitor=ConstTrueNode(name="stop"),
        monitored_node=ConstFalseNode(name="stuck"),
    )
    goal = TryInOrder(nodes=[first, ConstTrueNode(name="second")])
    _compile_and_tick(goal)

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert goal.last_observation_state == ObservationStateValues.TRUE


# %% monitored subtrees


def test_paused_while_true_holds_the_monitored_node_while_the_monitor_is_true():
    """
    The monitored node is held in PAUSED for exactly as long as the monitor observes
    True, and runs again once it turns False.
    """
    pulse_length = 2
    goal = PausedWhileTrue(
        monitor=Pulse(length=pulse_length, name="pulse"),
        monitored_node=CountTicks(ticks=2, name="work"),
    )
    executor = _compile_and_tick(goal, ticks=0)

    for _ in range(pulse_length):
        executor.tick()
        assert goal.monitor.observation_state == ObservationStateValues.TRUE
        assert goal.monitored_node.life_cycle_state == LifeCycleValues.PAUSED

    executor.tick()
    assert goal.monitor.observation_state == ObservationStateValues.FALSE
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.RUNNING


def test_paused_while_true_costs_the_monitored_node_the_paused_ticks():
    """
    Pausing does not merely delay the observation, it stops the monitored node from making
    progress: it needs the paused ticks *on top of* the ticks it needs on its own.
    """
    pulse_length = 2
    unmonitored = PausedWhileTrue(
        monitor=ConstFalseNode(name="never"),
        monitored_node=CountTicks(ticks=2, name="work"),
    )
    ticks_without_pause = _ticks_until_observed_true(
        unmonitored, unmonitored.monitored_node, max_ticks=20
    )

    paused = PausedWhileTrue(
        monitor=Pulse(length=pulse_length, name="pulse"),
        monitored_node=CountTicks(ticks=2, name="work"),
    )
    ticks_with_pause = _ticks_until_observed_true(
        paused, paused.monitored_node, max_ticks=20
    )

    assert ticks_with_pause == ticks_without_pause + pulse_length


def test_paused_until_true_holds_the_monitored_node_until_the_monitor_turns_true():
    """
    The monitored node is held in PAUSED for as long as the monitor observes False, and
    runs from the tick the monitor turns True.
    """
    ticks_until_monitor_fires = 2
    goal = PausedUntilTrue(
        monitor=CountTicks(ticks=ticks_until_monitor_fires, name="arrival"),
        monitored_node=CountTicks(ticks=2, name="work"),
    )
    executor = _compile_and_tick(goal, ticks=0)

    for _ in range(ticks_until_monitor_fires - 1):
        executor.tick()
        assert goal.monitor.observation_state == ObservationStateValues.FALSE
        assert goal.monitored_node.life_cycle_state == LifeCycleValues.PAUSED

    executor.tick()
    assert goal.monitor.observation_state == ObservationStateValues.TRUE
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.RUNNING


def test_paused_until_true_holds_the_monitored_node_while_the_monitor_has_not_decided() -> (
    None
):
    """
    A monitor that has not observed anything yet has not turned True, so the monitored
    node is held all the same.
    """
    goal = PausedUntilTrue(
        monitor=NodeObservingNothingYet(name="undecided"),
        monitored_node=CountTicks(ticks=2, name="work"),
    )

    _compile_and_tick(goal)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.PAUSED


def test_stopped_when_true_ends_the_monitored_node():
    """
    The monitored node is retired as soon as the monitor fires, without ever having
    succeeded.
    """
    goal = StoppedWhenTrue(
        monitor=CountTicks(ticks=2, name="trip"),
        monitored_node=CountTicks(ticks=99, name="work"),
    )
    _compile_and_tick(goal)

    assert goal.monitor.last_observation_state == ObservationStateValues.TRUE
    # Stopping a node decides when it ends, not that it failed.
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.INTERRUPTED


def test_stopped_when_true_fails_once_it_stopped_the_monitored_node():
    """
    Stopping a node short of its goal is reported as a failure of the subtree, which it
    declares itself so that whoever runs it is not left waiting.
    """
    goal = StoppedWhenTrue(
        monitor=CountTicks(ticks=2, name="trip"),
        monitored_node=CountTicks(ticks=99, name="work"),
    )
    _compile_and_tick(goal)

    assert goal.last_observation_state == ObservationStateValues.FALSE
    assert goal.life_cycle_state == LifeCycleValues.FAILED


def test_stopped_when_true_observes_false_once_it_stopped_a_node_at_its_goal() -> None:
    """
    Stopping a node interrupts it however close it was, so a node sitting at its goal
    when the monitor fires is reported as stopped rather than as having arrived.
    """
    goal = StoppedWhenTrue(
        monitor=CountTicks(ticks=2, name="trip"),
        monitored_node=ConstTrueNode(name="work"),
    )
    _compile_and_tick(goal)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert goal.last_observation_state == ObservationStateValues.FALSE


def test_a_stopped_subtree_makes_its_sequence_report_a_failure() -> None:
    """
    A stopped subtree used to leave the sequence running it waiting forever, because a
    node short of its goal never ends on its own.
    """
    stopped = StoppedWhenTrue(
        monitor=CountTicks(ticks=2, name="trip"),
        monitored_node=CountTicks(ticks=99, name="work"),
    )
    sequence = Sequence(nodes=[stopped])

    _compile_and_tick(sequence)

    assert stopped.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.last_observation_state == ObservationStateValues.FALSE


def test_monitored_goals_observe_the_monitored_node_when_the_monitor_never_fires():
    """
    A monitor that stays False leaves the monitored node's outcome untouched.
    """
    for goal_type in (PausedWhileTrue, StoppedWhenTrue):
        goal = goal_type(
            monitor=ConstFalseNode(name="never"),
            monitored_node=ConstTrueNode(name="work"),
        )
        _compile_and_tick(goal)

        assert goal.monitored_node.life_cycle_state == LifeCycleValues.RUNNING
        assert goal.observation_state == goal.monitored_node.observation_state
        assert goal.observation_state == ObservationStateValues.TRUE
