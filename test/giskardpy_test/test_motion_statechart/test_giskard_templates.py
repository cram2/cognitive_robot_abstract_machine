"""
Tests for the PyCRAM Giskard motion statechart templates ``TryAll`` and ``TryInOrder``
(see ``pycram/src/pycram/language_giskard_templates.py``).

The templates are exercised by compiling them into a real :class:`MotionStatechart` and
ticking the executor, asserting the resulting observation and life cycle states.
``ConstTrueNode`` / ``ConstFalseNode`` are used as deterministic children that always
succeed / fail.
"""

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
)
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)

from coraplex.language import TryAllNode, TryInOrderNode
from giskardpy.motion_statechart.goals.templates import TryAll, TryInOrder
from giskardpy.motion_statechart.monitors.templates import (
    PausedWhileTrue,
    PausedUntilTrue,
    StoppedWhenTrue,
)

# Number of ticks after which the templates below have settled into their final observation.
SETTLE_TICKS = 4


def _compile_and_tick(
    goal: MotionStatechartNode, ticks: int = SETTLE_TICKS
) -> Executor:
    """
    Add the goal to a fresh statechart, compile it and tick the executor ``ticks``
    times.

    :return: The executor, so a caller can keep ticking and inspect intermediate states.
    """
    msc = MotionStatechart()
    msc.add_node(goal)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=msc)
    for _ in range(ticks):
        executor.tick()
    return executor


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


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_language_nodes_use_templates():
    """
    The parallel/sequential try-nodes point at the matching statechart templates.
    """
    assert TryAllNode.motion_state_chart_template is TryAll
    assert TryInOrderNode.motion_state_chart_template is TryInOrder


# --------------------------------------------------------------------------- #
# TryAll – parallel, succeeds if any child succeeds
# --------------------------------------------------------------------------- #


def test_try_all_succeeds_if_any_child_succeeds():
    goal = TryAll(nodes=[ConstFalseNode(name="a"), ConstTrueNode(name="b")])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE
    # Children run in parallel: both are RUNNING regardless of outcome.
    assert all(n.life_cycle_state == LifeCycleValues.RUNNING for n in goal.nodes)


def test_try_all_fails_only_if_all_children_fail():
    goal = TryAll(nodes=[ConstFalseNode(name="a"), ConstFalseNode(name="b")])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.FALSE


def test_try_all_single_child():
    goal = TryAll(nodes=[ConstTrueNode(name="only")])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE


# --------------------------------------------------------------------------- #
# TryInOrder – sequential, short-circuits on first success
# --------------------------------------------------------------------------- #


def test_try_in_order_short_circuits_on_first_success():
    first = ConstTrueNode(name="first")
    second = ConstFalseNode(name="second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE
    # First child succeeded and finished...
    assert first.life_cycle_state == LifeCycleValues.DONE
    # ...so the second child is never started (short-circuit).
    assert second.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_try_in_order_advances_after_failure():
    first = ConstFalseNode(name="first")
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE
    # Both children ran: the first failed, the second was started and succeeded.
    assert first.life_cycle_state == LifeCycleValues.DONE
    assert second.life_cycle_state == LifeCycleValues.DONE


def test_try_in_order_fails_only_if_all_children_fail():
    goal = TryInOrder(nodes=[ConstFalseNode(name="a"), ConstFalseNode(name="b")])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.FALSE
    assert all(n.life_cycle_state == LifeCycleValues.DONE for n in goal.nodes)


def test_try_in_order_single_child():
    goal = TryInOrder(nodes=[ConstTrueNode(name="only")])
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE


# %% monitored subtrees


def test_paused_while_true_holds_the_monitored_node_while_the_monitor_is_true():
    """
    The monitored node is held in PAUSED for exactly as long as the monitor observes
    True, and runs again once it turns False.
    """
    pulse_length = 2
    goal = PausedWhileTrue(
        monitor=Pulse(length=pulse_length, name="pulse"),
        monitored_node=CountControlCycles(control_cycles=2, name="work"),
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
        monitored_node=CountControlCycles(control_cycles=2, name="work"),
    )
    ticks_without_pause = _ticks_until_observed_true(
        unmonitored, unmonitored.monitored_node, max_ticks=20
    )

    paused = PausedWhileTrue(
        monitor=Pulse(length=pulse_length, name="pulse"),
        monitored_node=CountControlCycles(control_cycles=2, name="work"),
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
        monitor=CountControlCycles(
            control_cycles=ticks_until_monitor_fires, name="arrival"
        ),
        monitored_node=CountControlCycles(control_cycles=2, name="work"),
    )
    executor = _compile_and_tick(goal, ticks=0)

    for _ in range(ticks_until_monitor_fires - 1):
        executor.tick()
        assert goal.monitor.observation_state == ObservationStateValues.FALSE
        assert goal.monitored_node.life_cycle_state == LifeCycleValues.PAUSED

    executor.tick()
    assert goal.monitor.observation_state == ObservationStateValues.TRUE
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.RUNNING


def test_stopped_when_true_ends_the_monitored_node():
    """
    The monitored node is retired as soon as the monitor fires, without ever having
    succeeded.
    """
    goal = StoppedWhenTrue(
        monitor=CountControlCycles(control_cycles=2, name="trip"),
        monitored_node=CountControlCycles(control_cycles=99, name="work"),
    )
    _compile_and_tick(goal)

    assert goal.monitor.observation_state == ObservationStateValues.TRUE
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.DONE
    assert goal.monitored_node.observation_state == ObservationStateValues.FALSE


def test_stopped_when_true_fails_once_it_stopped_the_monitored_node():
    """
    Its observation turns False, reporting that the monitored node was cut short rather
    than reaching its goal.
    """
    goal = StoppedWhenTrue(
        monitor=CountControlCycles(control_cycles=2, name="trip"),
        monitored_node=CountControlCycles(control_cycles=99, name="work"),
    )
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.FALSE


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
