"""
Tests for the motion statechart templates that try alternatives, ``TryAll`` and
``TryInOrder``, and for the goals that run a node under a monitor.

The templates are exercised by compiling them into a real :class:`MotionStatechart` and
ticking the executor, asserting the resulting observation and life cycle states.
``ConstTrueNode`` / ``ConstFalseNode`` are used as deterministic children that always
succeed / fail.
"""

from math import ceil

import pytest
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import GoalWithoutChildrenError
from giskardpy.motion_statechart.goals.templates import Sequence, TryAll, TryInOrder
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeObservingNothingYet,
)
from giskardpy.motion_statechart.monitors.templates import (
    PausedUntilTrue,
    PausedWhileTrue,
    StoppedWhenTrue,
)
from semantic_digital_twin.world import World

from coraplex.language import TryAllNode, TryInOrderNode

# Number of ticks after which the templates below have settled into their final observation.
SETTLE_TICKS = 6

# Simulated seconds an alternative is given before it is abandoned. Short so that a test
# that has to wait out the give-up budget stays fast.
GIVE_UP_AFTER = 0.2


def _compile_and_tick(
    goal: MotionStatechartNode,
    ticks: int = SETTLE_TICKS,
    alternatives_to_abandon: int = 0,
) -> Executor:
    """
    Add the goal to a fresh statechart, compile it and tick the executor.

    :param goal: The template under test.
    :param ticks: Control cycles to run on top of the give-up budget.
    :param alternatives_to_abandon: How many alternatives have to exhaust
        :data:`GIVE_UP_AFTER` before the assertion holds. Turned into control cycles
        using the control rate the executor actually runs at.
    :return: The executor, so a caller can keep ticking and inspect intermediate states.
    """
    msc = MotionStatechart()
    msc.add_node(goal)
    context = MotionStatechartContext(world=World())
    executor = Executor(context)
    executor.compile(motion_statechart=msc)
    cycles_per_alternative = ceil(
        GIVE_UP_AFTER / context.qp_controller_config.control_dt
    )
    for _ in range(ticks + alternatives_to_abandon * cycles_per_alternative):
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


# %% wiring


def test_language_nodes_use_templates():
    """
    The parallel/sequential try-nodes point at the matching statechart templates.
    """
    assert TryAllNode.motion_state_chart_template is TryAll
    assert TryInOrderNode.motion_state_chart_template is TryInOrder


# %% TryAll, parallel and succeeding if any child succeeds


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


# %% TryInOrder, sequential and short-circuiting on the first success


def test_try_in_order_short_circuits_on_first_success():
    first = ConstTrueNode(name="first")
    second = ConstFalseNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE
    # First child succeeded and finished...
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    # ...so the second child is never started (short-circuit).
    assert second.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_try_in_order_advances_after_failure():
    first = ConstFalseNode(name="first")
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert goal.observation_state == ObservationStateValues.TRUE
    # Both children ran: the first failed, the second was started and succeeded.
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_try_in_order_fails_only_if_all_children_fail():
    first = ConstFalseNode(name="first")
    second = ConstFalseNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, alternatives_to_abandon=2)

    assert goal.observation_state == ObservationStateValues.FALSE
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.FAILED


def test_try_in_order_single_child():
    goal = TryInOrder(nodes=[ConstTrueNode(name="only")], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal)

    assert goal.observation_state == ObservationStateValues.TRUE


# %% progress monitors


def test_a_progress_monitor_ends_with_the_alternative_it_watches():
    """
    A monitor that outlived its alternative would keep measuring progress against a node
    that has ended, and would eventually report that node as stalled long after it was
    decided.
    """
    first = ConstFalseNode(name="first")
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, alternatives_to_abandon=1)

    monitors = [node for node in goal.nodes if isinstance(node, StillProgressing)]

    # The first alternative was abandoned because its monitor saw it stall, the second
    # succeeded while its monitor was still seeing progress.
    assert [monitor.life_cycle_state for monitor in monitors] == [
        LifeCycleValues.FAILED,
        LifeCycleValues.SUCCEEDED,
    ]


# %% alternatives that need more than one tick to reach their goal

#: Control cycles a slow alternative needs before its observation turns True.
SLOW_ALTERNATIVE_CYCLES = 5


def test_slow_alternative_is_not_abandoned_while_still_working():
    """
    An alternative whose observation is still False because it has not reached its goal
    yet must not be mistaken for one that failed.
    """
    slow = CountControlCycles(name="slow", control_cycles=SLOW_ALTERNATIVE_CYCLES)
    fallback = ConstTrueNode(name="fallback")
    goal = TryInOrder(nodes=[slow, fallback], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, ticks=2)

    assert slow.life_cycle_state == LifeCycleValues.RUNNING
    assert fallback.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% when the next alternative takes over


def test_the_next_alternative_starts_on_the_cycle_the_previous_one_fails():
    """
    An alternative waits for its predecessor's verdict, which it reads on the cycle that
    verdict is reached, so no control cycle passes with neither of them running.
    """
    first = ConstFalseNode(name="first")
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)

    msc = MotionStatechart()
    msc.add_node(goal)
    context = MotionStatechartContext(world=World())
    executor = Executor(context)
    executor.compile(motion_statechart=msc)

    cycles_to_abandon_an_alternative = ceil(
        GIVE_UP_AFTER / context.qp_controller_config.control_dt
    )
    for _ in range(cycles_to_abandon_an_alternative + SETTLE_TICKS):
        executor.tick()
        if first.life_cycle_state == LifeCycleValues.FAILED:
            break

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.RUNNING


# %% alternatives abandoned before they observed anything


def test_an_alternative_abandoned_undecided_hands_over_to_the_next_one():
    """
    An alternative that is given up on while it still observes nothing is no more use
    than one that failed outright, so the next one has to be tried.
    """
    first = NodeObservingNothingYet(name="first")
    second = ConstTrueNode(name="second")
    goal = TryInOrder(nodes=[first, second], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert first.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.observation_state == ObservationStateValues.TRUE


def test_a_composite_alternative_that_never_arrives_hands_over_to_the_next_one():
    """
    An alternative built from several nodes observes nothing decisive while its steps
    are still short of their goals, which is what it looks like when it is abandoned.
    """
    first = Sequence(
        name="first",
        nodes=[ConstFalseNode(name="stuck step"), ConstTrueNode(name="unreached step")],
    )
    fallback = ConstTrueNode(name="fallback")
    goal = TryInOrder(nodes=[first, fallback], give_up_after=GIVE_UP_AFTER)
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert first.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert fallback.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.observation_state == ObservationStateValues.TRUE


def test_the_goal_fails_once_every_alternative_was_abandoned():
    """
    Giving up on the last alternative decides the goal, instead of leaving whoever waits
    for it waiting forever.
    """
    goal = TryInOrder(
        nodes=[
            NodeObservingNothingYet(name="first"),
            NodeObservingNothingYet(name="second"),
        ],
        give_up_after=GIVE_UP_AFTER,
    )
    _compile_and_tick(goal, alternatives_to_abandon=2)

    assert goal.observation_state == ObservationStateValues.FALSE


# %% goals built without children


def test_a_try_all_without_nodes_is_rejected():
    msc = MotionStatechart()
    msc.add_node(TryAll(nodes=[]))

    executor = Executor(MotionStatechartContext(world=World()))
    with pytest.raises(GoalWithoutChildrenError):
        executor.compile(motion_statechart=msc)


def test_a_try_in_order_without_nodes_is_rejected():
    msc = MotionStatechart()
    msc.add_node(TryInOrder(nodes=[], give_up_after=GIVE_UP_AFTER))

    executor = Executor(MotionStatechartContext(world=World()))
    with pytest.raises(GoalWithoutChildrenError):
        executor.compile(motion_statechart=msc)


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
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.FAILED


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
