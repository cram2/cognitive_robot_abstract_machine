"""
Tests for the motion statechart templates that try alternatives, ``TryAll`` and
``TryInOrder``, and for the goals that run a node under a monitor.

The templates are exercised by compiling them into a real :class:`MotionStatechart` and
ticking the executor, asserting the resulting observation and life cycle states.
``ConstTrueNode`` / ``ConstFalseNode`` are used as deterministic children that always
succeed / fail.
"""

from datetime import timedelta
from math import ceil

import pytest
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import GoalWithoutChildrenError
from giskardpy.motion_statechart.goals.templates import (
    Attempt,
    Sequence,
    TryAll,
    TryInOrder,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
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

# Simulated time an alternative is given before it is abandoned. Short so that a test
# that has to wait out the give-up budget stays fast.
GIVE_UP_AFTER = timedelta(seconds=0.2)


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
        GIVE_UP_AFTER.total_seconds() / context.qp_controller_config.control_dt
    )
    for _ in range(ticks + alternatives_to_abandon * cycles_per_alternative):
        executor.tick()
    return executor


def _alternative(node: MotionStatechartNode) -> Attempt:
    """
    Wrap a node the way a caller of the try-templates has to: an attempt that reaches a
    verdict on its own, giving up once the node stops making progress.

    :param node: The node to try.
    :return: The alternative to hand to the template.
    """
    return Attempt(
        name=f"{node.name}/attempt",
        task=node,
        failure_monitors=[
            Stalled(
                name=f"{node.name}/stalled",
                monitored_node=node,
                timeout=GIVE_UP_AFTER,
            )
        ],
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


# %% wiring


def test_language_nodes_use_templates():
    """
    The parallel/sequential try-nodes point at the matching statechart templates.
    """
    assert TryAllNode.motion_state_chart_template is TryAll
    assert TryInOrderNode.motion_state_chart_template is TryInOrder


# %% TryAll, parallel and succeeding if any child succeeds


def test_try_all_succeeds_if_any_child_succeeds():
    stuck = _alternative(ConstFalseNode(name="a"))
    arriving = _alternative(ConstTrueNode(name="b"))
    goal = TryAll(nodes=[stuck, arriving])
    _compile_and_tick(goal)

    assert goal.goal_reached_state == ObservationStateValues.TRUE
    assert arriving.life_cycle_state == LifeCycleValues.SUCCEEDED
    # Alternatives run side by side, so the other one is not cut short by the success.
    assert stuck.life_cycle_state != LifeCycleValues.NOT_STARTED


def test_try_all_fails_only_if_all_children_fail():
    goal = TryAll(
        nodes=[
            _alternative(ConstFalseNode(name="a")),
            _alternative(ConstFalseNode(name="b")),
        ]
    )
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert goal.goal_reached_state == ObservationStateValues.FALSE


def test_try_all_single_child():
    goal = TryAll(nodes=[_alternative(ConstTrueNode(name="only"))])
    _compile_and_tick(goal)

    assert goal.goal_reached_state == ObservationStateValues.TRUE


# %% TryInOrder, sequential and short-circuiting on the first success


def test_try_in_order_short_circuits_on_first_success():
    first = _alternative(ConstTrueNode(name="first"))
    second = _alternative(ConstFalseNode(name="second"))
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal)

    assert goal.goal_reached_state == ObservationStateValues.TRUE
    # First child succeeded and finished...
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    # ...so the second child is never started (short-circuit).
    assert second.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_try_in_order_advances_after_failure():
    first = _alternative(ConstFalseNode(name="first"))
    second = _alternative(ConstTrueNode(name="second"))
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert goal.goal_reached_state == ObservationStateValues.TRUE
    # Both children ran: the first failed, the second was started and succeeded.
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_try_in_order_fails_only_if_all_children_fail():
    first = _alternative(ConstFalseNode(name="first"))
    second = _alternative(ConstFalseNode(name="second"))
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal, alternatives_to_abandon=2)

    assert goal.goal_reached_state == ObservationStateValues.FALSE
    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.FAILED


def test_try_in_order_single_child():
    goal = TryInOrder(nodes=[_alternative(ConstTrueNode(name="only"))])
    _compile_and_tick(goal)

    assert goal.goal_reached_state == ObservationStateValues.TRUE


# %% progress monitors


def test_a_progress_monitor_ends_with_the_alternative_it_watches():
    """
    A monitor that outlived its alternative would keep measuring progress against a node
    that has ended, and would eventually report that node as stalled long after it was
    decided.
    """
    first = _alternative(ConstFalseNode(name="first"))
    second = _alternative(ConstTrueNode(name="second"))
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert all(
        monitor.life_cycle_state.is_terminal
        for alternative in (first, second)
        for monitor in alternative.failure_monitors
    )


# %% alternatives that need more than one tick to reach their goal

#: Control cycles a slow alternative needs before its observation turns True.
SLOW_ALTERNATIVE_CYCLES = 5


def test_slow_alternative_is_not_abandoned_while_still_working():
    """
    An alternative whose observation is still False because it has not reached its goal
    yet must not be mistaken for one that failed.
    """
    slow = _alternative(
        CountControlCycles(name="slow", control_cycles=SLOW_ALTERNATIVE_CYCLES)
    )
    fallback = _alternative(ConstTrueNode(name="fallback"))
    goal = TryInOrder(nodes=[slow, fallback])
    _compile_and_tick(goal, ticks=2)

    assert slow.life_cycle_state == LifeCycleValues.RUNNING
    assert fallback.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% when the next alternative takes over


def test_the_next_alternative_starts_on_the_cycle_the_previous_one_fails():
    """
    An alternative waits for its predecessor's verdict, which it reads on the cycle that
    verdict is reached, so no control cycle passes with neither of them running.
    """
    first = _alternative(ConstFalseNode(name="first"))
    second = _alternative(ConstTrueNode(name="second"))
    goal = TryInOrder(nodes=[first, second])

    msc = MotionStatechart()
    msc.add_node(goal)
    context = MotionStatechartContext(world=World())
    executor = Executor(context)
    executor.compile(motion_statechart=msc)

    cycles_to_abandon_an_alternative = ceil(
        GIVE_UP_AFTER.total_seconds() / context.qp_controller_config.control_dt
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
    first = _alternative(NodeObservingNothingYet(name="first"))
    second = _alternative(ConstTrueNode(name="second"))
    goal = TryInOrder(nodes=[first, second])
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.goal_reached_state == ObservationStateValues.TRUE


def test_a_composite_alternative_that_never_arrives_hands_over_to_the_next_one():
    """
    An alternative built from several nodes observes nothing decisive while its steps
    are still short of their goals, which is what it looks like when it is abandoned.
    """
    first = _alternative(
        Sequence(
            name="first",
            nodes=[
                _alternative(ConstFalseNode(name="stuck step")),
                _alternative(ConstTrueNode(name="unreached step")),
            ],
        )
    )
    fallback = _alternative(ConstTrueNode(name="fallback"))
    goal = TryInOrder(nodes=[first, fallback])
    _compile_and_tick(goal, alternatives_to_abandon=1)

    assert first.life_cycle_state == LifeCycleValues.FAILED
    assert fallback.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.goal_reached_state == ObservationStateValues.TRUE


def test_the_goal_fails_once_every_alternative_was_abandoned():
    """
    Giving up on the last alternative decides the goal, instead of leaving whoever waits
    for it waiting forever.
    """
    goal = TryInOrder(
        nodes=[
            _alternative(NodeObservingNothingYet(name="first")),
            _alternative(NodeObservingNothingYet(name="second")),
        ]
    )
    _compile_and_tick(goal, alternatives_to_abandon=2)

    assert goal.goal_reached_state == ObservationStateValues.FALSE


# %% goals built without children


def test_a_try_all_without_nodes_is_rejected():
    msc = MotionStatechart()
    msc.add_node(TryAll(nodes=[]))

    executor = Executor(MotionStatechartContext(world=World()))
    with pytest.raises(GoalWithoutChildrenError):
        executor.compile(motion_statechart=msc)


def test_a_try_in_order_without_nodes_is_rejected():
    msc = MotionStatechart()
    msc.add_node(TryInOrder(nodes=[]))

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

    assert goal.monitor.goal_reached_state == ObservationStateValues.TRUE
    # Stopping a node decides when it ends, not that it failed.
    assert goal.monitored_node.life_cycle_state == LifeCycleValues.INTERRUPTED


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


def test_stopped_when_true_observes_false_once_it_stopped_a_node_at_its_goal():
    """
    Stopping a node interrupts it however close it was, so a node sitting at its goal
    when the monitor fires is reported as stopped rather than as having arrived.
    """
    goal = StoppedWhenTrue(
        monitor=CountControlCycles(control_cycles=2, name="trip"),
        monitored_node=ConstTrueNode(name="work"),
    )
    _compile_and_tick(goal)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.INTERRUPTED
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
