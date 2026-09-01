"""
Tests for the PyCRAM Giskard motion statechart templates ``TryAll`` and ``TryInOrder``
(see ``pycram/src/pycram/language_giskard_templates.py``).

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
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import CountControlCycles
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeObservingNothingYet,
)
from semantic_digital_twin.world import World

from coraplex.language import TryAllNode, TryInOrderNode
from coraplex.language_giskard_templates import TryAll, TryInOrder

# Number of ticks after which the templates below have settled into their final observation.
SETTLE_TICKS = 6

# Simulated seconds an alternative is given before it is abandoned. Short so that a test
# that has to wait out the give-up budget stays fast.
GIVE_UP_AFTER = 0.2


def _compile_and_tick(
    goal: MotionStatechartNode,
    ticks: int = SETTLE_TICKS,
    alternatives_to_abandon: int = 0,
) -> None:
    """
    Add the goal to a fresh statechart, compile it and tick the executor.

    :param goal: The template under test.
    :param ticks: Control cycles to run on top of the give-up budget.
    :param alternatives_to_abandon: How many alternatives have to exhaust
        :data:`GIVE_UP_AFTER` before the assertion holds. Turned into control cycles
        using the control rate the executor actually runs at.
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
