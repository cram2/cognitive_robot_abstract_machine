"""
Tests for the PyCRAM Giskard motion statechart templates ``TryAll`` and ``TryInOrder``
(see ``pycram/src/pycram/language_giskard_templates.py``).

The templates are exercised by compiling them into a real :class:`MotionStatechart` and ticking the
executor, asserting the resulting observation and life cycle states. ``ConstTrueNode`` /
``ConstFalseNode`` are used as deterministic children that always succeed / fail.
"""

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ConstFalseNode,
    ConstTrueNode,
)
from semantic_digital_twin.world import World

from coraplex.language import TryAllNode, TryInOrderNode
from coraplex.language_giskard_templates import TryAll, TryInOrder

# Number of ticks after which the templates below have settled into their final observation.
SETTLE_TICKS = 4


def _compile_and_tick(goal: MotionStatechartNode, ticks: int = SETTLE_TICKS) -> None:
    """Add the goal to a fresh statechart, compile it and tick the executor ``ticks`` times."""
    msc = MotionStatechart()
    msc.add_node(goal)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=msc)
    for _ in range(ticks):
        executor.tick()


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_language_nodes_use_templates():
    """The parallel/sequential try-nodes point at the matching statechart templates."""
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
