"""
Tests for :attr:`~cramph.node.StatechartNode.start_time` and
:attr:`~cramph.node.StatechartNode.end_time`, derived from
:class:`~cramph.state_history.StateHistory`.
"""

from __future__ import annotations

import pytest

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.exceptions import TickDurationUnknownError
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode, NodeRecordingItsCallbacks
from cramph.statechart import Statechart
from krrood.symbolic_math.symbolic_math import Scalar

# %% compiling


def _compile(statechart: Statechart) -> StatechartExecutor:
    """
    :param statechart: The statechart to run.
    :return: An executor that compiled `statechart`, which already ticked once.
    """
    executor = StatechartExecutor(statechart.context)
    executor.compile(statechart=statechart)
    return executor


# %% run times across the life cycle


def test_a_node_that_has_not_started_has_no_run_times(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context)
    node = NodeRecordingItsCallbacks()
    node.start_condition = Scalar.const_false()
    statechart.add_node(node)
    _compile(statechart)

    assert node.start_time is None
    assert node.end_time is None


def test_a_running_node_has_a_start_time_and_no_end_time(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context)
    trigger = ConstTrueNode()
    node = NodeRecordingItsCallbacks()
    node.start_condition = trigger.observes_true
    statechart.add_nodes([trigger, node])
    executor = _compile(statechart)

    executor.tick()

    assert node.life_cycle_state == LifeCycleValues.RUNNING
    assert node.start_time == statechart_context.require_tick_duration()
    assert node.end_time is None


def test_pausing_and_unpausing_does_not_change_the_start_time(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context)
    first = NodeRecordingItsCallbacks()
    second = NodeRecordingItsCallbacks()
    first.pause_condition = second.is_running
    second.pause_condition = first.is_running
    statechart.add_nodes([first, second])
    executor = _compile(statechart)
    start_time = first.start_time

    executor.tick()
    assert first.life_cycle_state == LifeCycleValues.PAUSED
    assert first.start_time == start_time
    assert first.end_time is None

    executor.tick()
    assert first.life_cycle_state == LifeCycleValues.RUNNING
    assert first.start_time == start_time
    assert first.end_time is None


def test_an_ended_node_has_both_a_start_time_and_an_end_time(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context)
    trigger = ConstTrueNode()
    node = NodeRecordingItsCallbacks()
    node.fail_condition = trigger.observes_true
    statechart.add_nodes([trigger, node])
    executor = _compile(statechart)

    executor.tick()

    assert node.life_cycle_state == LifeCycleValues.FAILED
    tick_duration = statechart_context.require_tick_duration()
    assert node.start_time == 0 * tick_duration
    assert node.end_time == 1 * tick_duration


def test_resetting_and_restarting_reports_only_the_new_run(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context)
    trigger = ConstTrueNode()
    node = NodeRecordingItsCallbacks()
    node.fail_condition = trigger.observes_true
    node.reset_condition = node.is_failed
    statechart.add_nodes([trigger, node])
    executor = _compile(statechart)

    executor.tick()
    assert node.life_cycle_state == LifeCycleValues.FAILED

    executor.tick()
    assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node.start_time is None
    assert node.end_time is None

    executor.tick()
    assert node.life_cycle_state == LifeCycleValues.RUNNING
    assert node.start_time == 3 * statechart_context.require_tick_duration()
    assert node.end_time is None


# %% run times without a tick duration


def test_start_time_raises_without_a_known_tick_duration(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context_without_tick_duration)
    node = ConstTrueNode()
    statechart.add_node(node)
    _compile(statechart)

    assert node.life_cycle_state == LifeCycleValues.RUNNING
    with pytest.raises(TickDurationUnknownError):
        node.start_time
