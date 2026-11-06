"""
Tests for the repeating template :class:`~cramph.composites.RepeatUntil` (see
``cramph/src/cramph/composites.py``).

The loop is exercised with tasks that observe a fixed value and fail on observing False,
so it needs neither a world nor a converging task.
"""

from functools import partial

from typing_extensions import Callable

import pytest

from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import ChildCannotFailError
from cramph.composites import RepeatUntil
from cramph.node import EndStatechart, StatechartNode
from cramph.statechart import Statechart
from cramph.monitors import CountNodeResets
from cramph.nodes_for_testing import (
    ConstTrueNode,
    NodeAssertionError,
    NodeFailingOnObservingFalse,
    NodeObservingNothingYet,
)
from semantic_digital_twin.world import World

# Ticks after which the world free loops below have settled on an outcome.
# A task declares its own failure and is reset one tick later, so every retry
# costs a few ticks.
SETTLE_TICKS = 20


def _failing_task() -> NodeFailingOnObservingFalse:
    """
    :return: A task that fails on every tick it runs.
    """
    return NodeFailingOnObservingFalse(
        name="task", observation=ObservationStateValues.FALSE
    )


def _repeat(
    executor: StatechartExecutor,
    task: StatechartNode,
    target: int,
    repeat_template: Callable[..., RepeatUntil] = RepeatUntil,
) -> tuple[RepeatUntil, Statechart, StatechartExecutor]:
    """
    Build a compiled chart around a task that is retried until it has been reset
    `target` times.

    :param executor: Compiles the chart.
    :param repeat_template: Builds the loop around the task.
    """
    loop = repeat_template(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=target),
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(loop)
    statechart.add_node(EndStatechart.when_true(loop))
    executor.compile(statechart=statechart)
    return loop, statechart, executor


# %% the loop


def test_repeat_until_retries_until_the_monitor_gives_up(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A task that never succeeds is retried exactly as often as the monitor allows, and
    the goal then reports the failure rather than stalling on Unknown.
    """
    task = _failing_task()
    loop, _, executor = _repeat(statechart_executor, task, target=3)

    for _ in range(SETTLE_TICKS):
        executor.tick()

    assert loop.stop_retry_monitor.resets == 3
    assert loop.last_observation_state == ObservationStateValues.FALSE


def test_repeat_until_succeeds_without_retrying(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A task that succeeds first time is never reset, so the loop exits on success instead
    of running to the monitor's bound.
    """
    task = NodeFailingOnObservingFalse(
        name="task", observation=ObservationStateValues.TRUE
    )
    loop, statechart, executor = _repeat(statechart_executor, task, target=3)

    executor.tick_until_end(SETTLE_TICKS)

    assert loop.stop_retry_monitor.resets == 0
    assert loop.last_observation_state == ObservationStateValues.TRUE
    assert statechart.is_ended()


def test_repeat_until_puts_the_task_back_to_not_started(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Retrying really restarts the task rather than leaving it running, so a task that
    only behaves correctly from its start is safe to retry.

    Every reset but the last starts a fresh run. The monitor calls the retrying off on
    the tick it counts the last reset, so that one is not followed by a run.
    """
    task = _failing_task()
    loop, statechart, executor = _repeat(statechart_executor, task, target=3)

    for _ in range(SETTLE_TICKS):
        executor.tick()

    life_cycles = statechart.state_history.get_life_cycle_history_of_node(task)
    restarts = [
        index
        for index in range(1, len(life_cycles) - 1)
        if life_cycles[index - 1] != LifeCycleValues.NOT_STARTED
        and life_cycles[index] == LifeCycleValues.NOT_STARTED
        and life_cycles[index + 1] == LifeCycleValues.RUNNING
    ]
    assert loop.stop_retry_monitor.resets == 3
    assert len(restarts) == loop.stop_retry_monitor.target - 1


def test_repeat_until_does_not_retry_after_giving_up(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Once the monitor has called the retrying off, no further attempt is started, so a
    finished loop stops consuming ticks.
    """
    task = _failing_task()
    loop, _, executor = _repeat(statechart_executor, task, target=2)

    for _ in range(SETTLE_TICKS):
        executor.tick()
    resets_when_given_up = loop.stop_retry_monitor.resets
    assert loop.last_observation_state == ObservationStateValues.FALSE

    for _ in range(SETTLE_TICKS):
        executor.tick()

    assert loop.stop_retry_monitor.resets == resets_when_given_up
    assert loop.task.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert loop.last_observation_state == ObservationStateValues.FALSE


def test_repeat_until_starts_its_task_while_the_stop_monitor_has_not_decided() -> None:
    """
    A stop monitor that has not observed anything yet has not called the retrying off,
    so the task runs meanwhile.
    """
    loop = RepeatUntil(
        name="loop",
        task=NodeFailingOnObservingFalse(
            name="task", observation=ObservationStateValues.UNKNOWN
        ),
        stop_retry_monitor=NodeObservingNothingYet(name="undecided"),
    )
    executor = StatechartExecutor(StatechartContext(world=World()))
    statechart = Statechart(context=executor.context)
    statechart.add_node(loop)
    executor.compile(statechart=statechart)

    executor.tick()

    assert loop.task.life_cycle_state == LifeCycleValues.RUNNING


def test_repeat_until_rejects_a_task_that_cannot_fail() -> None:
    """
    A plain task has no way of failing, so it would never be retried.
    """
    task = ConstTrueNode(name="task")
    loop = RepeatUntil(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=1),
    )
    executor = StatechartExecutor(StatechartContext(world=World()))
    statechart = Statechart(context=executor.context)
    statechart.add_node(loop)

    with pytest.raises(ChildCannotFailError) as error:
        executor.compile(statechart=statechart)

    assert error.value.node is loop
    assert error.value.child is task


def test_repeat_until_ends_the_statechart_with_its_exception_once_retrying_stops(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A loop handed an exception reports running out of attempts by ending the statechart
    with it, rather than only observing False.
    """
    task = _failing_task()
    exception = NodeAssertionError(reason="attempts exhausted")
    _, _, executor = _repeat(
        statechart_executor,
        task,
        target=2,
        repeat_template=partial(RepeatUntil, exception=exception),
    )

    with pytest.raises(type(exception)) as error:
        executor.tick_until_end(SETTLE_TICKS)

    assert error.value is exception
