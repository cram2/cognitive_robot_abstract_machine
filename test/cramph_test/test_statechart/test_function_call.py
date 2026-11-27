"""
Tests for :class:`~cramph.threaded_nodes.FunctionCall`, a node calling a function once
in a thread of its own without holding up the ticks, and for how
:class:`~cramph.threaded_nodes.ThreadedNode` keeps only the outcome of its latest run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta

import pytest
from typing_extensions import Callable, Never

from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, EndedByOwner, StatechartNode
from cramph.statechart import Statechart
from cramph.threaded_nodes import FunctionCall


# %% errors the called functions raise


class ExpectedFailure(Exception):
    """
    A failure the statechart is told to expect from the called function.
    """


class UnexpectedError(Exception):
    """
    An error the statechart is not told to expect from the called function.
    """


# %% nodes and functions the tests drive


@dataclass(kw_only=True)
class CallReleasedByTheTest:
    """
    One call of a :class:`FunctionReleasedByTheTest`, held until the test releases it.
    """

    release: threading.Event = field(default_factory=threading.Event)
    """
    Set by the test to let the call end.
    """

    error: BaseException | None = None
    """
    What the call raises once released; None if it returns.
    """

    thread: threading.Thread | None = None
    """
    The thread the call ran in, once it started.
    """

    def wait_until_ended(self) -> None:
        """
        Block until the call's thread ended, its outcome kept or discarded.
        """
        self.thread.join(timeout=10)


@dataclass(kw_only=True)
class FunctionReleasedByTheTest:
    """
    A function whose n-th call waits until the test releases :attr:`calls` [n].
    """

    calls: list[CallReleasedByTheTest]
    """
    The calls this function goes through, in the order they start.
    """

    started_calls: int = field(default=0, init=False)
    """
    How many calls have started.
    """

    def wait_for_release(self) -> None:
        """
        Make the next call: wait until the test released it, then raise its error if it
        has one.
        """
        call = self.calls[self.started_calls]
        self.started_calls += 1
        call.thread = threading.current_thread()
        call.release.wait(timeout=10)
        if call.error is not None:
            raise call.error


@dataclass(eq=False, repr=False, kw_only=True)
class SwitchSetByTheTest(EndedByOwner, StatechartNode):
    """
    A node observing what the test last set, so other nodes' transitions can follow it.
    """

    observation: ObservationStateValue = ObservationStateValue.FALSE
    """
    What this node observes on its next tick.
    """

    def on_tick(self) -> ObservationStateValue:
        """
        Observe what the test last set in :attr:`observation`.
        """
        return self.observation


# %% ticking


def tick_until(
    *,
    statechart_executor: StatechartExecutor,
    condition: Callable[[], bool],
    timeout: timedelta = timedelta(seconds=10),
) -> None:
    """
    Tick until `condition` holds, giving the function's thread time to run.

    :raises TimeoutError: If it does not hold within `timeout`.
    """
    deadline = time.monotonic() + timeout.total_seconds()
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError()
        statechart_executor.tick()


# %% function call outcomes


def run_until_end(
    statechart_executor: StatechartExecutor, function_call: FunctionCall
) -> None:
    """
    Run a statechart holding only `function_call` until it ends.
    """
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=function_call)
    statechart.add_node(node=EndStatechart.when_true(node=function_call))
    statechart_executor.compile(statechart=statechart)
    statechart_executor.tick_until_end(timeout=10)


def test_a_function_that_returns_makes_the_node_succeed(
    statechart_executor: StatechartExecutor,
) -> None:
    called = threading.Event()
    function_call = FunctionCall(function=called.set)

    run_until_end(statechart_executor, function_call)

    assert called.is_set()
    assert function_call.life_cycle_state == LifeCycleValue.SUCCEEDED


def test_the_function_runs_in_a_thread_of_its_own(
    statechart_executor: StatechartExecutor,
) -> None:
    calling_threads = []
    function_call = FunctionCall(
        function=lambda: calling_threads.append(threading.current_thread())
    )

    run_until_end(statechart_executor, function_call)

    assert calling_threads != [threading.current_thread()]


def test_an_expected_failure_makes_the_node_fail(
    statechart_executor: StatechartExecutor,
) -> None:
    def fail() -> Never:
        raise ExpectedFailure()

    statechart = Statechart(context=statechart_executor.context)
    function_call = FunctionCall(function=fail, failure_types=(ExpectedFailure,))
    statechart.add_node(node=function_call)
    statechart_executor.compile(statechart=statechart)

    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: function_call.life_cycle_state == LifeCycleValue.FAILED,
    )

    assert function_call.life_cycle_state == LifeCycleValue.FAILED


def test_an_unexpected_error_is_raised_out_of_the_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    def fail() -> Never:
        raise UnexpectedError()

    function_call = FunctionCall(function=fail, failure_types=(ExpectedFailure,))

    with pytest.raises(UnexpectedError):
        run_until_end(statechart_executor, function_call)


# %% ticks never wait


def test_the_ticks_go_on_while_the_function_runs(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A function still running neither holds up the ticks nor ends the node.
    """
    call = CallReleasedByTheTest()
    function = FunctionReleasedByTheTest(calls=[call])
    function_call = FunctionCall(function=function.wait_for_release)
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=function_call)
    statechart_executor.compile(statechart=statechart)

    for _ in range(5):
        statechart_executor.tick()

    assert not call.release.is_set()
    assert function_call.life_cycle_state == LifeCycleValue.RUNNING
    assert function_call.observation_state == ObservationStateValue.UNKNOWN
    call.release.set()


# %% calls a restart superseded


def test_the_outcome_of_a_superseded_call_is_discarded(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A call that a restart superseded and that ends last does not decide the outcome.
    """
    superseded_call = CallReleasedByTheTest(error=ExpectedFailure())
    latest_call = CallReleasedByTheTest()
    function = FunctionReleasedByTheTest(calls=[superseded_call, latest_call])
    function_call = FunctionCall(
        function=function.wait_for_release, failure_types=(ExpectedFailure,)
    )
    restart = SwitchSetByTheTest()
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=restart)
    statechart.add_node(node=function_call)
    function_call.start_condition = restart.observes_false
    function_call.reset_condition = restart.observes_true
    statechart_executor.compile(statechart=statechart)
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: function.started_calls == 1,
    )

    restart.observation = ObservationStateValue.TRUE
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: function_call.life_cycle_state == LifeCycleValue.NOT_STARTED,
    )
    restart.observation = ObservationStateValue.FALSE
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: function.started_calls == 2,
    )
    latest_call.release.set()
    latest_call.wait_until_ended()
    superseded_call.release.set()
    superseded_call.wait_until_ended()
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: function_call.life_cycle_state != LifeCycleValue.RUNNING,
    )

    assert function_call.life_cycle_state == LifeCycleValue.SUCCEEDED
