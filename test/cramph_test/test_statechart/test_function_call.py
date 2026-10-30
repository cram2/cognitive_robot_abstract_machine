"""
Tests for :class:`~cramph.threaded_nodes.FunctionCall`, a node calling a function once
in a thread of its own.
"""

from __future__ import annotations

import threading

import pytest
from typing_extensions import Never

from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart
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


# %% function call outcomes


def run_until_end(
    statechart_executor: StatechartExecutor, function_call: FunctionCall
) -> None:
    """
    Run a statechart holding only `function_call` until it ends.
    """
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(function_call)
    statechart.add_node(EndStatechart.when_true(function_call))
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=10)


def test_a_function_that_returns_makes_the_node_succeed(
    statechart_executor: StatechartExecutor,
) -> None:
    called = threading.Event()
    function_call = FunctionCall(function=called.set)

    run_until_end(statechart_executor, function_call)

    assert called.is_set()
    assert function_call.life_cycle_state == LifeCycleValues.SUCCEEDED


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
    statechart.add_node(function_call)
    statechart_executor.compile(statechart)

    for _ in range(5):
        statechart_executor.tick()

    assert function_call.life_cycle_state == LifeCycleValues.FAILED


def test_an_unexpected_error_is_raised_out_of_the_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    def fail() -> Never:
        raise UnexpectedError()

    function_call = FunctionCall(function=fail, failure_types=(ExpectedFailure,))

    with pytest.raises(UnexpectedError):
        run_until_end(statechart_executor, function_call)
