"""
Tests for recovering from a failure while a statechart runs: resetting a node from a
callback, :mod:`cramph.failure_handling`, and inserting into a compiled plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

import pytest

import krrood.symbolic_math.symbolic_math as sm
from cramph.compiler import RecompileCallback
from cramph.composites import ControlStructureNode, Sequence
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    ChildCannotFailError,
    ChildTransitionAlreadyWiredError,
    ConditionScopeError,
    NotInStatechartError,
    StatechartAlreadyCompiledError,
)
from cramph.executor import StatechartExecutor
from cramph.failure_handling import (
    FailureHandlingNode,
    FailureHandlingStrategy,
    HandledOnFailure,
    RetryFailedNode,
)
from cramph.node import (
    FailsOnObservingFalse,
    StatechartNode,
    StructureCopyCompositeNode,
    SucceedsOnObservingTrue,
)
from cramph.nodes_for_testing import (
    ConstTrueNode,
    LifeCycleCallback,
    NodeFailingOnObservingFalse,
    NodeRecordingItsCallbacks,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart

# %% mimics


class ModificationDeliberatelyFailed(Exception):
    """
    Raised inside a modification block to abandon it.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeFailingItsFirstRuns(
    SucceedsOnObservingTrue, FailsOnObservingFalse, StatechartNode
):
    """
    A node that fails on its first :attr:`failing_runs` runs and succeeds after that,
    counting its runs and resets.
    """

    failing_runs: int
    """
    How many runs fail before one succeeds.
    """

    runs: int = field(default=0, init=False)
    """
    How often this node started.
    """

    resets: int = field(default=0, init=False)
    """
    How often this node was reset.
    """

    def on_start(self) -> None:
        self.runs += 1

    def on_reset(self) -> None:
        self.resets += 1

    def on_tick(self) -> ObservationStateValues:
        if self.runs <= self.failing_runs:
            return ObservationStateValues.FALSE
        return ObservationStateValues.TRUE


@dataclass
class StrategyGivingUp(FailureHandlingStrategy):
    """
    A failure handling strategy that leaves the failed node as it is and records it.
    """

    failed_nodes: list[StatechartNode] = field(default_factory=list)
    """
    The node this strategy was given on every call, in order.
    """

    def handle_failure(self, failed_node: StatechartNode) -> None:
        self.failed_nodes.append(failed_node)


@dataclass
class StrategyRetrying(StrategyGivingUp):
    """
    A failure handling strategy that records the failed node and runs it again.
    """

    def handle_failure(self, failed_node: StatechartNode) -> None:
        super().handle_failure(failed_node)
        failed_node.statechart.request_reset(failed_node)


@dataclass
class StrategyInsertingBeforeTheFailedStep(FailureHandlingStrategy):
    """
    A failure handling strategy that runs a new step before the step holding the failed
    node, and then that step again.
    """

    inserted_step: StatechartNode
    """
    The step to insert.
    """

    def handle_failure(self, failed_node: StatechartNode) -> None:
        step = ControlStructureNode.step_holding(failed_node)
        step.parent_node.insert_before(step, self.inserted_step)
        failed_node.statechart.request_reset(step)


@dataclass
class CallbackHoldingBackRecompiles(RecompileCallback):
    """
    A recompile callback that answers it is at rest only once told to.
    """

    at_rest: bool = False
    """
    What :meth:`before_recompile` answers.
    """

    def before_recompile(self) -> bool:
        return self.at_rest

    def after_recompile(self) -> None:
        return None


# %% helpers


def _always_failing_node() -> NodeFailingItsFirstRuns:
    """
    :return: A node that fails on more runs than any test here lets it run.
    """
    return NodeFailingItsFirstRuns(name="always failing", failing_runs=100)


def _node_arriving_at_once(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _life_cycle_changes_of(
    statechart: Statechart, node: StatechartNode
) -> list[LifeCycleValues]:
    """
    :return: The life cycle states recorded for `node`, without consecutive repetitions.
    """
    history = statechart.state_history.get_life_cycle_history_of_node(node)
    return [life_cycle for life_cycle, _ in groupby(history)]


def _compile_with_nodes(
    executor: StatechartExecutor, nodes: list[StatechartNode]
) -> Statechart:
    """
    :return: A statechart holding `nodes`, compiled by `executor`, which ticks it once.
    """
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(nodes)
    executor.compile(statechart)
    return statechart


def _tick_times(executor: StatechartExecutor, ticks: int) -> None:
    """
    Tick `executor` `ticks` times.
    """
    for _ in range(ticks):
        executor.tick()


# %% resetting a node from a callback


def test_a_requested_reset_keeps_the_failure_in_the_history(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    statechart = _compile_with_nodes(statechart_executor, [node])
    statechart_executor.tick()
    assert node.life_cycle_state == LifeCycleValues.FAILED

    statechart.request_reset(node)
    assert node.life_cycle_state == LifeCycleValues.FAILED
    _tick_times(statechart_executor, 2)

    assert _life_cycle_changes_of(statechart, node) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.FAILED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
    ]


def test_a_node_stays_not_started_on_the_tick_its_requested_reset_lands(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    statechart = _compile_with_nodes(statechart_executor, [node])
    statechart_executor.tick()

    statechart.request_reset(node)
    statechart_executor.tick()

    assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node.runs == 1


def test_a_requested_reset_runs_on_reset_once(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    statechart = _compile_with_nodes(statechart_executor, [node])
    statechart_executor.tick()

    statechart.request_reset(node)
    statechart.request_reset(node)
    _tick_times(statechart_executor, 3)

    assert node.resets == 1


def test_a_requested_reset_of_a_composite_node_resets_its_children(
    statechart_executor: StatechartExecutor,
) -> None:
    step = NodeFailingItsFirstRuns(failing_runs=1)
    sequence = Sequence(nodes=[step])
    statechart = _compile_with_nodes(statechart_executor, [sequence])
    statechart_executor.tick()
    assert sequence.life_cycle_state == LifeCycleValues.FAILED

    statechart.request_reset(sequence)
    statechart_executor.tick()

    assert sequence.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert step.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert step.resets == 1


def test_a_requested_reset_of_a_running_node_ends_its_run(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeRecordingItsCallbacks()
    statechart = _compile_with_nodes(statechart_executor, [node])
    assert node.take_callbacks() == [LifeCycleCallback.START]

    statechart.request_reset(node)
    _tick_times(statechart_executor, 2)

    assert node.take_callbacks() == [LifeCycleCallback.RESET, LifeCycleCallback.START]


def test_a_requested_reset_of_a_node_not_started_changes_nothing(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeRecordingItsCallbacks()
    node.start_condition = sm.Scalar.const_false()
    statechart = _compile_with_nodes(statechart_executor, [node])

    statechart.request_reset(node)
    statechart_executor.tick()

    assert node.take_callbacks() == []
    assert node.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_a_reset_cannot_be_requested_for_a_node_of_another_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = _compile_with_nodes(statechart_executor, [ConstTrueNode()])

    with pytest.raises(NotInStatechartError):
        statechart.request_reset(ConstTrueNode())


def test_a_requested_reset_waits_while_a_recompile_is_held_back(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    statechart = _compile_with_nodes(statechart_executor, [node])
    statechart_executor.tick()
    holding_back = CallbackHoldingBackRecompiles()
    statechart.compiler.add_recompile_callback(holding_back)
    statechart.add_node(_node_arriving_at_once("joined"))

    statechart.request_reset(node)
    statechart_executor.tick()
    assert node.life_cycle_state == LifeCycleValues.FAILED

    holding_back.at_rest = True
    statechart_executor.tick()
    assert node.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_an_abandoned_modification_drops_the_resets_requested_in_it(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    statechart = _compile_with_nodes(statechart_executor, [node])
    statechart_executor.tick()

    with pytest.raises(ModificationDeliberatelyFailed):
        with statechart.modify():
            statechart.request_reset(node)
            raise ModificationDeliberatelyFailed()
    statechart_executor.tick()

    assert node.life_cycle_state == LifeCycleValues.FAILED


# %% the failure handling node


def test_the_strategy_does_not_run_for_a_node_that_succeeds(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=0)
    strategy = StrategyGivingUp()
    handler = FailureHandlingNode(
        observed_node=node, failure_handling_strategy=strategy
    )
    _compile_with_nodes(statechart_executor, [node, handler])

    _tick_times(statechart_executor, 3)

    assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert handler.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert strategy.failed_nodes == []


def test_the_strategy_is_given_the_failed_node_on_the_tick_it_fails(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    strategy = StrategyGivingUp()
    handler = FailureHandlingNode(
        observed_node=node, failure_handling_strategy=strategy
    )
    _compile_with_nodes(statechart_executor, [node, handler])

    statechart_executor.tick()

    assert node.life_cycle_state == LifeCycleValues.FAILED
    assert strategy.failed_nodes == [node]


def test_the_handler_succeeds_once_the_strategy_ran(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    handler = FailureHandlingNode(
        observed_node=node, failure_handling_strategy=StrategyGivingUp()
    )
    _compile_with_nodes(statechart_executor, [node, handler])

    _tick_times(statechart_executor, 2)

    assert handler.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert node.life_cycle_state == LifeCycleValues.FAILED


def test_the_strategy_runs_once_for_every_failure(
    statechart_executor: StatechartExecutor,
) -> None:
    failing_runs = 3
    node = NodeFailingItsFirstRuns(failing_runs=failing_runs)
    strategy = StrategyRetrying()
    handler = FailureHandlingNode(
        observed_node=node, failure_handling_strategy=strategy
    )
    _compile_with_nodes(statechart_executor, [node, handler])

    _tick_times(statechart_executor, 12)

    assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert strategy.failed_nodes == [node] * failing_runs
    assert node.runs == failing_runs + 1


def test_a_sequence_rejects_a_handler_as_a_step(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    handler = FailureHandlingNode(
        observed_node=node, failure_handling_strategy=StrategyGivingUp()
    )
    statechart = Statechart(context=statechart_executor.context)

    with pytest.raises(ChildTransitionAlreadyWiredError):
        statechart.add_node(Sequence(nodes=[node, handler]))


def test_a_handler_cannot_observe_a_node_that_is_not_its_sibling(
    statechart_executor: StatechartExecutor,
) -> None:
    step = NodeFailingItsFirstRuns(failing_runs=1)
    handler = FailureHandlingNode(
        observed_node=step, failure_handling_strategy=StrategyGivingUp()
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([Sequence(nodes=[step]), handler])

    with pytest.raises(ConditionScopeError):
        statechart_executor.compile(statechart)


# %% handling the failure of a step


def test_a_sequence_keeps_running_while_its_step_is_recovered(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    sequence = Sequence(
        nodes=[
            HandledOnFailure(
                handled_node=node, failure_handling_strategy=StrategyRetrying()
            )
        ]
    )
    _compile_with_nodes(statechart_executor, [sequence])

    statechart_executor.tick()

    assert node.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.life_cycle_state == LifeCycleValues.RUNNING


def test_a_sequence_goes_on_once_its_step_recovered(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    handled = HandledOnFailure(
        handled_node=node, failure_handling_strategy=StrategyRetrying()
    )
    after = _node_arriving_at_once("after")
    sequence = Sequence(nodes=[handled, after])
    _compile_with_nodes(statechart_executor, [sequence])

    _tick_times(statechart_executor, 8)

    assert handled.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert after.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_sequence_fails_once_the_strategy_gave_up(
    statechart_executor: StatechartExecutor,
) -> None:
    handled = HandledOnFailure(
        handled_node=_always_failing_node(),
        failure_handling_strategy=StrategyGivingUp(),
    )
    after = _node_arriving_at_once("after")
    sequence = Sequence(nodes=[handled, after])
    _compile_with_nodes(statechart_executor, [sequence])

    _tick_times(statechart_executor, 3)

    assert handled.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.life_cycle_state == LifeCycleValues.FAILED
    assert after.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_the_strategy_does_not_run_for_a_step_that_succeeds(
    statechart_executor: StatechartExecutor,
) -> None:
    strategy = StrategyGivingUp()
    handled = HandledOnFailure(
        handled_node=NodeFailingItsFirstRuns(failing_runs=0),
        failure_handling_strategy=strategy,
    )
    _compile_with_nodes(statechart_executor, [handled])

    _tick_times(statechart_executor, 3)

    assert handled.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert strategy.failed_nodes == []


def test_a_handled_node_its_owner_ends_succeeds_at_its_goal(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingOnObservingFalse(observation=ObservationStateValues.TRUE)
    handled = HandledOnFailure(
        handled_node=node, failure_handling_strategy=StrategyGivingUp()
    )
    _compile_with_nodes(statechart_executor, [handled])

    _tick_times(statechart_executor, 3)

    assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert handled.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_that_cannot_fail_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    node = ConstTrueNode()
    handled = HandledOnFailure(
        handled_node=node, failure_handling_strategy=StrategyGivingUp()
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(handled)

    with pytest.raises(ChildCannotFailError) as error:
        statechart_executor.compile(statechart)

    assert error.value.node is handled
    assert error.value.child is node


def test_a_structure_copy_holds_the_handler_below_a_composite_stand_in(
    statechart_executor: StatechartExecutor,
) -> None:
    handled = HandledOnFailure(
        handled_node=NodeFailingItsFirstRuns(failing_runs=1),
        failure_handling_strategy=StrategyGivingUp(),
    )
    statechart = _compile_with_nodes(statechart_executor, [handled])

    structure_copy = statechart.create_structure_copy()

    stand_in = structure_copy.get_node_by_index(handled.index)
    assert isinstance(stand_in, StructureCopyCompositeNode)
    assert [child.index for child in stand_in.nodes] == [
        child.index for child in handled.nodes
    ]


# %% retrying


def test_retrying_recovers_a_node_within_the_attempts(
    statechart_executor: StatechartExecutor,
) -> None:
    maximum_attempts = 2
    node = NodeFailingItsFirstRuns(failing_runs=maximum_attempts)
    handled = HandledOnFailure(
        handled_node=node,
        failure_handling_strategy=RetryFailedNode(maximum_attempts=maximum_attempts),
    )
    _compile_with_nodes(statechart_executor, [handled])

    _tick_times(statechart_executor, 10)

    assert node.runs == maximum_attempts + 1
    assert handled.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_retrying_gives_up_once_the_attempts_ran_out(
    statechart_executor: StatechartExecutor,
) -> None:
    maximum_attempts = 2
    node = _always_failing_node()
    handled = HandledOnFailure(
        handled_node=node,
        failure_handling_strategy=RetryFailedNode(maximum_attempts=maximum_attempts),
    )
    _compile_with_nodes(statechart_executor, [handled])

    _tick_times(statechart_executor, 20)

    assert node.resets == maximum_attempts
    assert node.runs == maximum_attempts + 1
    assert node.life_cycle_state == LifeCycleValues.FAILED
    assert handled.life_cycle_state == LifeCycleValues.FAILED


def test_the_attempts_are_not_counted_anew_when_the_handled_node_is_reset(
    statechart_executor: StatechartExecutor,
) -> None:
    node = _always_failing_node()
    handled = HandledOnFailure(
        handled_node=node,
        failure_handling_strategy=RetryFailedNode(maximum_attempts=1),
    )
    statechart = _compile_with_nodes(statechart_executor, [handled])
    _tick_times(statechart_executor, 10)
    assert handled.life_cycle_state == LifeCycleValues.FAILED
    runs_before = node.runs

    statechart.request_reset(handled)
    _tick_times(statechart_executor, 10)

    assert node.runs == runs_before + 1
    assert handled.life_cycle_state == LifeCycleValues.FAILED


# %% inserting into a compiled plan


def test_a_rewriting_modification_lets_a_compiled_sequence_take_a_step(
    statechart_executor: StatechartExecutor,
) -> None:
    first = _node_arriving_at_once("first")
    sequence = Sequence(nodes=[first])
    statechart = _compile_with_nodes(statechart_executor, [sequence])
    second = _node_arriving_at_once("second")

    with statechart.modify(rewrites_compiled_nodes=True):
        sequence.add_node(second)

    assert sequence.nodes == [first, second]
    assert second.parent_node is sequence


def test_a_compiled_sequence_rejects_steps_again_after_a_rewriting_modification(
    statechart_executor: StatechartExecutor,
) -> None:
    sequence = Sequence(nodes=[_node_arriving_at_once("first")])
    statechart = _compile_with_nodes(statechart_executor, [sequence])
    with statechart.modify(rewrites_compiled_nodes=True):
        sequence.add_node(_node_arriving_at_once("second"))

    with pytest.raises(StatechartAlreadyCompiledError):
        sequence.add_node(_node_arriving_at_once("third"))


def test_a_compiled_sequence_rejects_steps_after_an_abandoned_rewriting_modification(
    statechart_executor: StatechartExecutor,
) -> None:
    sequence = Sequence(nodes=[_node_arriving_at_once("first")])
    statechart = _compile_with_nodes(statechart_executor, [sequence])

    with pytest.raises(ModificationDeliberatelyFailed):
        with statechart.modify(rewrites_compiled_nodes=True):
            raise ModificationDeliberatelyFailed()

    with pytest.raises(StatechartAlreadyCompiledError):
        sequence.add_node(_node_arriving_at_once("rejected"))


def test_a_modification_nested_in_a_rewriting_one_may_rewrite_too(
    statechart_executor: StatechartExecutor,
) -> None:
    sequence = Sequence(nodes=[_node_arriving_at_once("first")])
    statechart = _compile_with_nodes(statechart_executor, [sequence])
    second = _node_arriving_at_once("second")

    with statechart.modify(rewrites_compiled_nodes=True):
        with statechart.modify():
            sequence.add_node(second)

    assert second.parent_node is sequence


def test_a_strategy_may_insert_into_a_compiled_sequence(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    inserted = _node_arriving_at_once("inserted")
    handled = HandledOnFailure(
        handled_node=node,
        failure_handling_strategy=StrategyInsertingBeforeTheFailedStep(inserted),
    )
    sequence = Sequence(nodes=[handled])
    _compile_with_nodes(statechart_executor, [sequence])

    statechart_executor.tick()

    assert sequence.nodes == [inserted, handled]


def test_the_failed_step_runs_again_only_after_the_inserted_one(
    statechart_executor: StatechartExecutor,
) -> None:
    node = NodeFailingItsFirstRuns(failing_runs=1)
    inserted = _node_arriving_at_once("inserted")
    handled = HandledOnFailure(
        handled_node=node,
        failure_handling_strategy=StrategyInsertingBeforeTheFailedStep(inserted),
    )
    sequence = Sequence(nodes=[handled])
    _compile_with_nodes(statechart_executor, [sequence])
    statechart_executor.tick()

    statechart_executor.tick()
    assert inserted.life_cycle_state == LifeCycleValues.RUNNING
    assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node.runs == 1

    _tick_times(statechart_executor, 5)
    assert inserted.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert node.runs == 2
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_inserted_step_failing_fails_the_compiled_sequence(
    statechart_executor: StatechartExecutor,
) -> None:
    inserted = NodeFailingOnObservingFalse(observation=ObservationStateValues.FALSE)
    handled = HandledOnFailure(
        handled_node=NodeFailingItsFirstRuns(failing_runs=1),
        failure_handling_strategy=StrategyInsertingBeforeTheFailedStep(inserted),
    )
    sequence = Sequence(nodes=[handled])
    _compile_with_nodes(statechart_executor, [sequence])

    _tick_times(statechart_executor, 5)

    assert inserted.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.life_cycle_state == LifeCycleValues.FAILED
