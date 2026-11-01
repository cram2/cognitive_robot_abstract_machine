from __future__ import annotations

from dataclasses import dataclass

import pytest

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import StatechartAlreadyCompiledError
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart

# %% mimics


class ModificationDeliberatelyFailed(Exception):
    """
    Raised inside a modification block to abandon it.
    """


@dataclass
class ExtensionCountingCompiles(TickingExecutorExtension):
    """
    An executor extension that counts how often it was told the statechart compiled.
    """

    compile_count: int = 0
    """
    How often :meth:`after_compile` ran.
    """

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.compile_count += 1


@dataclass
class ExtensionTellingRecompilesApart(TickingExecutorExtension):
    """
    An executor extension that counts separately how often it was told the statechart
    compiled and how often that it compiled again.
    """

    compile_count: int = 0
    """
    How often :meth:`after_compile` ran.
    """

    recompile_count: int = 0
    """
    How often :meth:`after_recompile` ran.
    """

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.compile_count += 1

    def after_recompile(self, executor: StatechartExecutor) -> None:
        self.recompile_count += 1


# %% helpers


def _node_arriving_at_once(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _compile_and_run_one_node(
    executor: StatechartExecutor,
) -> NodeSucceedingOnObservingTrue:
    """
    Compile a statechart holding one node and tick it until that node succeeded.

    :return: The node, succeeded.
    """
    statechart = Statechart(context=executor.context)
    statechart.add_node(first := _node_arriving_at_once("first"))
    executor.compile(statechart)
    executor.tick()
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    return first


def _tick_until_succeeded(
    executor: StatechartExecutor, node: NodeSucceedingOnObservingTrue
) -> None:
    """
    Tick `executor` for as long as a node arriving at once needs: one tick to start and
    one to observe.
    """
    executor.tick()
    executor.tick()
    assert node.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% growing


def test_a_node_added_in_a_modification_runs_after_the_node_it_waits_for(
    statechart_executor: StatechartExecutor,
) -> None:
    first = _compile_and_run_one_node(statechart_executor)
    statechart = statechart_executor.statechart
    second = _node_arriving_at_once("second")
    second.start_condition = first.is_succeeded

    with statechart.modify():
        statechart.add_node(second)
    statechart_executor.tick()
    assert second.life_cycle_state == LifeCycleValues.RUNNING
    statechart_executor.tick()

    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_added_outside_a_modification_is_compiled_right_away(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    second = _node_arriving_at_once("second")

    statechart_executor.statechart.add_node(second)

    _tick_until_succeeded(statechart_executor, second)


def test_growing_keeps_the_outcome_of_the_nodes_already_there(
    statechart_executor: StatechartExecutor,
) -> None:
    first = _compile_and_run_one_node(statechart_executor)

    statechart_executor.statechart.add_node(_node_arriving_at_once("second"))
    statechart_executor.tick()

    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_growing_does_not_restart_the_tick_count(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    ticks_before = statechart_executor.tick_count

    statechart_executor.statechart.add_node(_node_arriving_at_once("second"))

    assert statechart_executor.tick_count == ticks_before


def test_the_history_of_an_added_node_starts_when_it_joined(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    history = statechart_executor.statechart.state_history
    recorded_before_joining = len(history)
    second = _node_arriving_at_once("second")

    statechart_executor.statechart.add_node(second)
    statechart_executor.tick()

    recorded_since_joining = len(history) - recorded_before_joining
    assert len(history.get_life_cycle_history_of_node(second)) == recorded_since_joining
    assert second.start_time is not None


def test_a_history_item_reads_the_same_once_a_node_joined(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    recorded_before_joining = statechart_executor.statechart.state_history.latest_item
    representation_before_joining = repr(recorded_before_joining)

    statechart_executor.statechart.add_node(_node_arriving_at_once("second"))

    assert repr(recorded_before_joining) == representation_before_joining


def test_an_added_composite_node_runs_its_children(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    steps = [_node_arriving_at_once("step one"), _node_arriving_at_once("step two")]

    statechart_executor.statechart.add_node(Sequence(nodes=steps))

    for step in steps:
        _tick_until_succeeded(statechart_executor, step)


# %% telling the extensions


def test_an_extension_is_told_apart_that_the_statechart_compiled_again(
    statechart_executor: StatechartExecutor,
) -> None:
    telling_apart = ExtensionTellingRecompilesApart()
    statechart_executor.extensions.append(telling_apart)
    _compile_and_run_one_node(statechart_executor)

    statechart_executor.statechart.add_node(_node_arriving_at_once("second"))

    assert telling_apart.compile_count == 1
    assert telling_apart.recompile_count == 1


def test_an_extension_not_telling_them_apart_takes_a_recompile_as_a_compile(
    statechart_executor: StatechartExecutor,
) -> None:
    counting = ExtensionCountingCompiles()
    statechart_executor.extensions.append(counting)
    _compile_and_run_one_node(statechart_executor)

    statechart_executor.statechart.add_node(_node_arriving_at_once("second"))

    assert counting.compile_count == 2


# %% batching


def test_nested_modifications_compile_once_when_the_outermost_one_ends(
    statechart_context: StatechartContext,
) -> None:
    counting = ExtensionCountingCompiles()
    executor = StatechartExecutor(context=statechart_context, extensions=[counting])
    _compile_and_run_one_node(executor)
    statechart = executor.statechart
    compiles_before = counting.compile_count

    with statechart.modify():
        with statechart.modify():
            statechart.add_node(_node_arriving_at_once("second"))
        statechart.add_nodes(
            [_node_arriving_at_once("third"), _node_arriving_at_once("fourth")]
        )
        assert counting.compile_count == compiles_before

    assert counting.compile_count == compiles_before + 1


def test_a_modification_on_a_statechart_not_compiled_yet_only_adds(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)

    with statechart.modify():
        statechart.add_node(first := _node_arriving_at_once("first"))

    assert not statechart.compiler.is_compiled
    statechart_executor.compile(statechart)
    _tick_until_succeeded(statechart_executor, first)


def test_an_abandoned_modification_leaves_the_statechart_as_it_was(
    statechart_executor: StatechartExecutor,
) -> None:
    first = _compile_and_run_one_node(statechart_executor)
    statechart = statechart_executor.statechart
    nodes_before = list(statechart.nodes)

    with pytest.raises(ModificationDeliberatelyFailed):
        with statechart.modify():
            statechart.add_node(_node_arriving_at_once("second"))
            raise ModificationDeliberatelyFailed()

    assert statechart.nodes == nodes_before
    statechart_executor.tick()
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_abandoned_inner_modification_keeps_the_outer_one_working(
    statechart_executor: StatechartExecutor,
) -> None:
    _compile_and_run_one_node(statechart_executor)
    statechart = statechart_executor.statechart

    with statechart.modify():
        with pytest.raises(ModificationDeliberatelyFailed):
            with statechart.modify():
                statechart.add_node(_node_arriving_at_once("dropped"))
                raise ModificationDeliberatelyFailed()
        statechart.add_node(kept := _node_arriving_at_once("kept"))

    _tick_until_succeeded(statechart_executor, kept)


def test_a_compiled_composite_node_still_rejects_new_children(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence := Sequence(nodes=[_node_arriving_at_once("first")]))
    statechart_executor.compile(statechart)

    with pytest.raises(StatechartAlreadyCompiledError):
        sequence.add_node(_node_arriving_at_once("second"))


def test_a_child_a_compiled_composite_node_rejects_does_not_join_the_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence := Sequence(nodes=[_node_arriving_at_once("first")]))
    statechart_executor.compile(statechart)
    nodes_before = list(statechart.nodes)
    rejected = _node_arriving_at_once("rejected")

    with pytest.raises(StatechartAlreadyCompiledError):
        sequence.add_node(rejected)

    assert statechart.nodes == nodes_before
    assert not rejected.belongs_to_statechart()


def test_a_child_a_compiled_composite_node_rejects_does_not_become_its_child(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first = _node_arriving_at_once("first")
    statechart.add_node(sequence := Sequence(nodes=[first]))
    statechart_executor.compile(statechart)

    with pytest.raises(StatechartAlreadyCompiledError):
        sequence.add_node(_node_arriving_at_once("rejected"))

    assert sequence.nodes == [first]
