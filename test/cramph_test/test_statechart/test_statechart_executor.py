from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto

import pytest

from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    MissingExecutorExtensionError,
    StatechartOfDifferentContextError,
    TickDurationUnknownError,
)
from cramph.executor import (
    Executor,
    ExecutorExtension,
    SimulationPacer,
    StatechartExecutor,
    TickingExecutorExtension,
)
from cramph.monitors import CountSimulationTimeSeconds, CountTicks
from cramph.node import CancelStatechart, EndStatechart, NodeArtifacts
from cramph.nodes_for_testing import ConstTrueNode, NodeAssertionError
from cramph.statechart import Statechart

# %% ticking generic nodes


def test_generic_nodes_tick_until_the_statechart_ends(
    statechart_executor: StatechartExecutor,
) -> None:
    first_step = CountTicks(ticks=2)
    second_step = CountTicks(ticks=3)
    sequence = Sequence(nodes=[first_step, second_step])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence)
    statechart.add_node(EndStatechart.when_true(sequence))

    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end()

    assert statechart.is_ended()
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED
    # +1 for EndStatechart to observe True
    assert statechart_executor.tick_count == first_step.ticks + second_step.ticks + 1


def test_executing_ticks_the_compiled_statechart_until_it_ends(
    statechart_executor: StatechartExecutor,
) -> None:
    step = CountTicks(ticks=2)
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(step)
    statechart.add_node(EndStatechart.when_true(step))
    statechart_executor.compile(statechart)

    statechart_executor.execute()

    assert statechart.is_ended()
    # +1 for EndStatechart to observe True
    assert statechart_executor.tick_count == step.ticks + 1


def test_end_statechart_ends_the_tick_after_it_starts(
    statechart_executor: StatechartExecutor,
) -> None:
    node = ConstTrueNode()
    end = EndStatechart.when_true(node)
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([node, end])
    statechart_executor.compile(statechart)

    statechart_executor.tick()
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not statechart.is_ended()

    statechart_executor.tick()
    assert statechart.is_ended()


def test_cancel_statechart_raises_its_exception(
    statechart_executor: StatechartExecutor,
) -> None:
    node = ConstTrueNode()
    exception = NodeAssertionError(reason="cancelled")
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([node, CancelStatechart.when_true(node, exception)])
    statechart_executor.compile(statechart)

    with pytest.raises(NodeAssertionError) as raised:
        statechart_executor.tick()

    assert raised.value is exception


def test_generic_nodes_build_plain_node_artifacts(
    statechart_executor: StatechartExecutor,
) -> None:
    node = ConstTrueNode()
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([node, EndStatechart.when_true(node)])

    artifacts = node.build()

    assert type(artifacts) is NodeArtifacts


# %% tick duration


def test_simulation_time_is_counted_in_tick_durations(
    statechart_executor: StatechartExecutor,
) -> None:
    ticks = 3
    tick_duration = statechart_executor.context.tick_duration
    counter = CountSimulationTimeSeconds(seconds=ticks * tick_duration)
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([counter, EndStatechart.when_true(counter)])
    statechart_executor.compile(statechart)

    for _ in range(ticks - 1):
        statechart_executor.tick()
    assert counter.observation_state == ObservationStateValues.FALSE

    statechart_executor.tick()
    assert counter.observation_state == ObservationStateValues.TRUE
    assert statechart_executor.time == ticks * tick_duration


def test_a_context_does_not_know_its_tick_duration_by_default(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    assert statechart_context_without_tick_duration.tick_duration is None


def test_a_statechart_ticks_without_knowing_its_tick_duration(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    executor = StatechartExecutor(context=statechart_context_without_tick_duration)
    counter = CountTicks(ticks=2)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes([counter, EndStatechart.when_true(counter)])
    executor.compile(statechart)

    executor.tick_until_end()

    assert statechart.is_ended()


def test_time_is_unknown_without_a_tick_duration(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    executor = StatechartExecutor(context=statechart_context_without_tick_duration)

    with pytest.raises(TickDurationUnknownError):
        executor.time


def test_simulation_time_cannot_be_counted_without_a_tick_duration(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    executor = StatechartExecutor(context=statechart_context_without_tick_duration)
    counter = CountSimulationTimeSeconds(seconds=1.0)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes([counter, EndStatechart.when_true(counter)])
    executor.compile(statechart)

    with pytest.raises(TickDurationUnknownError):
        executor.tick()


def test_a_simulation_cannot_be_paced_without_a_tick_duration(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    with pytest.raises(TickDurationUnknownError):
        StatechartExecutor(
            context=statechart_context_without_tick_duration,
            pacer=SimulationPacer(),
        )


# %% executor extensions


class ExecutorStage(StrEnum):
    """
    The moments at which an executor hands control to its extensions.
    """

    EXTEND_CONTEXT = auto()
    AFTER_COMPILE = auto()
    BEFORE_TICK = auto()
    AFTER_TICK = auto()
    AFTER_RUN = auto()


@dataclass
class StageRecord:
    """
    One stage an extension was called at.
    """

    extension: ExecutorExtension
    """
    The extension that was called.
    """

    stage: ExecutorStage
    """
    The stage it was called at.
    """


@dataclass
class ExtensionRecordingCompileAndRun(ExecutorExtension):
    """
    An executor extension that appends every stage of an :class:`ExecutorExtension` it
    is called at to a shared record.
    """

    records: list[StageRecord]
    """
    The record shared by every extension of one executor.
    """

    def extend_context(self, context: StatechartContext) -> None:
        self._record(ExecutorStage.EXTEND_CONTEXT)

    def after_compile(self, executor: Executor) -> None:
        self._record(ExecutorStage.AFTER_COMPILE)

    def after_run(self, executor: Executor) -> None:
        self._record(ExecutorStage.AFTER_RUN)

    def _record(self, stage: ExecutorStage) -> None:
        """
        :param stage: The stage this extension was called at.
        """
        self.records.append(StageRecord(extension=self, stage=stage))


@dataclass
class ExtensionRecordingItsStages(
    ExtensionRecordingCompileAndRun, TickingExecutorExtension
):
    """
    An executor extension that also records every tick it is called around.
    """

    def before_tick(self, executor: StatechartExecutor) -> None:
        self._record(ExecutorStage.BEFORE_TICK)

    def after_tick(self, executor: StatechartExecutor) -> None:
        self._record(ExecutorStage.AFTER_TICK)


@dataclass
class ExecutorRunningWithoutTicks(Executor):
    """
    An executor that runs a statechart without ticking it, as one that hands the
    statechart to another process does.
    """

    def default_context_extensions(self) -> list[ContextExtension]:
        return []

    def default_executor_extensions(self) -> list[ExecutorExtension]:
        return []

    def compile(self, statechart: Statechart) -> None:
        super().compile(statechart)
        self._notify_compiled()

    def execute(self) -> None:
        self.finish_run()


@dataclass
class CountingContext(ContextExtension):
    """
    A context extension counting how often the nodes that require it were built.
    """

    builds: int = 0
    """
    How many nodes were built with this extension.
    """


@dataclass
class ExtensionInstallingACountingContext(ExecutorExtension):
    """
    An executor extension that makes a :class:`CountingContext` available to the nodes.
    """

    counting_context: CountingContext = field(default_factory=CountingContext)
    """
    The context extension this extension installs.
    """

    def extend_context(self, context: StatechartContext) -> None:
        context.add_extension(self.counting_context)


@dataclass(eq=False, repr=False, kw_only=True)
class NodeRequiringACountingContext(ConstTrueNode):
    """
    A node that can only be built in a context that holds a :class:`CountingContext`.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.context.require_extension(CountingContext).builds += 1
        return super().build_artifacts()


@dataclass
class ExtensionSettingTheTickDuration(ExecutorExtension):
    """
    An executor extension that decides how long a tick lasts.
    """

    tick_duration: float
    """
    How many seconds one tick lasts.
    """

    def extend_context(self, context: StatechartContext) -> None:
        context.set_tick_duration(self.tick_duration)


@dataclass(eq=False, repr=False, kw_only=True)
class NodeFailingToStart(ConstTrueNode):
    """
    A node whose start callback raises, recording whether it was cleaned up.
    """

    cleaned_up: bool = field(default=False, init=False)
    """
    Whether :meth:`cleanup` ran.
    """

    def on_start(self) -> None:
        raise NodeAssertionError(reason="the node failed to start")

    def cleanup(self) -> None:
        self.cleaned_up = True


def _statechart_ending_after_one_node(executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart in the context of `executor` that ends once its only node
        observes True.
    """
    node = ConstTrueNode()
    statechart = Statechart(context=executor.context)
    statechart.add_nodes([node, EndStatechart.when_true(node)])
    return statechart


def test_an_extension_is_called_at_every_stage_of_a_run(
    statechart_context: StatechartContext,
) -> None:
    records: list[StageRecord] = []
    executor = StatechartExecutor(
        context=statechart_context,
        extensions=[ExtensionRecordingItsStages(records)],
    )
    assert [record.stage for record in records] == [ExecutorStage.EXTEND_CONTEXT]

    executor.compile(_statechart_ending_after_one_node(executor))
    assert [record.stage for record in records[1:]] == [ExecutorStage.AFTER_COMPILE]

    executor.tick_until_end()
    assert [record.stage for record in records[2:]] == [
        ExecutorStage.BEFORE_TICK,
        ExecutorStage.AFTER_TICK,
        ExecutorStage.BEFORE_TICK,
        ExecutorStage.AFTER_TICK,
        ExecutorStage.AFTER_RUN,
    ]


def test_an_executor_that_does_not_tick_calls_the_stages_around_its_run(
    statechart_context: StatechartContext,
) -> None:
    records: list[StageRecord] = []
    executor = ExecutorRunningWithoutTicks(
        context=statechart_context,
        extensions=[ExtensionRecordingCompileAndRun(records)],
    )

    executor.compile(_statechart_ending_after_one_node(executor))
    executor.execute()

    assert [record.stage for record in records] == [
        ExecutorStage.EXTEND_CONTEXT,
        ExecutorStage.AFTER_COMPILE,
        ExecutorStage.AFTER_RUN,
    ]


def test_only_ticking_extensions_are_called_around_a_tick(
    statechart_context: StatechartContext,
) -> None:
    records: list[StageRecord] = []
    not_ticking = ExtensionRecordingCompileAndRun(records)
    executor = StatechartExecutor(context=statechart_context, extensions=[not_ticking])

    executor.compile(_statechart_ending_after_one_node(executor))
    executor.tick_until_end()

    assert [record.stage for record in records] == [
        ExecutorStage.EXTEND_CONTEXT,
        ExecutorStage.AFTER_COMPILE,
        ExecutorStage.AFTER_RUN,
    ]


def test_extensions_are_called_in_the_order_they_are_listed(
    statechart_context: StatechartContext,
) -> None:
    records: list[StageRecord] = []
    first = ExtensionRecordingItsStages(records)
    second = ExtensionRecordingItsStages(records)
    executor = StatechartExecutor(
        context=statechart_context, extensions=[first, second]
    )

    executor.compile(_statechart_ending_after_one_node(executor))
    executor.tick()

    assert [record.extension for record in records] == [first, second] * 4


def test_nodes_are_built_with_the_context_extensions_of_every_executor_extension(
    statechart_context: StatechartContext,
) -> None:
    first = ExtensionInstallingACountingContext()
    records: list[StageRecord] = []
    second = ExtensionRecordingItsStages(records)
    executor = StatechartExecutor(
        context=statechart_context, extensions=[first, second]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            node := NodeRequiringACountingContext(),
            EndStatechart.when_true(node),
        ]
    )

    executor.compile(statechart)
    executor.tick_until_end()

    assert first.counting_context.builds == 1
    assert records[-1].stage == ExecutorStage.AFTER_RUN


def test_a_compile_whose_first_tick_raises_tells_the_extensions_the_run_stopped(
    statechart_context: StatechartContext,
) -> None:
    records: list[StageRecord] = []
    executor = StatechartExecutor(
        context=statechart_context,
        extensions=[ExtensionRecordingCompileAndRun(records)],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(NodeFailingToStart())

    with pytest.raises(NodeAssertionError):
        executor.compile(statechart)

    assert [record.stage for record in records] == [
        ExecutorStage.EXTEND_CONTEXT,
        ExecutorStage.AFTER_COMPILE,
        ExecutorStage.AFTER_RUN,
    ]


def test_a_compile_whose_first_tick_raises_cleans_up_the_nodes(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node := NodeFailingToStart())

    with pytest.raises(NodeAssertionError):
        statechart_executor.compile(statechart)

    assert node.cleaned_up


def test_an_extension_can_decide_the_tick_duration_a_pacer_paces(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    extension = ExtensionSettingTheTickDuration(tick_duration=0.02)
    pacer = SimulationPacer()

    StatechartExecutor(
        context=statechart_context_without_tick_duration,
        pacer=pacer,
        extensions=[extension],
    )

    assert pacer.target_frequency == 1 / extension.tick_duration


def test_an_executor_finds_its_extension_by_type(
    statechart_context: StatechartContext,
) -> None:
    extension = ExtensionInstallingACountingContext()
    executor = StatechartExecutor(
        context=statechart_context,
        extensions=[ExtensionRecordingItsStages([]), extension],
    )

    assert executor.require_extension(ExtensionInstallingACountingContext) is extension


def test_requiring_an_extension_the_executor_does_not_have_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    with pytest.raises(MissingExecutorExtensionError):
        statechart_executor.require_extension(ExtensionRecordingItsStages)


# %% default extensions


@dataclass
class ExecutorWithDefaults(StatechartExecutor):
    """
    An executor that brings a :class:`CountingContext` and an
    :class:`ExtensionRecordingItsStages` unless it is given one of each.
    """

    def default_context_extensions(self) -> list[ContextExtension]:
        return [CountingContext()]

    def default_executor_extensions(self) -> list[ExecutorExtension]:
        return [ExtensionRecordingItsStages(records=[])]


def test_an_executor_adds_its_default_context_extensions(
    statechart_context: StatechartContext,
) -> None:
    executor = ExecutorWithDefaults(context=statechart_context)

    assert type(executor.context.require_extension(CountingContext)) is CountingContext


def test_a_supplied_context_extension_replaces_the_default_of_its_type(
    statechart_context: StatechartContext,
) -> None:
    supplied = CountingContext(builds=3)
    statechart_context.add_extension(supplied)

    executor = ExecutorWithDefaults(context=statechart_context)

    assert executor.context.require_extension(CountingContext) is supplied


def test_an_executor_adds_its_default_executor_extensions(
    statechart_context: StatechartContext,
) -> None:
    executor = ExecutorWithDefaults(context=statechart_context)

    assert [type(extension) for extension in executor.extensions] == [
        ExtensionRecordingItsStages
    ]


def test_a_supplied_executor_extension_replaces_the_default_of_its_type(
    statechart_context: StatechartContext,
) -> None:
    supplied = ExtensionRecordingItsStages(records=[])

    executor = ExecutorWithDefaults(context=statechart_context, extensions=[supplied])

    assert [extension is supplied for extension in executor.extensions] == [True]


# %% the context a statechart is built in


def test_an_executor_rejects_a_statechart_built_in_a_different_context(
    statechart_executor: StatechartExecutor,
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    statechart = Statechart(context=statechart_context_without_tick_duration)
    statechart.add_node(ConstTrueNode())

    with pytest.raises(StatechartOfDifferentContextError):
        statechart_executor.compile(statechart)


def test_a_statechart_ticks_in_the_context_it_was_built_in(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(counter := CountTicks(ticks=2))
    statechart.add_node(EndStatechart.when_true(counter))
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end()

    assert statechart.is_ended()
