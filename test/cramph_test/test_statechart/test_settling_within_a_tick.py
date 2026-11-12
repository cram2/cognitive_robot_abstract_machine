from typing_extensions import Callable

import pytest

from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.composites import Parallel, Sequence
from cramph.node import CancelStatechart
from cramph.node import EndStatechart, StatechartNode
from cramph.monitors import Pulse
from cramph.exceptions import TickDoesNotSettleError
from cramph.statechart import Statechart
from cramph.nodes_for_testing import (
    CompositeNodeObservingItsCancellingChildRun,
    CompositeNodeObservingItsSecondChildRun,
    CompositeNodeWithARecordingChild,
    ConstFalseNode,
    ConstTrueNode,
    LifeCycleCallback,
    NodeObservingAPredicate,
    NodeObservingAWrittenVariable,
    NodeObservingTheOppositeOfAnObservationPredicate,
    NodeObservingTrueOnlyOnTick,
    NodeRecordingItsCallbacks,
    NodeWritingAVariableOnStart,
    NodeAssertionError,
)
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world import World

TICKS_TO_WATCH = 8
"""
How many ticks a test ticks through, enough for every chart here to settle.
"""

DEEP_NESTING = 30
"""
How many sequences are nested around one task in the deepest statechart here, more than
a fixed number of passes per tick would settle.
"""


def _compile(statechart: Statechart) -> StatechartExecutor:
    """
    :param statechart: The statechart to run.
    :return: An executor that compiled `statechart`, which already ticked once.
    """
    executor = StatechartExecutor(statechart.context)
    executor.compile(statechart=statechart)
    return executor


def _tick_until(executor: StatechartExecutor, happened: Callable[[], bool]) -> None:
    """
    Ticks `executor` until `happened` is true, which it may already be.

    :param executor: The executor to tick.
    :param happened: Whether the awaited event has happened by now.
    """
    for _ in range(TICKS_TO_WATCH):
        if happened():
            return
        executor.tick()
    assert happened()


def _first_ticks(
    executor: StatechartExecutor, events: list[Callable[[], bool]]
) -> list[int | None]:
    """
    :param executor: The executor to tick, whose compile tick counts as tick 0.
    :param events: Whether each awaited event has happened by now.
    :return: Per event, the first tick after which it had happened, or None if
        it did not within :data:`TICKS_TO_WATCH`.
    """
    first_ticks: list[int | None] = [None] * len(events)
    for tick in range(TICKS_TO_WATCH + 1):
        if tick > 0:
            executor.tick()
        for index, happened in enumerate(events):
            if first_ticks[index] is None and happened():
                first_ticks[index] = tick
    return first_ticks


def _nested_sequence_chart() -> Statechart:
    """
    :return: A statechart whose only step finishes two composite levels above its task,
        so finishing it takes more than one pass through the statechart.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(Sequence(nodes=[Sequence(nodes=[ConstTrueNode()])]))
    return statechart


# %% reaction time does not depend on nesting


def _bare(task: ConstTrueNode) -> StatechartNode:
    task.success_condition = task.observes_true
    return task


def _in_a_sequence(task: ConstTrueNode) -> StatechartNode:
    return Sequence(nodes=[task])


def _in_nested_sequences(task: ConstTrueNode) -> StatechartNode:
    return Sequence(nodes=[Sequence(nodes=[Sequence(nodes=[task])])])


class TestReactionTimeAcrossNesting:
    """
    A node waiting for a step reacts on the tick the step's task reaches its goal,
    however many composite levels lie between the two.
    """

    @pytest.mark.parametrize(
        "make_step",
        [_bare, _in_a_sequence, _in_nested_sequences],
        ids=["bare task", "task in a sequence", "task in nested sequences"],
    )
    def test_a_node_waiting_on_a_step_starts_on_the_tick_its_task_reaches_its_goal(
        self, make_step: Callable[[ConstTrueNode], StatechartNode]
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        task = ConstTrueNode()
        step = make_step(task)
        waiting = ConstFalseNode()
        statechart.add_nodes([step, waiting])
        waiting.start_condition = step.is_succeeded
        executor = _compile(statechart)

        goal_reached_tick, started_tick = _first_ticks(
            executor,
            [
                lambda: statechart.observation_state[task]
                == ObservationStateValues.TRUE,
                lambda: waiting.life_cycle_state == LifeCycleValues.RUNNING,
            ],
        )

        assert started_tick == goal_reached_tick

    @pytest.mark.parametrize(
        "make_step",
        [_bare, _in_a_sequence, _in_nested_sequences],
        ids=["bare task", "task in a sequence", "task in nested sequences"],
    )
    def test_the_statechart_ends_on_the_tick_after_a_step_reaches_its_goal(
        self, make_step: Callable[[ConstTrueNode], StatechartNode]
    ) -> None:
        """
        The EndStatechart node starts on the tick the step's task reaches its goal and,
        like any node started during a tick, first observes on the next one.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        task = ConstTrueNode()
        step = make_step(task)
        statechart.add_nodes([step, EndStatechart.when_true(step)])
        executor = _compile(statechart)

        goal_reached_tick, end_tick = _first_ticks(
            executor,
            [
                lambda: statechart.observation_state[task]
                == ObservationStateValues.TRUE,
                statechart.is_ended,
            ],
        )

        assert end_tick == goal_reached_tick + 1

    def test_a_parent_reads_the_outcome_its_child_reaches_on_the_same_tick(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        child = ConstTrueNode()
        parallel = Parallel(nodes=[child])
        statechart.add_node(parallel)
        child.success_condition = child.observes_true
        parallel.success_condition = child.is_succeeded
        executor = _compile(statechart)

        _tick_until(
            executor, lambda: child.life_cycle_state == LifeCycleValues.SUCCEEDED
        )

        assert parallel.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_a_parent_reads_what_its_child_observes_on_the_same_tick(self) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        child = ConstTrueNode()
        parallel = Parallel(nodes=[child])
        statechart.add_node(parallel)
        parallel.success_condition = child.observes_true
        executor = _compile(statechart)

        _tick_until(
            executor,
            lambda: statechart.observation_state[child] == ObservationStateValues.TRUE,
        )

        assert parallel.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_an_observation_reads_the_outcome_another_node_reaches_on_the_same_tick(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        watched = ConstTrueNode()
        observer = NodeObservingAPredicate(watched_node=watched)
        statechart.add_nodes([watched, observer])
        watched.success_condition = watched.observes_true
        executor = _compile(statechart)

        _tick_until(
            executor, lambda: watched.life_cycle_state == LifeCycleValues.SUCCEEDED
        )

        assert statechart.observation_state[observer] == ObservationStateValues.TRUE


# %% what is decided once per tick


class TestOncePerTick:
    """
    Python code on a node runs once per tick, however often the statechart has to be
    evaluated before that tick settles.
    """

    def test_on_tick_is_called_once_per_tick(self) -> None:
        statechart = _nested_sequence_chart()
        statechart.add_node(ticked := NodeObservingTrueOnlyOnTick())
        executor = _compile(statechart)

        for _ in range(TICKS_TO_WATCH):
            executor.tick()

        assert ticked.on_tick_calls == TICKS_TO_WATCH

    def test_what_on_tick_returns_is_the_observation_a_settled_tick_ends_with(
        self,
    ) -> None:
        statechart = _nested_sequence_chart()
        statechart.add_node(ticked := NodeObservingTrueOnlyOnTick())
        executor = _compile(statechart)

        observations = []
        for _ in range(TICKS_TO_WATCH):
            executor.tick()
            observations.append(statechart.observation_state[ticked])

        assert observations == [ObservationStateValues.TRUE] * TICKS_TO_WATCH

    def test_a_node_started_this_tick_first_observes_on_the_next_one(self) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_nodes(
            [trigger := ConstTrueNode(), started_late := ConstTrueNode()]
        )
        started_late.start_condition = trigger.observes_true
        executor = _compile(statechart)

        executor.tick()
        assert started_late.life_cycle_state == LifeCycleValues.RUNNING
        assert (
            statechart.observation_state[started_late] == ObservationStateValues.UNKNOWN
        )

        executor.tick()
        assert statechart.observation_state[started_late] == ObservationStateValues.TRUE

    def test_what_a_start_callback_writes_is_observed_from_the_next_tick(self) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        writer = NodeWritingAVariableOnStart()
        reader = NodeObservingAWrittenVariable(writer=writer)
        statechart.add_nodes([trigger := ConstTrueNode(), writer, reader])
        writer.start_condition = trigger.observes_true
        executor = _compile(statechart)

        executor.tick()
        assert writer.life_cycle_state == LifeCycleValues.RUNNING
        assert statechart.observation_state[reader] == ObservationStateValues.FALSE

        executor.tick()
        assert statechart.observation_state[reader] == ObservationStateValues.TRUE


# %% ticks that never settle


class TestUnsettledTick:
    """
    A tick whose passes keep changing the statechart is stopped instead of blocking the
    control loop.
    """

    def test_observations_contradicting_each_other_stop_the_tick(self) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        first = NodeObservingTheOppositeOfAnObservationPredicate()
        second = NodeObservingTheOppositeOfAnObservationPredicate(watched_node=first)
        first.watched_node = second
        statechart.add_nodes([first, second])
        executor = _compile(statechart)

        with pytest.raises(TickDoesNotSettleError) as error:
            executor.tick()

        assert error.value.pass_limit == statechart.pass_limit
        assert error.value.unsettled_nodes == [first, second]

    def test_observations_contradicting_each_other_are_stopped_once_they_repeat(
        self,
    ) -> None:
        """
        A tick that returns to a state it already had can never settle, so it is stopped
        right away instead of using up the passes a tick may take.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        first = NodeObservingTheOppositeOfAnObservationPredicate()
        second = NodeObservingTheOppositeOfAnObservationPredicate(watched_node=first)
        first.watched_node = second
        statechart.add_nodes([first, second])
        executor = _compile(statechart)

        with pytest.raises(TickDoesNotSettleError) as error:
            executor.tick()

        assert error.value.passes_taken < error.value.pass_limit

    def test_deeply_nested_steps_settle_within_one_tick(self) -> None:
        """
        Every nesting level may need its own passes to pass an outcome on, which is no
        reason to stop a tick that does settle.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        plan = ConstTrueNode()
        for _ in range(DEEP_NESTING):
            plan = Sequence(nodes=[plan])
        statechart.add_node(plan)
        executor = _compile(statechart)

        _tick_until(
            executor, lambda: plan.life_cycle_state == LifeCycleValues.SUCCEEDED
        )

    def test_a_pass_limit_given_to_a_statechart_still_applies_after_it_compiled_again(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()), pass_limit=7)
        first = NodeObservingTheOppositeOfAnObservationPredicate()
        second = NodeObservingTheOppositeOfAnObservationPredicate(watched_node=first)
        first.watched_node = second
        statechart.add_nodes([first, second])
        executor = _compile(statechart)
        statechart.add_node(ConstTrueNode())

        with pytest.raises(TickDoesNotSettleError) as error:
            executor.tick()

        assert error.value.pass_limit == statechart.pass_limit

    def test_a_statechart_given_no_passes_per_nesting_level_stops_deeply_nested_steps(
        self,
    ) -> None:
        """
        The passes a tick may take grow with the nesting only by what the statechart was
        given per nesting level.
        """
        statechart = Statechart(
            context=StatechartContext(world=World()), passes_per_nesting_level=0
        )
        plan = ConstTrueNode()
        for _ in range(DEEP_NESTING):
            plan = Sequence(nodes=[plan])
        statechart.add_node(plan)
        executor = _compile(statechart)

        with pytest.raises(TickDoesNotSettleError) as error:
            _tick_until(
                executor, lambda: plan.life_cycle_state == LifeCycleValues.SUCCEEDED
            )

        assert error.value.pass_limit == statechart.pass_limit
        assert error.value.passes_taken == statechart.pass_limit


# %% life cycle callbacks


def _callbacks_per_tick(
    executor: StatechartExecutor, node: NodeRecordingItsCallbacks
) -> list[list[LifeCycleCallback]]:
    """
    :param executor: The executor to tick, which has compiled but not ticked since.
    :param node: The node whose callbacks to record.
    :return: The callbacks run on `node` on every tick, starting with the
        compile tick.
    """
    callbacks = [node.take_callbacks()]
    for _ in range(TICKS_TO_WATCH):
        executor.tick()
        callbacks.append(node.take_callbacks())
    return callbacks


class TestLifeCycleCallbacks:
    """
    Tests which callbacks run on a node whose life cycle changes more than once around
    the same tick.
    """

    def test_a_node_cut_off_right_after_starting_runs_its_start_then_its_end(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_node(composite := CompositeNodeObservingItsSecondChildRun())
        composite.success_condition = composite.observes_true
        executor = _compile(statechart)

        for _ in range(TICKS_TO_WATCH):
            executor.tick()

        assert composite.second.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert composite.second.take_callbacks() == [
            LifeCycleCallback.START,
            LifeCycleCallback.END,
        ]

    def test_a_cancel_statechart_cut_off_right_after_starting_still_cancels_the_statechart(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_node(composite := CompositeNodeObservingItsCancellingChildRun())
        composite.success_condition = composite.observes_true
        executor = _compile(statechart)

        with pytest.raises(NodeAssertionError) as error:
            for _ in range(TICKS_TO_WATCH):
                executor.tick()

        assert error.value is composite.cancel.exception

    def test_a_cancel_statechart_lets_the_tick_it_starts_in_complete(self) -> None:
        """
        Cancelling the statechart ends it after the tick, so every callback of that tick
        still runs and the tick is still recorded.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        trigger = ConstTrueNode()
        cancel = CancelStatechart(
            exception=NodeAssertionError(reason="cancelled on the first goal")
        )
        ended = NodeRecordingItsCallbacks()
        statechart.add_nodes([trigger, cancel, ended])
        cancel.start_condition = trigger.observes_true
        ended.interrupt_condition = trigger.observes_true
        executor = _compile(statechart)

        with pytest.raises(NodeAssertionError) as error:
            executor.tick()

        assert error.value is cancel.exception
        assert ended.take_callbacks() == [
            LifeCycleCallback.START,
            LifeCycleCallback.END,
        ]
        last_snapshot = statechart.state_history.latest_item
        assert last_snapshot.tick_count == executor.tick_count
        assert last_snapshot.life_cycle_state[cancel] == LifeCycleValues.RUNNING

    def test_a_node_starting_while_its_pause_condition_holds_starts_paused(
        self,
    ) -> None:
        """
        A node that would be paused right away never runs, not even for the control tick
        it starts in, and still gets both callbacks in order.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_node(paused := NodeRecordingItsCallbacks())
        paused.pause_condition = Scalar.const_true()
        executor = _compile(statechart)

        assert paused.life_cycle_state == LifeCycleValues.PAUSED
        assert statechart.state_history.get_life_cycle_history_of_node(paused) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.PAUSED,
        ]
        assert (
            _callbacks_per_tick(executor, paused)
            == [[LifeCycleCallback.START, LifeCycleCallback.PAUSE]]
            + [[]] * TICKS_TO_WATCH
        )

    def test_a_node_that_restarts_after_failing_takes_one_step_per_tick(self) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_nodes(
            [trigger := ConstTrueNode(), restarting := NodeRecordingItsCallbacks()]
        )
        restarting.fail_condition = trigger.observes_true
        restarting.reset_condition = restarting.is_failed
        executor = _compile(statechart)

        callbacks = _callbacks_per_tick(executor, restarting)

        restart_loop = [
            [LifeCycleCallback.START],
            [LifeCycleCallback.END],
            [LifeCycleCallback.RESET],
        ]
        assert callbacks == (restart_loop * TICKS_TO_WATCH)[: TICKS_TO_WATCH + 1]

    def test_a_child_reset_by_its_parent_starts_again_only_on_the_next_tick(
        self,
    ) -> None:
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_nodes(
            [
                pulse := Pulse(),
                composite := CompositeNodeWithARecordingChild(),
            ]
        )
        composite.reset_condition = pulse.observes_true
        executor = _compile(statechart)

        callbacks = _callbacks_per_tick(executor, composite.child)

        assert callbacks == [
            [LifeCycleCallback.START],
            [LifeCycleCallback.RESET],
            [LifeCycleCallback.START],
        ] + [[]] * (TICKS_TO_WATCH - 2)

    def test_a_child_forced_through_pause_end_and_reset_runs_each_callback_once(
        self,
    ) -> None:
        """
        Each ancestor of the child takes one transition of its own, triggered by what
        the level below it did on the previous pass, so the child is paused, ended and
        reset within one tick and only starts again on the next one.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        child = NodeRecordingItsCallbacks()
        trigger = ConstTrueNode()
        inner = Parallel(nodes=[trigger, child])
        middle = Parallel(nodes=[inner])
        outer = Parallel(nodes=[middle])
        statechart.add_node(outer)
        inner.pause_condition = trigger.observes_true
        middle.success_condition = inner.is_paused
        outer.reset_condition = middle.is_succeeded
        executor = _compile(statechart)

        callbacks = _callbacks_per_tick(executor, child)

        forced_through_and_restarted = [
            [LifeCycleCallback.PAUSE, LifeCycleCallback.END, LifeCycleCallback.RESET],
            [LifeCycleCallback.START],
        ]
        assert (
            callbacks
            == [[LifeCycleCallback.START]]
            + (forced_through_and_restarted * TICKS_TO_WATCH)[:TICKS_TO_WATCH]
        )

    def test_nodes_pausing_each_other_alternate_once_per_tick(self) -> None:
        """
        Each node pauses while the other runs, which no single consistent state
        satisfies, so each node takes one step per tick rather than the chart being
        rejected.
        """
        statechart = Statechart(context=StatechartContext(world=World()))
        statechart.add_nodes(
            [
                first := NodeRecordingItsCallbacks(),
                second := NodeRecordingItsCallbacks(),
            ]
        )
        first.pause_condition = second.is_running
        second.pause_condition = first.is_running
        executor = _compile(statechart)

        callbacks = _callbacks_per_tick(executor, first)

        alternating = [[LifeCycleCallback.PAUSE], [LifeCycleCallback.UNPAUSE]]
        assert (
            callbacks
            == [[LifeCycleCallback.START]]
            + (alternating * TICKS_TO_WATCH)[:TICKS_TO_WATCH]
        )
