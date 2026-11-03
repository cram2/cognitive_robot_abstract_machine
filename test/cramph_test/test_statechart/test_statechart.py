import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

import cramph.node as node_module
import krrood.symbolic_math.symbolic_math as sm
from cramph.node import EndedByOwner
from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import (
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationPredicate,
    ObservationStateValues,
    TransitionKind,
)
from cramph.exceptions import (
    CancelledByNodeFailing,
    CancelledByNodeReachingItsGoal,
    ChildTransitionAlreadyWiredError,
    NotInStatechartError,
    EndInCompositeNodeError,
    CompositeNodeWithoutChildrenError,
    InputNotExpressionError,
    SelfInStartConditionError,
    UnsupportedConditionVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
    ConditionScopeError,
    TerminalNodeInConditionError,
    TickLimitReachedError,
    UnknownTransitionKindError,
)
from cramph.composites import Sequence, Parallel
from cramph.node import EndStatechart, StatechartNode
from cramph.node import (
    CancelStatechart,
    CompositeNode,
    NodeArtifacts,
    TerminalNode,
    TransitionCondition,
)
from cramph.node import ThreadPayloadMonitor
from cramph.monitors import (
    Print,
    Pulse,
    CountSeconds,
    CountTicks,
    CountSimulationTimeSeconds,
)
from cramph.statechart import Statechart
from cramph.nodes_for_testing import (
    ChangeStateOnEvents,
    CompositeNodeCuttingOffItsChild,
    CompositeNodeCuttingOffItsChildAtItsGoal,
    CompositeNodeCuttingOffItsGrandchild,
    CompositeNodeCuttingOffItsUndecidedChild,
    CompositeNodeWithChildInterruptedBySibling,
    CompositeNodeWithChildFailingOnItsOwn,
    CompositeNodeWithChildStartingLate,
    CompositeNodeWithChildSucceedingOnItsOwn,
    NodeObservingAnObservationPredicate,
    NodeObservingLastObservation,
    NodeObservingNothingYet,
    NodeFailingOnObservingFalse,
    NodeSucceedingOnObservingTrue,
    NodeDeclaringItsOwnFailure,
    NodeDeclaringNoWayToSucceed,
    ConstTrueNode,
    CompositeNodeWithChainedChildren,
    CompositeNodeWithNestedCompositeChild,
    ConstFalseNode,
    CompositeNodeCancellingIfChildRunsAfterItsEnd,
    CompositeNodeCancellingIfPausedChildResumesAfterItsEnd,
    CompositeNodeWithChildSucceedingBeforeItStarts,
    CompositeNodeResumingItsPausedChildren,
)
from krrood.symbolic_math.exceptions import CannotConvertToStringError
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    Scalar,
    logic_and,
    logic_or,
    logic_not,
)
from semantic_digital_twin.world import World

# %% failures a cancelling node is given


class CancellingFailure(Exception):
    """
    The failure a :class:`CancelStatechart` in these tests ends the statechart with.
    """


# %% a clock the test advances instead of waiting


@dataclass
class FakeClock:
    """
    Stands in for :func:`time.monotonic` so tests can advance time without sleeping.
    """

    seconds: float = 0.0
    """
    The current time, in seconds.
    """

    def time(self) -> float:
        """
        :return: The current time, in the shape a node's clock is called in.
        """
        return self.seconds

    def advance(self, seconds: float) -> None:
        """
        :param seconds: How far to move the clock forward.
        """
        self.seconds += seconds


def test_condition_to_str() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndStatechart()
    msc.add_node(end)

    end.start_condition = sm.logic_and(
        node1.observes_true,
        sm.logic_or(
            node2.observes_true,
            sm.logic_not(node3.observes_true),
        ),
    )
    a = str(end._start_condition)
    assert a == (
        '("ConstTrueNode#0.observes_true" and ("ConstTrueNode#1.observes_true" or not '
        '"ConstTrueNode#2.observes_true"))'
    )


def test_statechart_to_dot(tmp_path) -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    end = EndStatechart()
    msc.add_node(end)
    node1.success_condition = node2.observes_true
    end.start_condition = logic_and(node1.observes_true, node2.observes_true)
    msc.draw(str(tmp_path / "muh.pdf"))


def test_print() -> None:
    kin_sim = StatechartExecutor(StatechartContext(world=World()))
    msc = Statechart(context=kin_sim.context)
    print_node1 = Print(name="cow", message="muh")
    msc.add_node(print_node1)
    print_node2 = Print(name="cow2", message="muh")
    msc.add_node(print_node2)

    node1 = ConstTrueNode()
    msc.add_node(node1)
    end = EndStatechart()
    msc.add_node(end)

    node1.start_condition = print_node1.observes_true
    print_node2.start_condition = node1.observes_true
    end.start_condition = print_node2.observes_true

    kin_sim.compile(statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_ended()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_ended()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_ended()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_ended()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_ended()


def test_draw_with_invisible_node(tmp_path) -> None:
    kin_sim = StatechartExecutor(StatechartContext(world=World()))
    msc = Statechart(context=kin_sim.context)
    msc.add_nodes(
        [
            sequence := Sequence(
                nodes=[s1n1 := ConstTrueNode(), s1n2 := ConstTrueNode()]
            ),
            sequence2 := Sequence(
                nodes=[s2n1 := ConstTrueNode(), s2n2 := ConstTrueNode()]
            ),
        ]
    )
    msc.add_node(EndStatechart.when_all_true([sequence, sequence2]))

    sequence.plot_specifications.visible = False
    s1n2.plot_specifications.visible = False
    s2n2.plot_specifications.visible = False

    kin_sim.compile(statechart=msc)
    msc.draw(str(tmp_path / "muh.pdf"))


@dataclass(eq=False, repr=False, kw_only=True)
class _NodeThatEndsTheStatechart(TerminalNode):
    """
    A terminal node other than the two the statechart ships with.
    """


class TestConditions:
    def test_trinary_condition_default_expression_is_scalar(self) -> None:
        condition = TransitionCondition(kind=TransitionKind.START)
        assert isinstance(condition.expression, sm.Scalar)

    def test_InvalidConditionError(self) -> None:
        node = ConstTrueNode()
        with pytest.raises(InputNotExpressionError):
            node.success_condition = node

    def test_nodes_cannot_have_themselves_as_start_condition(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        node1 = ConstTrueNode()
        msc.add_node(node1)
        with pytest.raises(SelfInStartConditionError):
            node1.start_condition = node1.observes_true

    def test_unsupported_variable_in_condition(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())
        with pytest.raises(UnsupportedConditionVariableError):
            node.start_condition = FloatVariable(name="muh")

    @pytest.mark.parametrize(
        "read_observation",
        [lambda node: node.observation_variable, lambda node: node.last_observation],
        ids=["observation_variable", "last_observation"],
    )
    def test_a_condition_may_not_read_a_trinary_observation(
        self, read_observation
    ) -> None:
        """
        A condition is two-valued, so it asks about an observation through a predicate
        rather than reading the observation, which may be Unknown.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([watched := ConstTrueNode(), node := ConstTrueNode()])

        with pytest.raises(UnsupportedConditionVariableError) as exception_info:
            node.start_condition = read_observation(watched)

        assert exception_info.value.unsupported_variable is read_observation(watched)

    def test_an_unknown_constant_condition_is_rejected(self) -> None:
        """
        A condition is two-valued, so it has no Unknown to be set to.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())

        with pytest.raises(CannotConvertToStringError):
            node.pause_condition = sm.Scalar.const_trinary_unknown()

    def test_a_condition_built_from_trinary_operators_is_rejected(self) -> None:
        """
        Only the two-valued operators have a rendered form, so a condition combining
        predicates with the trinary ones is rejected when it is set.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                first := ConstTrueNode(),
                second := ConstTrueNode(),
                node := ConstTrueNode(),
            ]
        )

        with pytest.raises(CannotConvertToStringError):
            node.start_condition = sm.trinary_logic_and(
                first.observes_true, second.observes_true
            )

    def test_end_statechart_cannot_gate_another_node(self) -> None:
        """
        The statechart is over once an EndStatechart is true, so no transition can
        depend on it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([node := ConstTrueNode(), end := EndStatechart()])
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = end.observes_true

        assert exception_info.value.terminal_node is end

    def test_cancel_statechart_cannot_gate_another_node(self) -> None:
        """
        A CancelStatechart ends the statechart just like an EndStatechart does.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        cancel = CancelStatechart(exception=Exception("cancelled"))
        msc.add_nodes([node := ConstTrueNode(), cancel])
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = cancel.observes_true

        assert exception_info.value.terminal_node is cancel

    def test_terminal_nodes_are_rejected_in_every_condition_kind(self) -> None:
        """
        No transition of any kind can happen after the statechart has ended.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([node := ConstTrueNode(), end := EndStatechart()])
        with pytest.raises(TerminalNodeInConditionError):
            node.pause_condition = end.observes_true
        with pytest.raises(TerminalNodeInConditionError):
            node.success_condition = end.observes_true
        with pytest.raises(TerminalNodeInConditionError):
            node.reset_condition = end.observes_true

    def test_any_terminal_node_cannot_gate_another_node(self) -> None:
        """
        The rule follows from ending the statechart, not from being one of the two nodes
        that happen to do so today.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [node := ConstTrueNode(), terminal := _NodeThatEndsTheStatechart()]
        )
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = terminal.observes_true

        assert exception_info.value.terminal_node is terminal

    def test_a_terminal_node_cannot_reference_itself(self) -> None:
        """
        A terminal node's own transitions are as unreachable as everyone else's.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(end := EndStatechart())
        with pytest.raises(TerminalNodeInConditionError):
            end.success_condition = end.observes_true

    def test_add_node_to_multiple_goals(self) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        node = ConstTrueNode()
        msc.add_node(Sequence(nodes=[node]))

        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            msc.add_node(Sequence(nodes=[node]))

    def test_add_node_to_multiple_goals2(self) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        node = ConstTrueNode()
        msc.add_node(node)

        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            msc.add_node(Sequence(nodes=[node]))


@dataclass(eq=False, repr=False, kw_only=True)
class _BuildCountingNode(EndedByOwner, StatechartNode):
    """
    Node that records how often :meth:`build` is invoked.
    """

    build_count: int = field(default=0, init=False)
    """
    Number of times build() has run on this node.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False, kw_only=True)
class _BuildCountingCompositeNode(EndedByOwner, CompositeNode):
    """
    Composite statechart node that records its own build calls and owns a counting child
    node.
    """

    build_count: int = field(default=0, init=False)
    """
    Number of times build() has run on this goal.
    """

    child: _BuildCountingNode = field(default=None, init=False)
    """
    The child node expanded by this goal.
    """

    def expand(self) -> None:
        self.child = _BuildCountingNode(name="counting_child")
        self._add_child_to_statechart(self.child)

    def build_artifacts(self) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=self.child.observation_variable)


def _compile_msc(msc: Statechart) -> StatechartExecutor:
    executor = StatechartExecutor(msc.context)
    executor.compile(statechart=msc)
    return executor


def test_each_node_is_built_exactly_once() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    goal = _BuildCountingCompositeNode()
    msc.add_node(goal)
    msc.add_node(EndStatechart.when_true(goal))

    _compile_msc(msc)

    assert goal.build_count == 1
    assert goal.child.build_count == 1


# %% goals populated before compile


def test_adding_the_same_node_to_a_goal_twice_makes_it_its_child_once() -> None:
    """
    A goal that already holds a node does not hold it a second time.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(goal := Sequence())
    node = ConstTrueNode()

    goal.add_node(node)
    goal.add_node(node)

    assert goal.nodes == [node]


def test_node_added_to_a_joined_goal_joins_the_statechart_right_away() -> None:
    """
    A node added to a goal that already joined a statechart joins it below the goal.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(goal := Sequence())
    goal.add_node(node := ConstTrueNode())

    assert node in msc.nodes
    assert node.parent_node is goal


def test_goal_populated_before_compile_matches_one_populated_by_expand() -> None:
    """
    Adding a sequence's children up front yields the same children, wiring and
    observation as passing them to the template's constructor and letting expand add
    them.
    """
    populated_before_compile = Statechart(context=StatechartContext(world=World()))
    goal = Sequence()
    populated_before_compile.add_node(goal)
    goal.add_node(ConstTrueNode(name="a"))
    goal.add_node(ConstTrueNode(name="b"))
    populated_before_compile.add_node(EndStatechart.when_true(goal))

    populated_by_expand = Statechart(context=StatechartContext(world=World()))
    expanded_goal = Sequence(nodes=[ConstTrueNode(name="a"), ConstTrueNode(name="b")])
    populated_by_expand.add_node(expanded_goal)
    populated_by_expand.add_node(EndStatechart.when_true(expanded_goal))

    for msc in (populated_before_compile, populated_by_expand):
        executor = _compile_msc(msc)
        while not msc.is_ended():
            executor.tick()

    assert [node.name for node in goal.nodes] == ["a", "b"]
    assert [node.name for node in expanded_goal.nodes] == ["a", "b"]
    assert sorted(node.name for node in populated_before_compile.nodes) == sorted(
        node.name for node in populated_by_expand.nodes
    )
    # expand still wires the sequence: the second child waits for the first to succeed
    for sequence in (goal, expanded_goal):
        assert sequence.nodes[1].start_condition.free_variables() == [
            sequence.nodes[0].is_succeeded
        ]
    assert goal.observation_state == expanded_goal.observation_state
    assert populated_before_compile.is_ended()
    assert populated_by_expand.is_ended()


# %% build orchestration and artifact production


@dataclass(eq=False, repr=False, kw_only=True)
class _SetupThenArtifactsNode(EndedByOwner, StatechartNode):
    """
    Node that performs setup in :meth:`build` and describes itself in
    :meth:`build_artifacts`.
    """

    hook_calls: list[str] = field(default_factory=list, init=False)
    """
    Names of the build hooks that ran, in the order they ran.
    """

    def build(self) -> NodeArtifacts:
        self.hook_calls.append("build")
        return super().build()

    def build_artifacts(self) -> NodeArtifacts:
        self.hook_calls.append("build_artifacts")
        return NodeArtifacts(observation=sm.Scalar.const_true())


def test_build_delegates_to_build_artifacts() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node = _SetupThenArtifactsNode()
    msc.add_node(node)
    msc.add_node(EndStatechart.when_true(node))

    executor = _compile_msc(msc)

    assert node.hook_calls == ["build", "build_artifacts"]
    executor.tick()
    assert node.observation_state == ObservationStateValues.TRUE


def test_a_node_class_declaring_no_way_to_succeed_cannot_be_constructed() -> None:
    """
    Every node class has to say how it succeeds, so a node whose class leaves it open
    cannot be constructed.
    """
    with pytest.raises(TypeError):
        NodeDeclaringNoWayToSucceed()


def test_asking_for_a_condition_of_no_transition_kind_raises() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(node := ConstTrueNode())

    with pytest.raises(UnknownTransitionKindError):
        node.get_condition("not a transition kind")


def test_state_iteration_yields_nodes() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)

    assert list(iter(msc.life_cycle_state)) == msc.nodes
    assert dict(msc.observation_state).keys() == {node1, node2}


@dataclass(eq=False, repr=False, kw_only=True)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationStateValues.TRUE

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


@dataclass(eq=False, repr=False, kw_only=True)
class _RaisingThreadMonitor(ThreadPayloadMonitor):
    """
    Thread payload monitor whose observation computation always fails.
    """

    def _compute_observation(self) -> float:
        raise RuntimeError("observation failure")


@dataclass(eq=False, repr=False, kw_only=True)
class _SucceedingThreadMonitor(ThreadPayloadMonitor):
    """
    Thread payload monitor whose observation computation succeeds.
    """

    def _compute_observation(self) -> float:
        return ObservationStateValues.TRUE


def test_thread_payload_monitor_non_blocking_and_caching() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    mon = _TestThreadMonitor(
        delay=0.05,
        return_value=ObservationStateValues.TRUE,
    )
    msc.add_node(mon)
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationStateValues.UNKNOWN
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationStateValues.TRUE


def test_thread_payload_monitor_cleanup_stops_worker() -> None:
    monitor = _SucceedingThreadMonitor()
    assert monitor._thread.is_alive()

    monitor.cleanup()

    monitor._thread.join(timeout=1.0)
    assert not monitor._thread.is_alive()


def test_thread_payload_monitor_surfaces_compute_exception() -> None:
    records: list[logging.LogRecord] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CapturingHandler(level=logging.ERROR)
    node_module.logger.addHandler(handler)
    monitor = _RaisingThreadMonitor()
    try:
        monitor.compute_observation()
        for _ in range(100):
            if any(record.levelno >= logging.ERROR for record in records):
                break
            time.sleep(0.02)
        assert any(record.levelno >= logging.ERROR for record in records)
    finally:
        node_module.logger.removeHandler(handler)
        monitor.cleanup()


class TestStatechartLogic:

    def test_transition_triggers(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        changer = ChangeStateOnEvents()
        msc.add_node(changer)

        node1 = Pulse()
        msc.add_node(node1)

        node2 = Pulse()
        msc.add_node(node2)
        node2.start_condition = node1.observes_true

        node3 = Pulse()
        msc.add_node(node3)
        node3.start_condition = logic_and(
            node1.observes_false,
            node2.observes_false,
        )

        node4 = Pulse()
        msc.add_node(node4)
        node4.start_condition = node3.observes_true

        changer.start_condition = node1.observes_true
        changer.pause_condition = node2.observes_true
        changer.interrupt_condition = node3.observes_true
        changer.reset_condition = node4.observes_true

        kin_sim.compile(statechart=msc)

        assert changer.state is None

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.RUNNING
        assert changer.state == "on_start"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.PAUSED
        assert changer.state == "on_pause"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.RUNNING
        assert changer.state == "on_unpause"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        # A node that only records callbacks never decides what it observes, so ending
        # it cannot judge it.
        assert changer.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert changer.state == "on_end"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert changer.state == "on_reset"

    def test_live_state_requires_statechart_membership(self) -> None:
        node = ConstTrueNode()
        # State variables and conditions are available before the node is added.
        assert node.observation_variable is not None
        assert node.life_cycle_variable is not None
        node.pause_condition = node.observes_true
        node.success_condition = node.observes_true
        node.reset_condition = node.observes_true
        # Reading the live state still requires membership in a statechart.
        with pytest.raises(NotInStatechartError):
            _ = node.statechart
        with pytest.raises(NotInStatechartError):
            _ = node.observation_state
        with pytest.raises(NotInStatechartError):
            _ = node.life_cycle_state

    def test_cancel_statechart(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        node1 = ConstTrueNode()
        msc.add_node(node1)
        cancel = CancelStatechart(exception=CancellingFailure())
        msc.add_node(cancel)
        cancel.start_condition = node1.observes_true

        kin_sim.compile(statechart=msc)

        with pytest.raises(CancellingFailure):
            kin_sim.tick()  # cancel starts, which triggers it
        msc.draw(str(tmp_path / "cancelled.pdf"))

    def test_history_records_the_tick_of_each_snapshot(self) -> None:
        """
        Ticks in which nothing changes leave no snapshot, but the snapshots that are
        kept still name the tick they were taken in.
        """
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        counter = CountTicks(name="counter", ticks=10)
        msc.add_node(counter)
        msc.add_node(EndStatechart.when_true(counter))
        executor.compile(statechart=msc)

        ticks = 0
        while not msc.is_ended():
            executor.tick()
            ticks += 1

        assert msc.state_history.items[0].tick_count == 0
        assert msc.state_history.latest_item.tick_count == ticks

    def test_statechart(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        node1 = ConstTrueNode()
        msc.add_node(node1)
        node2 = ConstTrueNode()
        msc.add_node(node2)
        node3 = ConstTrueNode()
        msc.add_node(node3)
        end = EndStatechart()
        msc.add_node(end)

        node1.start_condition = logic_or(node3.observes_true, node2.observes_true)
        end.start_condition = node1.observes_true

        kin_sim.compile(statechart=msc)

        assert len(msc.nodes) == 4
        assert len(msc.edges) == 3
        kin_sim.tick_until_end()

        assert len(msc.state_history) == 5
        # %% node1
        assert msc.state_history.get_life_cycle_history_of_node(node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node2
        assert msc.state_history.get_life_cycle_history_of_node(node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node3
        assert msc.state_history.get_life_cycle_history_of_node(node3) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(node3) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% end
        assert msc.state_history.get_life_cycle_history_of_node(end) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(end) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]

    def test_goal(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        node1 = ConstTrueNode()
        msc.add_node(node1)

        goal = CompositeNodeWithChainedChildren()
        msc.add_node(goal)

        goal.start_condition = node1.observes_true

        end = EndStatechart()
        msc.add_node(end)
        end.start_condition = goal.observes_true

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        assert len(msc.state_history) == 6
        # %% goal
        assert msc.state_history.get_life_cycle_history_of_node(goal) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(goal) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node1
        assert msc.state_history.get_life_cycle_history_of_node(node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% sub_node1
        assert msc.state_history.get_life_cycle_history_of_node(goal.sub_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
        ]
        assert msc.state_history.get_observation_history_of_node(goal.sub_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
        ]
        # %% sub_node2
        assert msc.state_history.get_life_cycle_history_of_node(goal.sub_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(goal.sub_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% sub_node2
        assert msc.state_history.get_life_cycle_history_of_node(end) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(end) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]
        msc.draw(str(tmp_path / "muh.pdf"))

    def test_reset(self, tmp_path) -> None:
        """
        A reset returns a node to NOT_STARTED, from where it starts and observes again.

        The node that triggers the reset is ended by its own goal, so what outlives it
        is its outcome, and both the reset trigger and the end of the statechart read
        that.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        node1 = ConstTrueNode()
        msc.add_node(node1)
        node2 = ConstTrueNode()
        msc.add_node(node2)
        node3 = ConstTrueNode()
        msc.add_node(node3)
        end = EndStatechart()
        msc.add_node(end)
        node1.reset_condition = node2.observes_true
        node2.start_condition = node1.observes_true
        node2.success_condition = node2.observes_true
        node3.start_condition = node2.is_succeeded
        end.start_condition = logic_and(
            node1.observes_true,
            node2.is_succeeded,
            node3.observes_true,
        )

        kin_sim.compile(statechart=msc)
        msc.draw(str(tmp_path / "muh.pdf"))

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_ended()

        # node2 reaches its goal, which both ends it and resets node1.
        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_ended()

        # node1 starts over with nothing observed yet, and node2 stops observing.
        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert msc.is_ended()

    def test_nested_goals(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))

        node1 = ConstTrueNode(name="w")
        msc.add_node(node1)

        outer = CompositeNodeWithNestedCompositeChild()
        msc.add_node(outer)
        outer.start_condition = node1.observes_true

        end = EndStatechart()
        msc.add_node(end)
        end.start_condition = outer.observes_true

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)
        msc_copy = Statechart.from_json(
            new_json_data, context=StatechartContext(world=World())
        )

        for node in msc.nodes:
            assert node.index == msc_copy.get_node_by_index(node.index).index

        kin_sim = StatechartExecutor(msc_copy.context)
        node1 = msc_copy.get_nodes_by_type(ConstTrueNode)[0]
        outer = msc_copy.get_nodes_by_type(CompositeNodeWithNestedCompositeChild)[0]
        end = msc_copy.get_nodes_by_type(EndStatechart)[0]
        kin_sim.compile(statechart=msc_copy)

        assert node1.depth == 0
        assert outer.depth == 0
        assert end.depth == 0
        assert outer.inner.depth == 1
        assert outer.inner.sub_node1.depth == 2
        assert outer.inner.sub_node2.depth == 2

        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc_copy.is_ended()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert msc_copy.is_ended()


def test_counting() -> None:
    clock = FakeClock()

    kin_sim = StatechartExecutor(
        StatechartContext(
            world=World(),
        )
    )
    msc = Statechart(context=kin_sim.context)
    seconds = 1
    msc.add_nodes(
        [counter := CountSeconds(seconds=seconds, _now=clock.time), pulse := Pulse()]
    )

    pulse.start_condition = counter.observes_true
    counter.reset_condition = pulse.observes_true

    msc.add_node(end := EndStatechart())

    end.start_condition = logic_and(counter.observes_true, pulse.observes_false)

    kin_sim.compile(statechart=msc)

    # Advance fake time deterministically without wall-clock sleeps
    step = 0.1
    while not msc.is_ended():
        kin_sim.tick()
        clock.advance(step)
        if kin_sim.tick_count > 1000:
            raise TimeoutError("test stuck")

    # it takes 2 * seconds to finish the counters
    # + 1 for pulse to trigger
    # + 1 for reset
    # + 1 for EndStatechart to transition to RUNNING
    # + 1 for EndStatechart to observe True
    assert np.allclose(seconds * 2 + 0.4, clock.time())


def test_count_ticks() -> None:
    kin_sim = StatechartExecutor(StatechartContext(world=World()))
    msc = Statechart(context=kin_sim.context)
    msc.add_node(counter := CountTicks(ticks=3))
    msc.add_node(EndStatechart.when_true(counter))
    kin_sim.compile(statechart=msc)
    kin_sim.tick_until_end()
    # ending tacks 4 ticks, one to turn EndStatechart to true
    assert kin_sim.tick_count == 3 + 1


def test_count_ticks_returns_false_until_target() -> None:
    node = CountTicks(ticks=3)
    node.on_start()
    assert node.on_tick() == ObservationStateValues.FALSE
    assert node.on_tick() == ObservationStateValues.FALSE
    assert node.on_tick() == ObservationStateValues.TRUE


def test_count_simulation_time_seconds_reaches_target_on_exact_tick(
    statechart_context: StatechartContext,
) -> None:
    context = statechart_context
    ticks_until_true = 4
    seconds = context.tick_duration * ticks_until_true
    node = CountSimulationTimeSeconds(seconds=seconds)
    Statechart(context=context).add_node(node)
    node.on_start()
    for _ in range(ticks_until_true - 1):
        assert node.on_tick() == ObservationStateValues.FALSE
    assert node.on_tick() == ObservationStateValues.TRUE


def test_count_simulation_time_seconds_on_start_resets_counter(
    statechart_context: StatechartContext,
) -> None:
    context = statechart_context
    seconds = context.tick_duration * 2
    node = CountSimulationTimeSeconds(seconds=seconds)
    Statechart(context=context).add_node(node)
    node.on_start()
    node.on_tick()
    node.on_tick()
    node.on_start()
    assert node.on_tick() == ObservationStateValues.FALSE


def test_count_simulation_time_seconds_with_executor(
    statechart_context: StatechartContext,
) -> None:
    context = statechart_context
    ticks_until_true = 20
    seconds = context.tick_duration * ticks_until_true
    kin_sim = StatechartExecutor(context)
    msc = Statechart(context=kin_sim.context)
    msc.add_node(counter := CountSimulationTimeSeconds(seconds=seconds))
    msc.add_node(EndStatechart.when_true(counter))
    kin_sim.compile(statechart=msc)
    kin_sim.tick_until_end()
    # +1 for EndStatechart to turn True, as in test_count_ticks
    assert kin_sim.tick_count == ticks_until_true + 1


class TestEndStatechart:
    def test_end_statechart_when_all_done1(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
        end = EndStatechart.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_statechart_when_all_done2(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndStatechart.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim.compile(statechart=msc)
        with pytest.raises(TickLimitReachedError):
            kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_end_statechart_when_any_done1(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndStatechart.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_statechart_when_any_done2(self, tmp_path) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                ConstFalseNode(),
                ConstFalseNode(),
            ]
        )
        end = EndStatechart.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim.compile(statechart=msc)
        with pytest.raises(TickLimitReachedError):
            kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_end_statechart_when_all_true_accepts_a_single_node(self) -> None:
        """
        A list of one is valid input, so combining it must not depend on there being
        something to combine it with.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(ConstTrueNode())
        msc.add_node(end := EndStatechart.when_all_true(msc.nodes))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_statechart_when_any_true_accepts_a_single_node(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(ConstTrueNode())
        msc.add_node(end := EndStatechart.when_any_true(msc.nodes))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_cancel_statechart_when_all_true_accepts_a_single_node(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(ConstTrueNode())
        cancelled = Exception("cancelled")
        msc.add_node(CancelStatechart.when_all_true(msc.nodes, exception=cancelled))

        kin_sim.compile(statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            kin_sim.tick_until_end()

        assert error.value is cancelled

    def test_cancel_statechart_when_any_true_accepts_a_single_node(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(ConstTrueNode())
        cancelled = Exception("cancelled")
        msc.add_node(CancelStatechart.when_any_true(msc.nodes, exception=cancelled))

        kin_sim.compile(statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            kin_sim.tick_until_end()

        assert error.value is cancelled

    def test_end_statechart_when_failed_waits_for_the_node_to_end(self) -> None:
        """
        Being short of its goal is not yet a failure: the node has to have been ended
        while it was.
        """
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_nodes(
            [
                trigger := CountTicks(ticks=2),
                falling_short := ConstFalseNode(),
                end := EndStatechart.when_failed(falling_short),
            ]
        )
        falling_short.fail_condition = trigger.observes_true

        executor.compile(statechart=msc)
        executor.tick()

        assert falling_short.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()

        assert falling_short.life_cycle_state == LifeCycleValues.FAILED
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_cancel_statechart_when_failed_raises_once_the_node_fails(self) -> None:
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_nodes([trigger := ConstTrueNode(), falling_short := ConstFalseNode()])
        falling_short.fail_condition = trigger.observes_true
        cancelled = Exception("cancelled")
        msc.add_node(CancelStatechart.when_failed(falling_short, exception=cancelled))

        executor.compile(statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            executor.tick_until_end()

        assert error.value is cancelled

    def test_cancel_statechart_when_true_without_exception_names_the_node(
        self,
    ) -> None:
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_node(watched := ConstTrueNode())
        msc.add_node(CancelStatechart.when_true(watched))

        executor.compile(statechart=msc)
        with pytest.raises(CancelledByNodeReachingItsGoal) as error:
            executor.tick_until_end()

        assert error.value.node is watched

    def test_cancel_statechart_when_failed_without_exception_names_the_node(
        self,
    ) -> None:
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_nodes([trigger := ConstTrueNode(), falling_short := ConstFalseNode()])
        falling_short.fail_condition = trigger.observes_true
        msc.add_node(CancelStatechart.when_failed(falling_short))

        executor.compile(statechart=msc)
        with pytest.raises(CancelledByNodeFailing) as error:
            executor.tick_until_end()

        assert error.value.node is falling_short

    @pytest.mark.parametrize(
        "factory",
        [CancelStatechart.when_true, EndStatechart.when_true],
    )
    def test_when_true_reads_the_outcome_as_well_as_the_observation(
        self, factory
    ) -> None:
        """
        The observation behind an outcome is gone once the node ends, so a terminal node
        built from the observation alone would stop arming exactly when the outcome it
        waits for arrives.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(watched := ConstTrueNode())

        terminal_node = factory(watched)

        assert set(terminal_node._start_condition.expression.free_variables()) == {
            watched.observes_true,
            watched.is_succeeded,
        }

    def test_goals_cannot_have_end_statechart(self) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        with pytest.raises(EndInCompositeNodeError):
            msc.add_node(Sequence(nodes=[ConstTrueNode(), EndStatechart()]))


class TestTemplates:

    def test_sequence_goal(self, tmp_path) -> None:
        """
        Every step but the first starts on the tick its predecessor succeeds.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        steps = [ConstTrueNode(name=f"step {index}") for index in range(4)]
        node = Sequence(nodes=list(steps))
        msc.add_node(node)
        msc.add_node(EndStatechart.when_true(node))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert all(step.life_cycle_state == LifeCycleValues.SUCCEEDED for step in steps)
        starts = [
            msc.state_history.get_life_cycle_history_of_node(step).index(
                LifeCycleValues.RUNNING
            )
            for step in steps
        ]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(steps)

    def test_a_sequence_without_steps_is_rejected(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(Sequence(nodes=[]))

        with pytest.raises(CompositeNodeWithoutChildrenError):
            kin_sim.compile(statechart=msc)

    def test_a_parallel_without_nodes_is_rejected(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(Parallel(nodes=[]))

        with pytest.raises(CompositeNodeWithoutChildrenError):
            kin_sim.compile(statechart=msc)

    def test_sequence_gives_a_terminal_step_no_ending_condition(self) -> None:
        """
        A sequence ends each step by its own observation, but a step that ends the whole
        statechart has nothing left to transition to.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        cancel = CancelStatechart(exception=Exception("cancelled"))
        msc.add_node(sequence := Sequence(nodes=[CountTicks(ticks=3), cancel]))
        msc.add_node(EndStatechart.when_true(sequence))

        kin_sim.compile(statechart=msc)

        assert [
            cancel.get_condition(transition_kind).free_variables()
            for transition_kind in TransitionKind.ending_kinds()
        ] == [[] for _ in TransitionKind.ending_kinds()]

    def test_a_sequence_ends_a_bare_task_once_it_observes_true(self) -> None:
        """
        A task observes whether its constraints are satisfied, which is enough to decide
        that it reached its goal, so a sequence can supply the ending itself rather than
        making every caller write one.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        task = ConstTrueNode(name="step")
        msc.add_node(sequence := Sequence(nodes=[task]))

        _compile_msc(msc).tick()

        assert task.parent_node is sequence
        assert task.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_a_sequence_fails_a_bare_task_that_fails_while_observing_true(self) -> None:
        """
        A task declaring its own failure on the tick it observes True did not reach its
        goal, so the ending the sequence supplies stands back.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        task = ConstTrueNode(name="step")
        task.fail_condition = task.observes_true
        msc.add_node(Sequence(nodes=[task]))

        _compile_msc(msc).tick()

        assert task.life_cycle_state == LifeCycleValues.FAILED

    def test_a_sequence_runs_a_step_deciding_its_own_success_as_it_is(self) -> None:
        """
        A step that succeeds on its own needs nothing to end it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        step = NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
        msc.add_node(sequence := Sequence(nodes=[step]))

        _compile_msc(msc).tick()

        assert step.parent_node is sequence
        assert step.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_a_sequence_rejects_a_step_whose_life_cycle_the_caller_wired(self) -> None:
        """
        What starts and ends a step is the sequence's to decide, so a step that arrives
        already wired is a disagreement rather than an addition.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        give_up_signal = CountTicks(ticks=2)
        step = ConstFalseNode()
        step.interrupt_condition = give_up_signal.is_succeeded

        with pytest.raises(ChildTransitionAlreadyWiredError):
            msc.add_node(Sequence(nodes=[give_up_signal, step, ConstTrueNode()]))

    def test_a_sequence_fails_once_a_step_gives_up(self) -> None:
        """
        A step that declared it cannot continue decides the sequence, instead of leaving
        whoever waits for it waiting forever.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            sequence := Sequence(
                nodes=[
                    NodeDeclaringItsOwnFailure(name="step"),
                    ConstTrueNode(name="unreached step"),
                ]
            )
        )

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert sequence.nodes[0].life_cycle_state == LifeCycleValues.FAILED
        assert sequence.last_observation_state == ObservationStateValues.FALSE

    def test_parallel(self) -> None:
        kin_sim = StatechartExecutor(
            StatechartContext(
                world=World(),
            )
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                parallel := Parallel(
                    nodes=[
                        CountTicks(ticks=3),
                        CountTicks(ticks=5),
                    ]
                ),
            ]
        )
        msc.add_node(EndStatechart.when_true(parallel))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        # 5 (longest ticker, parallel turns True on the same tick) + 1 (for end to trigger)
        assert kin_sim.tick_count == 6

    def test_parallel_minimum_success(self) -> None:
        """
        Test that Parallel completes when minimum_success nodes are True.
        """
        kin_sim = StatechartExecutor(
            StatechartContext(world=World()),
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                parallel := Parallel(
                    nodes=[
                        CountTicks(ticks=2),
                        CountTicks(ticks=4),
                        CountTicks(ticks=6),
                    ],
                    minimum_success=2,
                ),
            ]
        )
        msc.add_node(EndStatechart.when_true(parallel))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        # 4 (second ticker completes, parallel turns True on the same tick) + 1 (for end to trigger)
        assert kin_sim.tick_count == 5

    def test_parallel_minimum_success_zero(self) -> None:
        """
        Test that Parallel completes when no node is True.
        """
        kin_sim = StatechartExecutor(
            StatechartContext(world=World()),
        )
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                parallel := Parallel(
                    nodes=[
                        CountTicks(ticks=3),
                        CountTicks(ticks=5),
                        CountTicks(ticks=7),
                    ],
                    minimum_success=0,
                ),
            ]
        )
        msc.add_node(EndStatechart.when_true(parallel))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        # 0 (no ticker completes) + 1 (for parallel to turn True) + 1 (for end to trigger)
        assert kin_sim.tick_count == 2


class TestLifeCycleTransitions:
    """
    Tests the LifeCycle transitions of nodes in various edge cases and intended
    behavior.
    """

    def test_run_after_stop(self) -> None:
        """
        Test for node to run after the parent node already stopped.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        msc.add_node(
            sequence := Sequence(
                nodes=[
                    ConstTrueNode(),
                    CompositeNodeCancellingIfChildRunsAfterItsEnd(),
                    CountTicks(name="delay EndStatechart", ticks=5),
                ]
            )
        )
        msc.add_node(EndStatechart.when_true(sequence))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        # The goal takes its counters down with it, which interrupts them.
        assert (
            sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_run_after_stop_from_pause(self) -> None:
        """
        Test for node to run from paused while the parent node already stopped.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        msc.add_node(
            sequence := Sequence(
                nodes=[
                    ConstTrueNode(),
                    CompositeNodeCancellingIfPausedChildResumesAfterItsEnd(),
                    CountTicks(name="delay EndStatechart", ticks=5),
                ]
            )
        )
        msc.add_node(EndStatechart.when_true(sequence))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        # The goal takes its counters down with it, which interrupts them.
        assert (
            sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking3.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert sequence.nodes[1].pulse.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_end_before_start(self) -> None:
        """
        Test for node to start even if its success condition is met before its start
        condition.

        Node3 should start and run for 1 tick before ending, instead of never starting.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        node1 = CountTicks(ticks=1)
        node2 = ConstTrueNode()
        node3 = ConstTrueNode()

        msc.add_nodes(nodes=[node1, node2, node3])
        msc.add_node(EndStatechart.when_true(node3))

        node3.start_condition = node1.observes_true
        node3.success_condition = node2.observes_true

        kin_sim.compile(statechart=msc)
        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.observation_state == ObservationStateValues.TRUE

    def test_end_before_start_in_template(self) -> None:
        """
        Test for node to start even if its success condition is met before its start
        condition, when the nodes are inside a template.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        node = CompositeNodeWithChildSucceedingBeforeItStarts()
        msc.add_node(node)

        kin_sim.compile(statechart=msc)
        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node.node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node.node3.observation_state == ObservationStateValues.TRUE

    def test_intended_transitions(self) -> None:
        """
        Test for intended LifeCycle transitions of nodes.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        count_node1 = CountTicks(ticks=1, name="node1")
        count_node2 = CountTicks(ticks=2, name="node2")
        end_count_node1 = CountTicks(ticks=11, name="end_node1")
        pulse_node1 = Pulse(name="pulse1")
        pulse_node2 = Pulse(name="pulse2")

        msc.add_nodes(
            nodes=[
                count_node1,
                count_node2,
                end_count_node1,
                pulse_node1,
                pulse_node2,
            ]
        )
        msc.add_node(end_node := EndStatechart.when_true(end_count_node1))

        pulse_node1.start_condition = count_node1.observes_true
        pulse_node2.start_condition = count_node2.observes_true
        count_node2.start_condition = pulse_node1.observes_true

        count_node1.pause_condition = pulse_node1.observes_true

        count_node1.success_condition = count_node2.observes_true
        pulse_node1.interrupt_condition = count_node2.observes_true

        count_node1.reset_condition = pulse_node2.observes_true
        count_node2.reset_condition = pulse_node2.observes_true
        pulse_node1.reset_condition = pulse_node2.observes_true

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        assert len(msc.state_history) == 14
        # %% count_node1 history
        assert msc.state_history.get_life_cycle_history_of_node(count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
        ]
        assert msc.state_history.get_observation_history_of_node(count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
        ]

        # %% count_node2 history
        assert msc.state_history.get_life_cycle_history_of_node(count_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(count_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_count_node1 history
        assert msc.state_history.get_life_cycle_history_of_node(end_count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(end_count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_node history
        assert msc.state_history.get_life_cycle_history_of_node(end_node) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(end_node) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]

        # %% pulse_node1 history
        # The pulse is interrupted whenever count_node2 fires.
        pulse_node1_observations = msc.state_history.get_observation_history_of_node(
            pulse_node1
        )
        assert pulse_node1_observations == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
        ]
        assert msc.state_history.get_life_cycle_history_of_node(pulse_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.INTERRUPTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.INTERRUPTED,
            LifeCycleValues.INTERRUPTED,
            LifeCycleValues.INTERRUPTED,
        ]

        # %% pulse_node2 history
        assert msc.state_history.get_life_cycle_history_of_node(pulse_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.state_history.get_observation_history_of_node(pulse_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]

    def test_unpause_from_parent_pause(self) -> None:
        """
        Test for child node to unpause when parent node unpauses.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)

        pulse = Pulse()
        unpause = CompositeNodeResumingItsPausedChildren()

        msc.add_nodes(nodes=[pulse, unpause])
        msc.add_node(EndStatechart.when_true(unpause))

        unpause.pause_condition = pulse.observes_true

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

        # The goal takes its counter down with it, which interrupts it.
        assert unpause.count_ticks1.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert unpause.cancel.life_cycle_state == LifeCycleValues.NOT_STARTED

        assert unpause.last_observation_state == ObservationStateValues.TRUE

    def test_long_pause(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                node1 := Parallel(nodes=[ConstTrueNode(), ConstFalseNode()]),
                pulse := Pulse(length=5),
            ]
        )
        node1.pause_condition = pulse.observes_true
        msc.add_node(EndStatechart.when_false(pulse))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()
        msc.plot_gantt_chart()

        assert len(msc.state_history) == 5

    def test_a_child_starts_while_its_parent_success_condition_has_no_answer(
        self,
    ) -> None:
        """
        Only a success condition that is true ends a node, so a parent whose success
        condition is still undecided is not ending and does not hold its child back.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_nodes(
            [
                undecided := NodeObservingNothingYet(),
                goal := CompositeNodeWithChildStartingLate(delay_in_ticks=2),
            ]
        )
        goal.success_condition = undecided.observes_true

        kin_sim.compile(statechart=msc)
        for _ in range(3):
            kin_sim.tick()

        assert undecided.observation_state == ObservationStateValues.UNKNOWN
        assert goal.life_cycle_state == LifeCycleValues.RUNNING
        assert goal.child.life_cycle_state == LifeCycleValues.RUNNING

    @pytest.mark.parametrize("transition_kind", TransitionKind.ending_kinds())
    def test_a_child_does_not_start_under_a_parent_ending_on_the_same_tick(
        self, transition_kind: TransitionKind
    ) -> None:
        """
        A child whose start condition turns true on the tick its parent ends would only
        be cut off again, so it never starts, however the parent ends.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := CountTicks(ticks=2),
                goal := CompositeNodeWithChildStartingLate(delay_in_ticks=2),
            ]
        )
        goal.set_condition(transition_kind, trigger.observes_true)

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert goal.life_cycle_state is transition_kind.outcome
        assert goal.child.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_reset_outranks_a_start(self) -> None:
        """
        A reset outranks every other transition, so a node whose reset is held true does
        not start while it is.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.start_condition = trigger.observes_true
        node.reset_condition = trigger.observes_true

        executor = _compile_msc(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_node_starts_once_its_reset_drops(self) -> None:
        """
        A reset holds a node back only for as long as it is true, so a start condition
        that outlives it still starts the node.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := Pulse(),
                node := ConstTrueNode(),
            ]
        )
        node.start_condition = trigger.observes_true
        node.reset_condition = reset.observes_true

        executor = _compile_msc(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_resetting_ancestor_holds_back_a_child_that_would_start(self) -> None:
        """
        An ancestor resets everything beneath it, so a child whose start condition turns
        true on the tick its ancestor is reset stays where it is.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                reset := CountTicks(ticks=2),
                goal := CompositeNodeWithChildStartingLate(delay_in_ticks=2),
            ]
        )
        goal.reset_condition = reset.observes_true

        executor = _compile_msc(msc)
        executor.tick()
        executor.tick()

        assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert goal.child.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% transitions a parent forces on a node


def _evaluate_in(
    expression: Scalar, life_cycle_states: dict[StatechartNode, LifeCycleValues]
) -> float:
    """
    :param expression: An expression reading the life cycle states of nodes.
    :param life_cycle_states: The life cycle state of every node `expression` reads.
    :return: The value of `expression` in those states.
    """
    return float(
        expression.substitute(
            [node.life_cycle_variable for node in life_cycle_states],
            [Scalar(state) for state in life_cycle_states.values()],
        )
    )


class TestLifeCycleForcedByParent:
    """
    The life cycle state a node is forced into by its parent alone is the one its
    transitions reach once none triggered by its own conditions may happen.
    """

    @pytest.mark.parametrize("parent_life_cycle_state", list(LifeCycleValues))
    @pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
    def test_a_child_reaches_its_transitions_without_its_own_ones(
        self,
        life_cycle_state: LifeCycleValues,
        parent_life_cycle_state: LifeCycleValues,
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(parent := Sequence(nodes=[child := ConstTrueNode()]))
        _compile_msc(msc)
        states = {child: life_cycle_state, parent: parent_life_cycle_state}
        transitions = dict(
            child.create_lifecycle_transitions(Scalar.const_false()).as_cases()
        )

        forced = _evaluate_in(child.create_life_cycle_forced_by_parent(), states)

        assert forced == _evaluate_in(transitions[life_cycle_state], states)

    @pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
    def test_a_top_level_node_reaches_its_transitions_without_its_own_ones(
        self, life_cycle_state: LifeCycleValues
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())
        _compile_msc(msc)
        states = {node: life_cycle_state}
        transitions = dict(
            node.create_lifecycle_transitions(Scalar.const_false()).as_cases()
        )

        forced = _evaluate_in(node.create_life_cycle_forced_by_parent(), states)

        assert forced == _evaluate_in(transitions[life_cycle_state], states)


# %% life cycle outcomes


class TestLifeCycleOutcomes:
    """
    Tests which terminal state a node ends in, depending on the condition that ended it.
    """

    @staticmethod
    def _compile(msc: Statechart) -> StatechartExecutor:
        """
        :param msc: The statechart to compile.
        :return: An executor ready to tick `msc`.
        """
        executor = StatechartExecutor(msc.context)
        executor.compile(statechart=msc)
        return executor

    def test_a_success_condition_succeeds_a_node_short_of_its_goal(self) -> None:
        """
        Succeeding is declared by the condition that ends a node, so what the node
        observes at that moment has no say in it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.success_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.FALSE
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_an_interrupt_condition_interrupts_a_node_at_its_goal(self) -> None:
        """
        Being interrupted is declared too, so a node sitting at its goal is not judged a
        success when whatever ended it only meant to stop it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.interrupt_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.TRUE
        assert node.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_success_outranks_failure_on_the_same_tick(self) -> None:
        """
        A node that arrived did what it was asked, whatever else was declared on that
        tick.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.success_condition = trigger.observes_true
        node.fail_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_failure_outranks_interruption_on_the_same_tick(self) -> None:
        """
        A node declaring that it cannot continue says more about it than being stopped
        does.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.fail_condition = trigger.observes_true
        node.interrupt_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_a_nodes_own_success_outranks_an_ending_ancestor(self) -> None:
        """
        A child that declares its success on the tick its parent ends keeps that outcome
        rather than being cut off.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                goal := CompositeNodeWithChildSucceedingOnItsOwn(),
            ]
        )
        goal.interrupt_condition = trigger.observes_true

        self._compile(msc).tick()

        assert goal.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert goal.child.life_cycle_state == LifeCycleValues.SUCCEEDED

    @pytest.mark.parametrize("transition_kind", TransitionKind.ending_kinds())
    def test_an_ending_ancestor_interrupts_a_child_at_its_goal(
        self, transition_kind: TransitionKind
    ) -> None:
        """
        However a parent ends, its children are only cut off by it, so a child sitting
        at its goal is interrupted rather than judged.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                goal := CompositeNodeCuttingOffItsChildAtItsGoal(),
            ]
        )
        goal.set_condition(transition_kind, trigger.observes_true)

        self._compile(msc).tick()

        assert goal.life_cycle_state is transition_kind.outcome
        assert goal.child.observation_state == ObservationStateValues.TRUE
        assert goal.child.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_a_success_condition_succeeds_a_node_that_has_observed_nothing(
        self,
    ) -> None:
        """
        An observation with no answer is no obstacle to succeeding either, since the
        condition alone decides.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := NodeObservingNothingYet()])
        node.success_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.UNKNOWN
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_an_interrupted_node_observing_false_is_not_judged_to_have_failed(
        self,
    ) -> None:
        """
        A node that observes whether something is the case has not failed to observe
        anything just because the answer was no when it was stopped.

        Only a node whose fail condition held has failed.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.interrupt_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.FALSE
        assert node.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_a_fail_condition_fails_a_node_at_its_goal(self) -> None:
        """
        Failing is declared rather than read off an observation, so a node whose fail
        condition holds fails even while it observes its goal as reached.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.fail_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.TRUE
        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_a_node_failing_on_observing_false_fails_once_it_observes_false(
        self,
    ) -> None:
        """
        A node declaring that observing False means it can no longer reach its goal
        fails without any condition declaring it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeFailingOnObservingFalse(
                observation=ObservationStateValues.FALSE
            )
        )

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_a_node_deciding_its_own_success_succeeds_once_it_observes_true(
        self,
    ) -> None:
        """
        A node whose ending undoes nothing it did succeeds without any condition
        declaring it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeSucceedingOnObservingTrue(
                observation=ObservationStateValues.TRUE
            )
        )

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_a_node_deciding_its_own_success_is_not_failed_by_observing_false(
        self,
    ) -> None:
        """
        Succeeding by itself says nothing about failing, which a node declares
        separately.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeSucceedingOnObservingTrue(
                observation=ObservationStateValues.FALSE
            )
        )

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_node_failing_on_observing_false_keeps_running_while_it_observes_unknown(
        self,
    ) -> None:
        """
        An observation with no answer yet says nothing about whether the goal can still
        be reached.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeFailingOnObservingFalse(
                observation=ObservationStateValues.UNKNOWN
            )
        )

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_node_failing_on_observing_false_is_not_succeeded_by_observing_true(
        self,
    ) -> None:
        """
        Failing itself says nothing about succeeding, which stays its owner's to decide.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeFailingOnObservingFalse(observation=ObservationStateValues.TRUE)
        )

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_node_failing_on_observing_false_keeps_the_fail_condition_it_was_given(
        self,
    ) -> None:
        """
        The failure the statechart supplies comes on top of the one declared for the
        node, rather than replacing it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                node := NodeFailingOnObservingFalse(
                    observation=ObservationStateValues.TRUE
                ),
            ]
        )
        node.fail_condition = trigger.observes_true

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_a_child_an_ancestor_took_down_at_its_goal_has_not_succeeded(self) -> None:
        """
        Only a declared success latches what a node reached, so a child interrupted at
        its goal is judged afterwards like any other interrupted node.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                goal := CompositeNodeCuttingOffItsChildAtItsGoal(),
            ]
        )
        goal.success_condition = trigger.observes_true

        self._compile(msc).tick()

        assert goal.child.is_succeeded.resolve() == (
            LifeCyclePredicate.IS_SUCCEEDED.truth_value(LifeCycleValues.INTERRUPTED)
        )

    def test_an_ending_ancestor_interrupts_a_grandchild_too(self) -> None:
        """
        Every node below an ending one is taken down with it, however deep, and each of
        them is interrupted.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                goal := CompositeNodeCuttingOffItsGrandchild(),
            ]
        )
        goal.success_condition = trigger.observes_true

        self._compile(msc).tick()

        assert goal.grandchild.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_a_child_ends_the_same_way_whether_a_sibling_or_its_parent_ends_it(
        self,
    ) -> None:
        """
        A sibling interrupting a node and a parent taking it down with it both only stop
        the node, so both leave it interrupted.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                ended_by_a_sibling := CompositeNodeWithChildInterruptedBySibling(),
                ended_by_its_parent := CompositeNodeCuttingOffItsChild(),
            ]
        )
        ended_by_its_parent.success_condition = trigger.observes_true

        self._compile(msc).tick()

        assert (
            ended_by_a_sibling.child.life_cycle_state
            == TransitionKind.INTERRUPT.outcome
        )
        assert (
            ended_by_its_parent.child.life_cycle_state
            == TransitionKind.INTERRUPT.outcome
        )

    def test_a_child_that_already_ended_keeps_its_outcome(self) -> None:
        """
        An outcome is only left by a reset, so a parent ending later does not overwrite
        one its child already earned.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := CountTicks(ticks=2),
                goal := CompositeNodeWithChildFailingOnItsOwn(),
            ]
        )
        goal.success_condition = trigger.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert goal.child.life_cycle_state == LifeCycleValues.FAILED

        executor.tick()
        assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert goal.child.life_cycle_state == LifeCycleValues.FAILED

    def test_reset_leaves_succeeded(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountTicks(ticks=2),
                node := ConstTrueNode(),
            ]
        )
        node.success_condition = trigger.observes_true
        node.reset_condition = reset.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_reset_leaves_failed(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountTicks(ticks=2),
                node := ConstFalseNode(),
            ]
        )
        node.fail_condition = trigger.observes_true
        node.reset_condition = reset.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_reset_leaves_interrupted(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountTicks(ticks=2),
                goal := CompositeNodeCuttingOffItsUndecidedChild(),
            ]
        )
        goal.success_condition = trigger.observes_true
        goal.reset_condition = reset.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert goal.child.life_cycle_state == LifeCycleValues.INTERRUPTED

        executor.tick()
        assert goal.child.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_an_ended_node_observes_nothing_and_keeps_its_outcome(self) -> None:
        """
        A node that is no longer running is no longer observing, so its observation says
        so and only its outcome still answers for it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.fail_condition = trigger.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert node.observation_state == ObservationStateValues.FALSE

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED
        assert node.observation_state == ObservationStateValues.UNKNOWN

    def test_a_paused_node_keeps_the_observation_it_made(self) -> None:
        """
        A paused node resumes and observes again, so the reading it was interrupted on
        is kept rather than discarded.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([pause_trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.pause_condition = pause_trigger.observes_true

        executor = self._compile(msc)
        executor.tick()
        assert node.observation_state == ObservationStateValues.TRUE

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.PAUSED
        assert node.observation_state == ObservationStateValues.TRUE

    def test_only_the_outcome_of_an_ended_node_still_starts_a_later_node(self) -> None:
        """
        A condition that outlives the node it reads has to read the outcome, since the
        observation behind it is gone by the time the condition is asked again.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                finished := ConstTrueNode(),
                later := CountTicks(ticks=3),
                on_outcome := ConstTrueNode(),
                on_observation := ConstTrueNode(),
            ]
        )
        finished.success_condition = trigger.observes_true
        on_outcome.start_condition = sm.logic_and(
            finished.is_succeeded, later.observes_true
        )
        on_observation.start_condition = sm.logic_and(
            finished.observes_true, later.observes_true
        )

        executor = self._compile(msc)
        for _ in range(5):
            executor.tick()

        assert finished.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert later.observation_state == ObservationStateValues.TRUE
        assert on_outcome.life_cycle_state == LifeCycleValues.RUNNING
        assert on_observation.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% what a composite goal reads from its children


class TestReadingChildrenThatEnded:
    """
    Tests how a composite goal reads children that may already have ended, whose
    observation is gone by then.
    """

    def test_a_sequence_survives_losing_its_last_step_observation(self) -> None:
        """
        A finished step's observation is only kept because a terminal state freezes it.

        Reading the outcome instead makes the sequence independent of that.
        """
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_node(sequence := Sequence(nodes=[ConstTrueNode(), ConstTrueNode()]))
        executor.compile(statechart=msc)
        for _ in range(6):
            executor.tick()
        assert sequence.last_observation_state == ObservationStateValues.TRUE

        last_step = sequence.nodes[-1]
        assert last_step.life_cycle_state == LifeCycleValues.SUCCEEDED
        msc.observation_state[last_step] = ObservationStateValues.UNKNOWN
        executor.tick()

        assert sequence.last_observation_state == ObservationStateValues.TRUE

    def test_a_sequence_fails_once_a_step_ended_short_of_its_goal(self) -> None:
        """
        A step that was given up on stalls the chain, so the sequence reports the
        failure instead of waiting for a step that will never succeed.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        step = NodeDeclaringItsOwnFailure(name="given up on")
        last_step = ConstTrueNode()
        msc.add_node(sequence := Sequence(nodes=[step, last_step]))

        executor = _compile_msc(msc)
        for _ in range(5):
            executor.tick()

        assert sequence.last_observation_state == ObservationStateValues.FALSE
        assert last_step.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_sequence_fails_once_its_last_step_ended_short_of_its_goal(self) -> None:
        """
        The last step is what the sequence reads its success off, so failing it has to
        be answered from the same expression rather than left unknown.

        The step observes nothing decisive, so the failure can only come from it having
        ended, not from what it observed on the way.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        last_step = NodeObservingNothingYet(name="given up on")
        last_step.fail_condition = sm.Scalar.const_true()
        msc.add_node(sequence := Sequence(nodes=[last_step]))

        executor = _compile_msc(msc)
        for _ in range(5):
            executor.tick()

        assert sequence.last_observation_state == ObservationStateValues.FALSE

    def test_a_sequence_stays_unknown_while_a_step_is_short_of_its_goal(self) -> None:
        """
        Being short of its goal is what a step observes on its way there, not a failure,
        so only a step that ended without reaching it decides anything.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(sequence := Sequence(nodes=[ConstFalseNode(), ConstTrueNode()]))

        executor = _compile_msc(msc)
        for _ in range(5):
            executor.tick()

        assert set(msc.state_history.get_observation_history_of_node(sequence)) == {
            ObservationStateValues.UNKNOWN
        }

    def test_a_parallel_counts_an_ended_child_and_a_running_one(self) -> None:
        """
        A parallel ends none of its children, so a child that keeps running is judged by
        what it observes now and a child something else ended by its outcome.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            parallel := Parallel(
                nodes=[ended := Pulse(), still_running := ConstTrueNode()]
            )
        )
        ended.success_condition = ended.observes_true

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert ended.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert still_running.life_cycle_state == LifeCycleValues.RUNNING
        assert parallel.observation_state == ObservationStateValues.TRUE

    def test_a_parallel_stops_counting_a_child_that_ended_without_succeeding(
        self,
    ) -> None:
        """
        A child that ended short of its goal cannot be at it any more, however true the
        observation it kept still reads.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            parallel := Parallel(
                nodes=[failed := Pulse(), still_trying := ConstFalseNode()],
                minimum_success=1,
            )
        )
        failed.fail_condition = failed.observes_true

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert failed.life_cycle_state == LifeCycleValues.FAILED
        assert failed.last_observation_state == ObservationStateValues.TRUE
        assert still_trying.life_cycle_state == LifeCycleValues.RUNNING
        assert parallel.observation_state == ObservationStateValues.FALSE

    def test_a_parallel_fails_once_too_few_children_can_reach_their_goals(self) -> None:
        """
        Nothing brings a child that ended back, so a parallel that can no longer reach
        its goal says so rather than holding whoever runs it open forever.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(parallel := Parallel(nodes=[failed := Pulse(), ConstTrueNode()]))
        failed.fail_condition = failed.observes_true

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert parallel.life_cycle_state == LifeCycleValues.FAILED

    def test_a_parallel_keeps_going_while_enough_children_can_still_reach_their_goals(
        self,
    ) -> None:
        """
        A child that ended without succeeding only decides the parallel once the ones
        left cannot make up the number it asks for.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            parallel := Parallel(
                nodes=[failed := Pulse(), at_its_goal := ConstTrueNode()],
                minimum_success=1,
            )
        )
        failed.fail_condition = failed.observes_true

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert failed.life_cycle_state == LifeCycleValues.FAILED
        assert at_its_goal.life_cycle_state == LifeCycleValues.RUNNING
        assert parallel.life_cycle_state == LifeCycleValues.RUNNING
        assert parallel.observation_state == ObservationStateValues.TRUE

    def test_a_parallel_is_not_satisfied_by_children_true_at_different_times(
        self,
    ) -> None:
        """
        A parallel asks whether its children reached their goals at the same time, so a
        child that reached its goal and drifted away again stops counting towards it.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            parallel := Parallel(
                nodes=[
                    drifting := Pulse(),
                    late := CountTicks(ticks=3),
                ]
            )
        )

        executor = _compile_msc(msc)
        for _ in range(6):
            executor.tick()

        assert (
            ObservationStateValues.TRUE
            in msc.state_history.get_observation_history_of_node(drifting)
        )
        assert (
            ObservationStateValues.TRUE
            in msc.state_history.get_observation_history_of_node(late)
        )
        assert set(msc.state_history.get_observation_history_of_node(parallel)) == {
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
        }


# %% last observation


class TestLastObservation:
    """
    Tests the observation a node took most recently, which outlasts the node ending and
    says nothing about how it ended.
    """

    def test_a_node_that_has_not_started_reads_unknown(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([blocker := ConstFalseNode(), node := ConstTrueNode()])
        node.start_condition = blocker.observes_true

        executor = _compile_msc(msc)
        executor.tick()

        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node.last_observation_state == ObservationStateValues.UNKNOWN

    @pytest.mark.parametrize(
        "node_type, expected",
        [
            (ConstTrueNode, ObservationStateValues.TRUE),
            (ConstFalseNode, ObservationStateValues.FALSE),
        ],
    )
    def test_a_running_node_reads_what_it_observes(self, node_type, expected) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := node_type())

        executor = _compile_msc(msc)
        executor.tick()

        assert node.life_cycle_state == LifeCycleValues.RUNNING
        assert node.last_observation_state == expected

    def test_every_ended_node_reads_what_it_observed_whatever_its_outcome(self) -> None:
        """
        One node per life cycle state and observation, so a node reading another node's
        row, or its outcome instead of its observation, would show up here.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                blocker := ConstFalseNode(),
                not_started := ConstTrueNode(),
                running := ConstTrueNode(),
                succeeded := ConstTrueNode(),
                failed_observing_false := ConstFalseNode(),
                failed_observing_true := ConstTrueNode(),
                interrupted_observing_true := ConstTrueNode(),
                interrupted_undecided := NodeObservingNothingYet(),
            ]
        )
        not_started.start_condition = blocker.observes_true
        succeeded.success_condition = trigger.observes_true
        failed_observing_false.fail_condition = trigger.observes_true
        failed_observing_true.fail_condition = trigger.observes_true
        interrupted_observing_true.interrupt_condition = trigger.observes_true
        interrupted_undecided.interrupt_condition = trigger.observes_true
        nodes = (
            not_started,
            running,
            succeeded,
            failed_observing_false,
            failed_observing_true,
            interrupted_observing_true,
            interrupted_undecided,
        )

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert {node: node.life_cycle_state for node in nodes} == {
            not_started: LifeCycleValues.NOT_STARTED,
            running: LifeCycleValues.RUNNING,
            succeeded: LifeCycleValues.SUCCEEDED,
            failed_observing_false: LifeCycleValues.FAILED,
            failed_observing_true: LifeCycleValues.FAILED,
            interrupted_observing_true: LifeCycleValues.INTERRUPTED,
            interrupted_undecided: LifeCycleValues.INTERRUPTED,
        }
        assert {node: node.last_observation_state for node in nodes} == {
            not_started: ObservationStateValues.UNKNOWN,
            running: ObservationStateValues.TRUE,
            succeeded: ObservationStateValues.TRUE,
            failed_observing_false: ObservationStateValues.FALSE,
            failed_observing_true: ObservationStateValues.TRUE,
            interrupted_observing_true: ObservationStateValues.TRUE,
            interrupted_undecided: ObservationStateValues.UNKNOWN,
        }

    def test_an_ended_node_keeps_the_observation_its_ending_transition_read(
        self,
    ) -> None:
        """
        The pulse observed True before it observed the False that ended it, so only the
        most recent reading tells the two apart.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(pulse := Pulse(length=1))
        pulse.interrupt_condition = pulse.observes_false

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert (
            ObservationStateValues.TRUE
            in msc.state_history.get_observation_history_of_node(pulse)
        )
        assert pulse.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert pulse.observation_state == ObservationStateValues.UNKNOWN
        assert pulse.last_observation_state == ObservationStateValues.FALSE

    def test_a_reset_forgets_the_previous_run(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([starter := Pulse(length=1), node := ConstTrueNode()])
        node.start_condition = starter.observes_true
        node.success_condition = node.observes_true
        node.reset_condition = node.is_succeeded

        executor = _compile_msc(msc)
        for _ in range(2):
            executor.tick()
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node.last_observation_state == ObservationStateValues.TRUE

        for _ in range(2):
            executor.tick()

        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node.last_observation_state == ObservationStateValues.UNKNOWN

    def test_a_condition_reads_it_after_the_node_ended(self) -> None:
        """
        The watched node was cut off at its goal, so neither its live observation nor
        its outcome could start the watcher once the delay is over.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                delay := CountTicks(ticks=3),
                ended := ConstTrueNode(),
                watcher := ConstFalseNode(),
            ]
        )
        ended.interrupt_condition = trigger.observes_true
        watcher.start_condition = logic_and(
            delay.observes_true, ended.last_observed_true
        )

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert ended.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert watcher.life_cycle_state == LifeCycleValues.RUNNING

    def test_an_observation_reads_it_after_the_node_ended(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                ended := ConstTrueNode(),
                observer := NodeObservingLastObservation(watched_node=ended),
            ]
        )
        ended.interrupt_condition = trigger.observes_true

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert ended.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert observer.observation_state == ObservationStateValues.TRUE

    def test_it_renders_as_one_variable(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([finished := ConstTrueNode(), later := ConstTrueNode()])
        later.start_condition = finished.last_observed_true

        assert (
            str(later._start_condition)
            == f'"{finished.last_observed_true.display_name}"'
        )

    def test_an_observation_expression_reads_the_current_tick(self) -> None:
        """
        A tick settles before it ends, so an observation expression reads the last
        observation taken on the same tick.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                watched := CountTicks(ticks=2),
                observer := NodeObservingLastObservation(watched_node=watched),
            ]
        )

        executor = _compile_msc(msc)
        executor.tick()

        assert watched.last_observation_state == ObservationStateValues.FALSE
        assert observer.observation_state == ObservationStateValues.FALSE

        executor.tick()

        assert watched.last_observation_state == ObservationStateValues.TRUE
        assert observer.observation_state == ObservationStateValues.TRUE

    def test_a_condition_reads_the_current_tick(self) -> None:
        """
        The last observation is taken over before the life cycle update, so a transition
        condition acts on it on the tick it is observed.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                watched := CountTicks(ticks=2),
                waiting := ConstFalseNode(),
            ]
        )
        waiting.start_condition = watched.last_observed_true

        executor = _compile_msc(msc)
        for _ in range(2):
            executor.tick()

        assert watched.last_observation_state == ObservationStateValues.TRUE
        assert waiting.life_cycle_state == LifeCycleValues.RUNNING


# %% life cycle predicates


class TestLifeCyclePredicates:
    """
    Tests the truth tables of the life cycle predicates and their use in conditions.
    """

    @pytest.mark.parametrize(
        "predicate, outcome",
        [
            (LifeCyclePredicate.IS_SUCCEEDED, LifeCycleValues.SUCCEEDED),
            (LifeCyclePredicate.IS_FAILED, LifeCycleValues.FAILED),
            (LifeCyclePredicate.IS_INTERRUPTED, LifeCycleValues.INTERRUPTED),
        ],
    )
    def test_an_outcome_predicate_is_true_only_in_its_own_state(
        self, predicate, outcome
    ) -> None:
        """
        A node that has not ended, or ended some other way, did not end this way, so a
        outcome predicate answers in every state.
        """
        for life_cycle_state in LifeCycleValues:
            expected = (
                ObservationStateValues.TRUE
                if life_cycle_state is outcome
                else ObservationStateValues.FALSE
            )
            assert predicate.truth_value(life_cycle_state) == expected

    def test_a_negated_outcome_predicate_fires_for_an_interrupted_node(self) -> None:
        """
        A node that was interrupted did not succeed, so a node waiting for another to
        end without succeeding starts once that one is interrupted.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                watched := NodeObservingNothingYet(),
                waiting := ConstTrueNode(),
            ]
        )
        watched.interrupt_condition = trigger.observes_true
        waiting.start_condition = logic_and(
            watched.is_terminated, logic_not(watched.is_succeeded)
        )

        executor = _compile_msc(msc)
        executor.tick()

        assert watched.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert waiting.life_cycle_state == LifeCycleValues.RUNNING

    @pytest.mark.parametrize(
        "predicate, phase",
        [
            (LifeCyclePredicate.IS_NOT_STARTED, LifeCycleValues.NOT_STARTED),
            (LifeCyclePredicate.IS_RUNNING, LifeCycleValues.RUNNING),
            (LifeCyclePredicate.IS_PAUSED, LifeCycleValues.PAUSED),
        ],
    )
    def test_phase_predicate_is_binary_in_every_state(self, predicate, phase) -> None:
        """
        Where a node is right now always has an answer, so a phase predicate is never
        unknown.
        """
        for life_cycle_state in LifeCycleValues:
            expected = (
                ObservationStateValues.TRUE
                if life_cycle_state is phase
                else ObservationStateValues.FALSE
            )
            assert predicate.truth_value(life_cycle_state) == expected

    @pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
    def test_is_terminated_matches_the_terminal_states(self, life_cycle_state) -> None:
        expected = (
            ObservationStateValues.TRUE
            if life_cycle_state.is_terminal
            else ObservationStateValues.FALSE
        )
        assert (
            LifeCyclePredicate.IS_TERMINATED.truth_value(life_cycle_state) == expected
        )

    def test_a_condition_starts_a_node_on_the_tick_an_outcome_is_reached(self) -> None:
        """
        A predicate reads the life cycle its node reaches in the same step, so a node
        reacting to an outcome starts on the tick that outcome is reached.
        """
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                first := ConstTrueNode(),
                second := ConstFalseNode(),
            ]
        )
        first.success_condition = trigger.observes_true
        second.start_condition = first.is_succeeded

        executor.compile(statechart=msc)
        assert second.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert second.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_predicate_follows_the_life_cycle_state_of_its_node(self) -> None:
        executor = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=executor.context)
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.fail_condition = trigger.observes_true

        executor.compile(statechart=msc)
        assert node.is_failed.resolve() == ObservationStateValues.FALSE

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED
        assert node.is_failed.resolve() == ObservationStateValues.TRUE

    def test_a_node_reset_by_its_own_outcome_is_reset_on_the_next_tick(self) -> None:
        """
        Failing and resetting are both triggered by the node's own conditions, and a
        node takes at most one such transition per tick.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.fail_condition = trigger.observes_true
        node.reset_condition = node.is_failed

        executor = _compile_msc(msc)

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_predicate_variable_is_created_once_per_node(self) -> None:
        node = ConstTrueNode()
        assert node.is_failed is node.is_failed
        assert node.is_failed is not node.is_succeeded

    def test_a_condition_renders_a_predicate_by_name(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = first.is_failed

        assert (
            str(second._start_condition)
            == f'"{first.unique_name}.{LifeCyclePredicate.IS_FAILED.attribute_name}"'
        )

    def test_a_predicate_makes_its_node_a_dependency_of_the_condition(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = first.is_failed

        assert second._start_condition.node_dependencies == [first]

    def test_variables_are_the_variables_the_condition_reads(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = sm.logic_and(first.is_failed, first.observes_true)

        assert set(second._start_condition.variables) == {
            first.is_failed,
            first.observes_true,
        }

    def test_an_observation_variable_resolves_to_what_its_node_observes(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())
        msc.observation_state[node] = ObservationStateValues.FALSE

        assert node.observation_variable.resolve() == ObservationStateValues.FALSE

    def test_a_last_observation_variable_resolves_to_what_its_node_observed_last(
        self,
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())
        msc.last_observation_state[node] = ObservationStateValues.FALSE

        assert node.last_observation.resolve() == ObservationStateValues.FALSE

    def test_a_predicate_variable_resolves_to_the_value_of_its_predicate(self) -> None:
        """
        An outcome predicate follows the life cycle state alone, however decisive the
        observation of its node already is.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())
        msc.life_cycle_state[node] = LifeCycleValues.RUNNING
        msc.observation_state[node] = ObservationStateValues.TRUE

        assert (
            node.is_succeeded.resolve()
            == LifeCyclePredicate.IS_SUCCEEDED.truth_value(LifeCycleValues.RUNNING)
        )

    def test_a_raw_life_cycle_variable_is_rejected_in_a_condition(self) -> None:
        """
        The raw life cycle value cannot be rendered back into a condition string, so
        only the predicates may be read.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])

        with pytest.raises(UnsupportedConditionVariableError):
            second.start_condition = first.life_cycle_variable

    def test_a_start_condition_may_not_read_its_own_outcome(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(node := ConstTrueNode())

        with pytest.raises(SelfInStartConditionError):
            node.start_condition = node.is_failed


# %% ending short of the goal


class TestIsFailedOrInterrupted:
    """
    Tests the predicate that answers whether a node ended anywhere but at its goal,
    which a transition condition may read where the life cycle variable is out of
    bounds.
    """

    @staticmethod
    def _answer(life_cycle_state: LifeCycleValues) -> ObservationStateValues:
        """
        :param life_cycle_state: The state to evaluate the predicate in.
        :return: What the predicate answers.
        """
        node = ConstTrueNode()
        substituted = sm.Scalar(node.is_failed_or_interrupted).substitute(
            [node.is_failed, node.is_interrupted],
            [
                float(LifeCyclePredicate.IS_FAILED.truth_value(life_cycle_state)),
                float(LifeCyclePredicate.IS_INTERRUPTED.truth_value(life_cycle_state)),
            ],
        )
        return ObservationStateValues(float(substituted))

    @pytest.mark.parametrize(
        "life_cycle_state, expected",
        [
            (LifeCycleValues.SUCCEEDED, ObservationStateValues.FALSE),
            (LifeCycleValues.FAILED, ObservationStateValues.TRUE),
            (LifeCycleValues.INTERRUPTED, ObservationStateValues.TRUE),
        ],
    )
    def test_every_way_of_ending_is_answered(self, life_cycle_state, expected) -> None:
        """
        Being cut off undecided counts as ending short of the goal just as much as being
        judged to have failed.
        """
        assert self._answer(life_cycle_state) == expected

    @pytest.mark.parametrize(
        "life_cycle_state",
        sorted(set(LifeCycleValues) - LifeCycleValues.terminal_states()),
    )
    def test_a_node_that_has_not_ended_answers_false(self, life_cycle_state) -> None:
        """
        A node that is still on its way has not ended at all.
        """
        assert self._answer(life_cycle_state) == ObservationStateValues.FALSE


# %% observation predicates


class TestObservationPredicates:
    """
    Tests the predicates that answer, True or False, whether a node observes or last
    observed a particular value.
    """

    @pytest.mark.parametrize(
        "predicate, observation, expected",
        [
            (
                ObservationPredicate.OBSERVES_TRUE,
                ObservationStateValues.TRUE,
                ObservationStateValues.TRUE,
            ),
            (
                ObservationPredicate.OBSERVES_TRUE,
                ObservationStateValues.UNKNOWN,
                ObservationStateValues.FALSE,
            ),
            (
                ObservationPredicate.OBSERVES_TRUE,
                ObservationStateValues.FALSE,
                ObservationStateValues.FALSE,
            ),
            (
                ObservationPredicate.OBSERVES_FALSE,
                ObservationStateValues.TRUE,
                ObservationStateValues.FALSE,
            ),
            (
                ObservationPredicate.OBSERVES_FALSE,
                ObservationStateValues.UNKNOWN,
                ObservationStateValues.FALSE,
            ),
            (
                ObservationPredicate.OBSERVES_FALSE,
                ObservationStateValues.FALSE,
                ObservationStateValues.TRUE,
            ),
            (
                ObservationPredicate.LAST_OBSERVED_TRUE,
                ObservationStateValues.TRUE,
                ObservationStateValues.TRUE,
            ),
            (
                ObservationPredicate.LAST_OBSERVED_TRUE,
                ObservationStateValues.UNKNOWN,
                ObservationStateValues.FALSE,
            ),
            (
                ObservationPredicate.LAST_OBSERVED_TRUE,
                ObservationStateValues.FALSE,
                ObservationStateValues.FALSE,
            ),
        ],
    )
    def test_a_predicate_is_true_only_for_the_value_it_asks_about(
        self, predicate, observation, expected
    ) -> None:
        """
        Unknown is neither of the values a predicate asks about, so it answers False.
        """
        variable = FloatVariable(name="observation")

        answer = predicate.expression(variable).substitute(
            [variable], [float(observation)]
        )

        assert ObservationStateValues(float(answer)) == expected
        assert predicate.truth_value(observation) == expected

    @pytest.mark.parametrize(
        "read_predicate, predicate",
        [
            (lambda node: node.observes_true, ObservationPredicate.OBSERVES_TRUE),
            (lambda node: node.observes_false, ObservationPredicate.OBSERVES_FALSE),
            (
                lambda node: node.last_observed_true,
                ObservationPredicate.LAST_OBSERVED_TRUE,
            ),
        ],
        ids=["observes_true", "observes_false", "last_observed_true"],
    )
    def test_the_node_reads_this_predicate(self, read_predicate, predicate) -> None:
        """
        The attribute a caller reaches for is what the tested truth table belongs to,
        and reading it twice hands out the same variable.
        """
        node = ConstTrueNode()

        assert read_predicate(node).predicate is predicate
        assert read_predicate(node) is read_predicate(node)

    @pytest.mark.parametrize(
        "watched_type, expected",
        [
            (ConstFalseNode, LifeCycleValues.RUNNING),
            (NodeObservingNothingYet, LifeCycleValues.NOT_STARTED),
            (ConstTrueNode, LifeCycleValues.NOT_STARTED),
        ],
    )
    def test_a_node_waiting_for_another_to_observe_false_ignores_unknown(
        self, watched_type, expected
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes([watched := watched_type(), waiting := ConstTrueNode()])
        waiting.start_condition = watched.observes_false

        executor = _compile_msc(msc)
        executor.tick()

        assert waiting.life_cycle_state == expected

    @pytest.mark.parametrize(
        "read_predicate, expected",
        [
            (lambda node: node.last_observed_true, LifeCycleValues.RUNNING),
            (lambda node: node.observes_true, LifeCycleValues.NOT_STARTED),
        ],
        ids=["last_observed_true", "observes_true"],
    )
    def test_only_the_last_observation_outlasts_the_node_ending(
        self, read_predicate, expected
    ) -> None:
        """
        The watched node ends on the first tick, observing True.

        A node that asks about it only on a later tick still finds what it last
        observed, while what it observes now has been Unknown since it ended.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                watched := ConstTrueNode(),
                delay := ConstTrueNode(),
                late := ConstTrueNode(),
            ]
        )
        watched.success_condition = watched.observes_true
        delay.start_condition = watched.is_succeeded
        late.start_condition = sm.logic_and(
            delay.observes_true, read_predicate(watched)
        )

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert watched.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert late.life_cycle_state == expected

    def test_an_observation_may_read_an_observation_predicate(self) -> None:
        """
        An observation reads the observations the previous tick left behind, so asking
        about one is as legal there as in a condition.
        """
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                watched := ConstTrueNode(),
                observer := NodeObservingAnObservationPredicate(watched_node=watched),
            ]
        )

        executor = _compile_msc(msc)
        for _ in range(2):
            executor.tick()

        assert observer.observation_state == ObservationStateValues.TRUE


class TestEagerStateVariables:
    """
    A node's observation and life cycle variables are available right after
    construction, before it is added to a statechart, so conditions can be wired on
    nested nodes.
    """

    def test_state_variables_available_before_adding_to_statechart(self) -> None:
        node = ConstTrueNode()
        assert node.observation_variable is node.observation_variable
        assert node.life_cycle_variable is node.life_cycle_variable

    def test_nested_self_referential_success_condition_before_compile(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            Parallel(
                nodes=[
                    ConstTrueNode(),
                    barrier := Parallel(
                        nodes=[ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.success_condition = barrier.observes_true
        msc.compiler.add_transitions()
        assert barrier in barrier._success_condition.node_dependencies

    def test_nested_success_condition_survives_json_round_trip(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            outer := Parallel(
                nodes=[
                    ConstTrueNode(),
                    barrier := Parallel(
                        nodes=[ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.success_condition = barrier.observes_true
        msc.add_node(EndStatechart.when_true(outer))
        json_data = msc.create_structure_copy().to_json()
        new_json_data = json.loads(json.dumps(json_data))
        msc_copy = Statechart.from_json(
            new_json_data, context=StatechartContext(world=World())
        )
        msc_copy.compiler.add_transitions()

        barrier_copy = msc_copy.get_node_by_index(barrier.index)
        assert barrier_copy in barrier_copy._success_condition.node_dependencies
        assert barrier_copy.unique_name in str(barrier_copy._success_condition)

    def test_nodes_with_same_name_have_distinct_variable_names(self) -> None:
        first = ConstTrueNode(name="same")
        second = ConstTrueNode(name="same")
        assert first.observation_variable.name != second.observation_variable.name

    def test_self_referential_start_condition_raises_before_add(self) -> None:
        node = ConstTrueNode()
        with pytest.raises(SelfInStartConditionError):
            node.start_condition = node.observes_true


class TestConditionScoping:
    """
    A condition may reference the node itself, a node sharing its parent, or a direct
    child of it.

    References across template levels raise :class:`ConditionScopeError` during
    compilation.
    """

    def test_outside_node_cannot_reference_node_inside_template(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        child = ConstTrueNode()
        msc.add_node(Sequence(nodes=[child]))
        msc.add_node(EndStatechart.when_true(child))

        with pytest.raises(ConditionScopeError):
            kin_sim.compile(statechart=msc)

    def test_template_node_cannot_reference_node_in_sibling_template(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        msc.add_node(first := Parallel(nodes=[node_a]))
        msc.add_node(Parallel(nodes=[node_b]))
        node_b.start_condition = node_a.observes_true
        msc.add_node(EndStatechart.when_true(first))

        with pytest.raises(ConditionScopeError):
            kin_sim.compile(statechart=msc)

    def test_nested_template_node_cannot_reference_outer_node(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        msc.add_node(sequence := Sequence(nodes=[node_a, Parallel(nodes=[node_b])]))
        node_b.pause_condition = node_a.observes_true
        msc.add_node(EndStatechart.when_true(sequence))

        with pytest.raises(ConditionScopeError):
            kin_sim.compile(statechart=msc)

    def test_parent_can_reference_child_through_its_last_observation(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        child = ConstTrueNode()
        parallel = Parallel(nodes=[child])
        msc.add_node(parallel)
        parallel.success_condition = child.last_observed_true
        msc.add_node(EndStatechart.when_true(parallel))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end(timeout=10)

        assert parallel.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_parent_cannot_reference_grandchild(self) -> None:
        """
        Reaching past a direct child skips the node that owns the one being read.
        """
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        grandchild = ConstTrueNode()
        outer = Parallel(nodes=[Parallel(nodes=[grandchild])])
        msc.add_node(outer)
        outer.success_condition = grandchild.last_observed_true
        msc.add_node(EndStatechart.when_true(outer))

        with pytest.raises(ConditionScopeError):
            kin_sim.compile(statechart=msc)

    def test_child_cannot_reference_parent(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        child = ConstTrueNode()
        parallel = Parallel(nodes=[child])
        child.pause_condition = parallel.observes_true
        msc.add_node(parallel)
        msc.add_node(EndStatechart.when_true(parallel))

        with pytest.raises(ConditionScopeError):
            kin_sim.compile(statechart=msc)

    def test_siblings_inside_template_can_reference_each_other(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        node_b.start_condition = node_a.observes_true
        msc.add_node(parallel := Parallel(nodes=[node_a, node_b]))
        msc.add_node(EndStatechart.when_true(parallel))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()

    def test_self_referential_success_condition_inside_template_compiles(self) -> None:
        kin_sim = StatechartExecutor(StatechartContext(world=World()))
        msc = Statechart(context=kin_sim.context)
        msc.add_node(
            outer := Parallel(
                nodes=[
                    ConstTrueNode(),
                    barrier := Parallel(
                        nodes=[ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.success_condition = barrier.observes_true
        msc.add_node(EndStatechart.when_true(outer))

        kin_sim.compile(statechart=msc)
        kin_sim.tick_until_end()


# %% written conditions survive compiling


class TestWrittenConditionsSurviveCompiling:
    """
    What a node brings itself is combined with the conditions written on it, without
    changing them.
    """

    def test_a_node_succeeding_on_observing_true_keeps_its_written_success_condition(
        self,
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeSucceedingOnObservingTrue(
                observation=ObservationStateValues.TRUE
            )
        )
        written = str(node.success_condition)

        _compile_msc(msc)

        assert str(node.success_condition) == written

    def test_a_node_failing_on_observing_false_keeps_its_written_fail_condition(
        self,
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(
            node := NodeFailingOnObservingFalse(
                observation=ObservationStateValues.FALSE
            )
        )
        written = str(node.fail_condition)

        _compile_msc(msc)

        assert str(node.fail_condition) == written

    def test_a_parallel_keeps_its_written_fail_condition(self) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_node(parallel := Parallel(nodes=[ConstTrueNode()]))
        written = str(parallel.fail_condition)

        _compile_msc(msc)

        assert str(parallel.fail_condition) == written

    def test_a_parallel_given_a_fail_condition_still_fails_once_it_cannot_arrive(
        self,
    ) -> None:
        msc = Statechart(context=StatechartContext(world=World()))
        msc.add_nodes(
            [
                never_true := ConstFalseNode(),
                parallel := Parallel(
                    nodes=[
                        NodeFailingOnObservingFalse(
                            observation=ObservationStateValues.FALSE
                        )
                    ]
                ),
            ]
        )
        parallel.fail_condition = never_true.observes_true

        executor = _compile_msc(msc)
        executor.tick()
        executor.tick()

        assert str(parallel.fail_condition) == str(sm.Scalar(never_true.observes_true))
        assert parallel.life_cycle_state == LifeCycleValues.FAILED
