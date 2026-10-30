from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    RepeatUntil,
    StoppedWhenTrue,
)
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    NoChildChooserForNodeError,
    SeveralChildChoosersForNodeError,
)
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.monitors import CountNodeResets, CountTicks
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import (
    ConstFalseNode,
    NodeFailingOnObservingFalse,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart

# %% mimics


@dataclass
class ChooserAnsweringInTurn(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Has no choice for the first few questions, then gives the prepared children one
    after another, then says no child is left.
    """

    children: list[StatechartNode | None]
    """
    The children still to give, in order, None for no child left.
    """

    unanswered_questions: int = 0
    """
    How many questions are still answered with having no choice yet.
    """

    asked_nodes: list[CompositeNodeChoosingItsChild] = field(default_factory=list)
    """
    Every node that asked, once per question.
    """

    def has_choice_for(self, node: CompositeNodeChoosingItsChild) -> bool:
        self.asked_nodes.append(node)
        if self.unanswered_questions == 0:
            return True
        self.unanswered_questions -= 1
        return False

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        if not self.children:
            return None
        return self.children.pop(0)


@dataclass
class ChooserGivingUpOnItsChildren(ChooserAnsweringInTurn):
    """
    Gives the prepared children in turn, each with the failure monitors prepared for it,
    and notes in which order it was asked about a child it gave.
    """

    failure_monitors: list[list[StatechartNode]] = field(default_factory=list)
    """
    The failure monitors still to give, one list per child, in the order of
    :attr:`children`.
    """

    calls: list[str] = field(default_factory=list)
    """
    Which of the questions about a given child were asked, in order, with whether the
    child belonged to the statechart by then.
    """

    def prepare_chosen_child(
        self,
        node: CompositeNodeChoosingItsChild,
        child: StatechartNode,
    ) -> None:
        self.calls.append(f"prepare {child.name}, joined: {child.index is not None}")

    def failure_monitors_for(
        self,
        node: CompositeNodeChoosingItsChild,
        child: StatechartNode,
    ) -> list[StatechartNode]:
        self.calls.append(f"monitors {child.name}")
        return self.failure_monitors.pop(0)


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


# %% helpers


def _succeeding_child(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _failing_child(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: A node that fails on the tick after it starts.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.FALSE
    )


def _undecided_child(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that never ends on its own.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.UNKNOWN
    )


def _run_count(statechart: Statechart, node: StatechartNode) -> int:
    """
    :return: How often `node` went from not started to running.
    """
    life_cycles = statechart.state_history.get_life_cycle_history_of_node(node)
    return sum(
        previous == LifeCycleValues.NOT_STARTED and current == LifeCycleValues.RUNNING
        for previous, current in zip(life_cycles, life_cycles[1:])
    )


def _run_repeated_until_decided(
    executor: StatechartExecutor,
    chooser: ChildChooser,
    task: StatechartNode,
    resets_until_stopped: int,
) -> Statechart:
    """
    Run `task` again whenever it failed, until it succeeds or was reset
    `resets_until_stopped` times.

    :return: The statechart that ran.
    """
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    loop = RepeatUntil(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(
            name="counter", node=task, target=resets_until_stopped
        ),
    )
    statechart.add_node(loop)
    statechart.add_node(EndStatechart.when_true(loop))
    statechart.add_node(EndStatechart.when_failed(loop))
    executor.compile(statechart)
    executor.tick_until_end(timeout=60)
    return statechart


def _run_choosing_node(
    executor: StatechartExecutor,
    chooser: ChildChooser,
    end_node_factory=EndStatechart.when_true,
) -> CompositeNodeChoosingItsChild:
    """
    Run a statechart holding one choosing node until it ends the statechart.

    :return: The choosing node.
    """
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    statechart.add_node(choosing_node)
    statechart.add_node(end_node_factory(choosing_node))
    executor.compile(statechart)
    executor.tick_until_end(timeout=30)
    return choosing_node


# %% choosing


def test_the_node_succeeds_with_the_child_it_chose(
    statechart_executor: StatechartExecutor,
) -> None:
    child = _succeeding_child("child")

    choosing_node = _run_choosing_node(
        statechart_executor, ChooserAnsweringInTurn([child])
    )

    assert choosing_node.children == [child]
    assert child.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_failed_child_makes_the_node_choose_again(
    statechart_executor: StatechartExecutor,
) -> None:
    failing = _failing_child("failing")
    succeeding = _succeeding_child("succeeding")

    choosing_node = _run_choosing_node(
        statechart_executor,
        ChooserAnsweringInTurn([failing, succeeding]),
    )

    assert choosing_node.children == [failing, succeeding]
    assert failing.life_cycle_state == LifeCycleValues.FAILED
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_no_child_left_fails_the_node(statechart_executor: StatechartExecutor) -> None:
    choosing_node = _run_choosing_node(
        statechart_executor,
        ChooserAnsweringInTurn([]),
        end_node_factory=EndStatechart.when_failed,
    )

    assert choosing_node.children == []
    assert choosing_node.life_cycle_state == LifeCycleValues.FAILED


def test_a_pending_choice_is_asked_again_on_the_next_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    child = _succeeding_child("child")
    chooser = ChooserAnsweringInTurn([child], unanswered_questions=2)

    choosing_node = _run_choosing_node(statechart_executor, chooser)

    assert chooser.asked_nodes == [choosing_node] * 3
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_that_is_not_running_is_not_asked(
    statechart_executor: StatechartExecutor,
) -> None:
    chooser = ChooserAnsweringInTurn([_succeeding_child("child")])
    statechart_executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=statechart_executor.context)
    never_true = ConstFalseNode(name="never true")
    statechart.add_node(never_true)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    choosing_node.start_condition = never_true.observes_true
    statechart.add_node(choosing_node)

    statechart_executor.compile(statechart)
    statechart_executor.tick()

    assert chooser.asked_nodes == []
    assert choosing_node.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_choosing_keeps_the_state_and_history_of_the_nodes_already_there(
    statechart_executor: StatechartExecutor,
) -> None:
    chooser = ChooserAnsweringInTurn([_succeeding_child("child")])
    statechart_executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=statechart_executor.context)
    earlier = _succeeding_child("earlier")
    statechart.add_node(earlier)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    choosing_node.start_condition = earlier.is_succeeded
    statechart.add_node(choosing_node)
    statechart.add_node(EndStatechart.when_true(choosing_node))
    statechart_executor.compile(statechart)
    statechart_executor.tick()
    assert earlier.life_cycle_state == LifeCycleValues.SUCCEEDED
    recorded_before_choosing = len(statechart.state_history)

    statechart_executor.tick_until_end(timeout=30)

    assert earlier.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert len(statechart.state_history) == recorded_before_choosing + (
        statechart_executor.tick_count - 1
    )


def test_a_choice_compiles_the_statechart_once(
    statechart_context,
) -> None:
    counting = ExtensionCountingCompiles()
    executor = StatechartExecutor(context=statechart_context, extensions=[counting])
    child = _succeeding_child("child")
    executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserAnsweringInTurn([child])])
    )
    statechart = Statechart(context=executor.context)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    statechart.add_node(choosing_node)

    executor.compile(statechart)

    assert choosing_node.children == [child]
    assert counting.compile_count == 2


def test_the_node_observes_what_its_latest_child_observed(
    statechart_executor: StatechartExecutor,
) -> None:
    child: StatechartNode = _succeeding_child("child")

    choosing_node = _run_choosing_node(
        statechart_executor, ChooserAnsweringInTurn([child])
    )

    assert choosing_node.last_observation_state == ObservationStateValues.TRUE


@dataclass
class ExtensionHoldingStill(TickingExecutorExtension):
    """
    An executor extension that notes when it was told to hold still.
    """

    is_holding_still: bool = False
    """
    Whether :meth:`before_recompile` ran.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        self.is_holding_still = True
        return True


@dataclass
class ChooserNotingWhetherTheExtensionHeldStill(
    ChildChooser[CompositeNodeChoosingItsChild]
):
    """
    Notes whether the extension held still by the time it was asked.
    """

    extension: ExtensionHoldingStill
    """
    The extension to look at.
    """

    child: StatechartNode
    """
    The child to choose.
    """

    held_still_when_asked: list[bool] = field(default_factory=list)
    """
    Whether the extension held still, once per question.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        self.held_still_when_asked.append(self.extension.is_holding_still)
        return self.child


def test_the_extensions_hold_still_before_a_node_chooses(statechart_context) -> None:
    """
    Choosing blocks the tick, so whatever an extension is driving is stopped first.
    """
    extension = ExtensionHoldingStill()
    executor = StatechartExecutor(context=statechart_context, extensions=[extension])
    chooser = ChooserNotingWhetherTheExtensionHeldStill(
        extension=extension, child=_succeeding_child("child")
    )
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    statechart.add_node(CompositeNodeChoosingItsChild(name="choosing"))

    executor.compile(statechart)

    assert chooser.held_still_when_asked == [True]


@dataclass
class EventLog:
    """
    The order in which nodes started and children were chosen.
    """

    events: list[str] = field(default_factory=list)
    """
    What happened, oldest first.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeNotingItsStart(NodeObservingAFixedValue):
    """
    Notes in a log when it starts.
    """

    log: EventLog
    """
    The log this node notes its start in.
    """

    def on_start(self) -> None:
        self.log.events.append(f"{self.name} started")


@dataclass
class ChooserNotingItsChoice(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Gives one child and notes in a log when it chooses it.
    """

    log: EventLog
    """
    The log this chooser notes its choice in.
    """

    child: StatechartNode
    """
    The child to give.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        self.log.events.append(f"{node.name} chose")
        return self.child


def test_a_node_chooses_before_the_nodes_starting_beside_it_act(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A child is chosen against the world as it was before the nodes that start in the
    same tick act on starting.
    """
    log = EventLog()
    chooser = ChooserNotingItsChoice(log=log, child=_succeeding_child("child"))
    statechart_executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=statechart_executor.context)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    sibling = NodeNotingItsStart(
        name="sibling", observation=ObservationStateValues.UNKNOWN, log=log
    )
    statechart.add_nodes([choosing_node, sibling])

    statechart_executor.compile(statechart)
    statechart_executor.tick()

    assert log.events == [
        f"{choosing_node.name} chose",
        f"{sibling.name} started",
    ]


# %% choosing with the chooser of the node's kind


@dataclass(eq=False, repr=False, kw_only=True)
class NodeChoosingItsChildByItsOwnRule(CompositeNodeChoosingItsChild):
    """
    A kind of node choosing its child that has a chooser of its own.
    """


@dataclass
class ChooserOfItsOwnKind(ChildChooser[NodeChoosingItsChildByItsOwnRule]):
    """
    Chooses for nodes of :class:`NodeChoosingItsChildByItsOwnRule` only, and never has a
    child for them.
    """

    asked_nodes: list[NodeChoosingItsChildByItsOwnRule] = field(default_factory=list)
    """
    Every node that asked, once per question.
    """

    def choose_child(
        self, node: NodeChoosingItsChildByItsOwnRule
    ) -> StatechartNode | None:
        self.asked_nodes.append(node)
        return None


def test_a_node_asks_the_chooser_of_its_own_kind(
    statechart_executor: StatechartExecutor,
) -> None:
    general_chooser = ChooserAnsweringInTurn([_succeeding_child("general")])
    own_kind_chooser = ChooserOfItsOwnKind()
    statechart_executor.context.add_extension(
        ChildChooserAccess(choosers=[general_chooser, own_kind_chooser])
    )
    statechart = Statechart(context=statechart_executor.context)
    choosing_node = NodeChoosingItsChildByItsOwnRule(name="choosing")
    statechart.add_node(choosing_node)
    statechart.add_node(EndStatechart.when_failed(choosing_node))

    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=30)

    assert own_kind_chooser.asked_nodes == [choosing_node]
    assert general_chooser.asked_nodes == []


def test_a_node_without_a_chooser_of_its_kind_cannot_choose(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart_executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserOfItsOwnKind()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(CompositeNodeChoosingItsChild(name="choosing"))

    with pytest.raises(NoChildChooserForNodeError):
        statechart_executor.compile(statechart)


def test_a_node_with_two_choosers_of_its_kind_cannot_choose(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart_executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserOfItsOwnKind(), ChooserOfItsOwnKind()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(NodeChoosingItsChildByItsOwnRule(name="choosing"))

    with pytest.raises(SeveralChildChoosersForNodeError):
        statechart_executor.compile(statechart)


# %% giving up on a child


def test_a_child_is_given_up_on_once_one_of_its_failure_monitors_fires(
    statechart_executor: StatechartExecutor,
) -> None:
    undecided = _undecided_child("undecided")
    timeout = CountTicks(name="timeout", ticks=3)
    succeeding = _succeeding_child("succeeding")
    chooser = ChooserGivingUpOnItsChildren(
        [undecided, succeeding], failure_monitors=[[timeout], []]
    )

    choosing_node = _run_choosing_node(statechart_executor, chooser)

    assert choosing_node.chosen_children == [undecided, succeeding]
    assert undecided.life_cycle_state == LifeCycleValues.FAILED
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_child_reaching_its_goal_succeeds_although_a_monitor_fires_on_that_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    child = _succeeding_child("child")
    firing_at_once = NodeObservingAFixedValue(
        name="firing at once", observation=ObservationStateValues.TRUE
    )
    chooser = ChooserGivingUpOnItsChildren([child], failure_monitors=[[firing_at_once]])

    choosing_node = _run_choosing_node(statechart_executor, chooser)

    assert child.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert choosing_node.failure_reasons == []


def test_the_failure_monitors_of_a_child_stop_once_it_ended(
    statechart_executor: StatechartExecutor,
) -> None:
    failing = _failing_child("failing")
    never_firing = ConstFalseNode(name="never firing")
    undecided = _undecided_child("undecided")
    timeout = CountTicks(name="timeout", ticks=5)
    chooser = ChooserGivingUpOnItsChildren(
        [failing, undecided], failure_monitors=[[never_firing], [timeout]]
    )

    _run_choosing_node(
        statechart_executor, chooser, end_node_factory=EndStatechart.when_failed
    )

    assert never_firing.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert timeout.last_observation_state == ObservationStateValues.TRUE


def test_the_failure_monitors_of_a_child_are_not_among_the_chosen_children(
    statechart_executor: StatechartExecutor,
) -> None:
    undecided = _undecided_child("undecided")
    timeout = CountTicks(name="timeout", ticks=3)
    chooser = ChooserGivingUpOnItsChildren([undecided], failure_monitors=[[timeout]])

    choosing_node = _run_choosing_node(
        statechart_executor, chooser, end_node_factory=EndStatechart.when_failed
    )

    assert choosing_node.chosen_children == [undecided]
    assert choosing_node.failure_monitors_of(undecided) == [timeout]
    assert choosing_node.latest_child is undecided


def test_the_failure_reasons_are_the_monitors_that_gave_up_on_a_child(
    statechart_executor: StatechartExecutor,
) -> None:
    undecided = _undecided_child("undecided")
    timeout = CountTicks(name="timeout", ticks=3)
    never_firing = ConstFalseNode(name="never firing")
    succeeding = _succeeding_child("succeeding")
    chooser = ChooserGivingUpOnItsChildren(
        [undecided, succeeding], failure_monitors=[[never_firing, timeout], []]
    )

    choosing_node = _run_choosing_node(statechart_executor, chooser)

    assert choosing_node.failure_reasons_of(undecided) == [timeout]
    assert choosing_node.failure_reasons_of(succeeding) == []
    assert choosing_node.failure_reasons == [timeout]


def test_a_child_failing_on_its_own_has_no_failure_reasons(
    statechart_executor: StatechartExecutor,
) -> None:
    failing = _failing_child("failing")
    never_firing = ConstFalseNode(name="never firing")
    succeeding = _succeeding_child("succeeding")
    chooser = ChooserGivingUpOnItsChildren(
        [failing, succeeding], failure_monitors=[[never_firing], []]
    )

    choosing_node = _run_choosing_node(statechart_executor, chooser)

    assert failing.life_cycle_state == LifeCycleValues.FAILED
    assert choosing_node.failure_reasons_of(failing) == []


def test_a_chosen_child_is_prepared_once_it_joined_and_before_its_monitors_are_asked_for(
    statechart_executor: StatechartExecutor,
) -> None:
    child = _succeeding_child("child")
    chooser = ChooserGivingUpOnItsChildren([child], failure_monitors=[[]])

    _run_choosing_node(statechart_executor, chooser)

    assert chooser.calls == ["prepare child, joined: True", "monitors child"]


# %% choosing again after a reset


def test_a_reset_node_chooses_again_instead_of_running_its_earlier_children(
    statechart_executor: StatechartExecutor,
) -> None:
    failing = _failing_child("failing")
    undecided = _undecided_child("undecided")
    succeeding = _succeeding_child("succeeding")
    chooser = ChooserAnsweringInTurn([failing, undecided, succeeding])
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    stopped = StoppedWhenTrue(
        name="stopped",
        monitor=CountTicks(name="timeout", ticks=6),
        monitored_node=choosing_node,
    )

    statechart = _run_repeated_until_decided(
        statechart_executor, chooser, stopped, resets_until_stopped=3
    )

    assert chooser.asked_nodes == [choosing_node] * 3
    assert choosing_node.children == [failing, undecided, succeeding]
    assert _run_count(statechart, failing) == 1
    assert _run_count(statechart, undecided) == 1
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_that_ran_out_of_children_chooses_again_after_a_reset(
    statechart_executor: StatechartExecutor,
) -> None:
    succeeding = _succeeding_child("succeeding")
    chooser = ChooserAnsweringInTurn([None, succeeding])
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")

    _run_repeated_until_decided(
        statechart_executor, chooser, choosing_node, resets_until_stopped=3
    )

    assert chooser.asked_nodes == [choosing_node] * 2
    assert choosing_node.children == [succeeding]
    assert not choosing_node.ran_out_of_children
    assert choosing_node.life_cycle_state == LifeCycleValues.SUCCEEDED
