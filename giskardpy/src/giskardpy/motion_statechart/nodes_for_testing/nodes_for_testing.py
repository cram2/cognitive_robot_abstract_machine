from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    CompositeStatechartNode,
    MaintenanceNode,
    NodeArtifacts,
    CancelMotion,
    SelfDecidingNode,
    SelfFailingNode,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.data_types.exceptions import GiskardException
from krrood.symbolic_math.symbolic_math import FloatVariable


@dataclass
class TestNodeAssertionError(GiskardException):
    """
    Raised by test motion statechart nodes when a behaviour they assert on is violated.
    """

    reason: str
    """
    Description of the violated assertion.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return ""


@dataclass(eq=False, repr=False)
class ConstTrueNode(MaintenanceNode):
    """
    A node that has always reached its goal, so ending it always succeeds it.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class ConstFalseNode(MaintenanceNode):
    """
    A node that never reaches its goal, so nothing but being released ever ends it.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False)
class ChangeStateOnEvents(MotionStatechartNode):
    state: str | None = None

    def on_start(self, context: MotionStatechartContext):
        self.state = "on_start"

    def on_pause(self, context: MotionStatechartContext):
        self.state = "on_pause"

    def on_unpause(self, context: MotionStatechartContext):
        self.state = "on_unpause"

    def on_end(self, context: MotionStatechartContext):
        self.state = "on_end"

    def on_reset(self, context: MotionStatechartContext):
        self.state = "on_reset"


@dataclass(repr=False, eq=False)
class TestCompositeStatechartNode(MaintenanceNode, CompositeStatechartNode):
    sub_node1: ConstTrueNode = field(init=False)
    sub_node2: ConstTrueNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.sub_node1 = ConstTrueNode(name="sub muh1")
        self._add_child_to_motion_statechart(self.sub_node1)
        self.sub_node2 = ConstTrueNode(name="sub muh2")
        self._add_child_to_motion_statechart(self.sub_node2)
        self.sub_node1.success_condition = self.sub_node1.observes_true
        self.sub_node2.start_condition = self.sub_node1.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.sub_node2.observation_variable)


@dataclass(repr=False, eq=False)
class TestNestedCompositeStatechartNode(MaintenanceNode, CompositeStatechartNode):
    sub_node1: TestCompositeStatechartNode = field(init=False)
    sub_node2: TestCompositeStatechartNode = field(init=False)
    inner: TestCompositeStatechartNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.inner = TestCompositeStatechartNode(name="inner")
        self._add_child_to_motion_statechart(self.inner)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.inner.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStop(SelfDecidingNode, CompositeStatechartNode):
    """
    Composite statechart node that tests if a child node runs after the parent node has
    stopped.

    Uses a CancelMotion node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountControlCycles = field(init=False)
    ticking2: CountControlCycles = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.ticking1 = CountControlCycles(name="3ticks", control_cycles=3)
        self.ticking2 = CountControlCycles(name="2ticks", control_cycles=2)
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=TestNodeAssertionError(
                reason="Node ticked after template stopped"
            ),
        )

        self._add_children_to_motion_statechart(
            nodes=[
                self.ticking1,
                self.ticking2,
                self.cancel,
            ]
        )
        self.cancel.start_condition = self.ticking1.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking2.observation_variable))


@dataclass(repr=False, eq=False)
class TestEndBeforeStart(CompositeStatechartNode):
    """
    Test if a child node can end before it was started.

    node1 waits 1 tick, then starts node 3. node2 fulfills the success condition of node
    3 immediately. node3 should start when node1 is True and transition to RUNNING with
    Observationstate UNKNOWN. On the next tick, node3 should be ended because its end
    condition is already fulfilled by node2.
    """

    node1: CountControlCycles = field(init=False)
    node2: ConstTrueNode = field(init=False)
    node3: ConstTrueNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.node1 = CountControlCycles(control_cycles=1)
        self.node2 = ConstTrueNode()
        self.node3 = ConstTrueNode()

        self._add_children_to_motion_statechart(
            nodes=[self.node1, self.node2, self.node3]
        )

        self.node3.start_condition = self.node1.observes_true
        self.node3.success_condition = self.node2.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.node3.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStopFromPause(SelfDecidingNode, CompositeStatechartNode):
    """
    Test if child node can transition to RUNNING from PAUSED after parent node is DONE.

    Uses a CancelMotion node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountControlCycles = field(init=False)
    ticking2: CountControlCycles = field(init=False)
    ticking3: CountControlCycles = field(init=False)
    pulse: Pulse = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.ticking1 = CountControlCycles(name="3ticks", control_cycles=3)
        self.ticking2 = CountControlCycles(
            name="trigger_cancel_after_unpause", control_cycles=4
        )
        self.ticking3 = CountControlCycles(name="2ticks", control_cycles=2)
        self.pulse = Pulse()
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=TestNodeAssertionError(
                reason="Node ticked after template stopped"
            ),
        )

        self._add_children_to_motion_statechart(
            nodes=[self.ticking1, self.ticking2, self.ticking3, self.cancel, self.pulse]
        )
        self.pulse.start_condition = self.ticking3.observes_true
        self.ticking2.pause_condition = self.pulse.observes_true
        self.cancel.start_condition = self.ticking2.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking1.observation_variable))


@dataclass(repr=False, eq=False)
class TestUnpauseFromParentPause(SelfDecidingNode, CompositeStatechartNode):
    """
    Tests if a child node paused by its parent transitions from PAUSED back to RUNNING
    once the parent unpauses, while its own pause condition does not hold.
    """

    count_ticks1: CountControlCycles = field(init=False)
    count_ticks2: CountControlCycles = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.count_ticks1 = CountControlCycles(control_cycles=2)
        self.count_ticks2 = CountControlCycles(control_cycles=5)
        self.cancel = CancelMotion(
            name="check_unpause_failed",
            exception=TestNodeAssertionError(reason="Node did not unpause correctly"),
        )

        self._add_child_to_motion_statechart(self.count_ticks1)
        self._add_child_to_motion_statechart(
            Sequence(nodes=[self.count_ticks2, self.cancel])
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :attr:`count_ticks1` is read through its observation, which is what it has
        counted while it runs.
        """
        return NodeArtifacts(
            observation=sm.Scalar(self.count_ticks1.observation_variable)
        )


# %% nodes that differ in what they can be judged by


@dataclass(eq=False, repr=False)
class NodeObservingNothingYet(MotionStatechartNode):
    """
    A node that runs without ever deciding what it observes, so ending it can only
    interrupt it.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_trinary_unknown())


@dataclass(eq=False, repr=False)
class NodeObservingAPredicate(MotionStatechartNode):
    """
    A node whose observation reads a life cycle predicate, which only a transition
    condition may do.
    """

    watched_node: MotionStatechartNode = field(default=None, kw_only=True)
    """
    The node whose verdict this node tries to observe.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.is_succeeded))


@dataclass(eq=False, repr=False)
class NodeObservingLastObservation(MotionStatechartNode):
    """
    A node whose observation reads the observation another node took most recently.
    """

    watched_node: MotionStatechartNode = field(default=None, kw_only=True)
    """
    The node whose most recent observation this node observes.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.last_observation))


@dataclass(eq=False, repr=False)
class NodeObservingAnObservationPredicate(MotionStatechartNode):
    """
    A node that observes whether another node observed True on the previous control
    cycle.
    """

    watched_node: MotionStatechartNode = field(default=None, kw_only=True)
    """
    The node whose observation this node asks about.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.observes_true))


@dataclass(eq=False, repr=False)
class NodeObservingTheOppositeOfAnObservationPredicate(MotionStatechartNode):
    """
    A node that observes True while another node does not observe True.
    """

    watched_node: MotionStatechartNode = field(default=None, kw_only=True)
    """
    The node whose observation this node contradicts.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.logic_not(sm.Scalar(self.watched_node.observes_true))
        )


@dataclass(eq=False, repr=False)
class SelfFailingMaintenanceNode(SelfFailingNode, MaintenanceNode):
    """
    A node that fails itself once it observes False and leaves succeeding to its owner.
    """

    observation: ObservationStateValues = field(kw_only=True)
    """
    What this node observes on every control cycle.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(float(self.observation)))


@dataclass(repr=False, eq=False)
class NodeDeclaringItsOwnFailure(MaintenanceNode, CompositeStatechartNode):
    """
    A node short of its goal that declares it cannot continue, which is what a node may
    decide about itself where succeeding is left to its owner.

    Being a composite statechart node is what gives it an :meth:`expand` hook to declare
    the failure in; it runs no children of its own.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.fail_condition = sm.Scalar.const_true()

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


# %% goals that end their child


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeCuttingOffItsChildAtItsGoal(CompositeStatechartNode):
    """
    Composite statechart node whose child has reached its goal but is never ended on its
    own terms, so the child is only ever taken down by this node ending.
    """

    child: ConstTrueNode = field(init=False)
    """
    The child that sits at its goal until it is cut off.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = ConstTrueNode()
        self._add_child_to_motion_statechart(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeCuttingOffItsChild(CompositeStatechartNode):
    """
    Composite statechart node whose child is short of its goal and is never ended on its
    own terms, so the child is only ever taken down by this node ending.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that keeps running until it is cut off.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = ConstFalseNode()
        self._add_child_to_motion_statechart(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeWithChildInterruptedBySibling(CompositeStatechartNode):
    """
    Composite statechart node whose child is interrupted by a sibling on the first tick,
    so that a caller ending this node on that same tick makes the two ways of being
    interrupted compete.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    Turns true on the first tick, which is what interrupts the child.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that is interrupted while its observation is false.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_motion_statechart(nodes=[self.trigger, self.child])
        self.child.interrupt_condition = self.trigger.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeWithChildFailingOnItsOwn(CompositeStatechartNode):
    """
    Composite statechart node whose child declares its own failure on the first tick, so
    that a caller ending this node on that same tick makes the child's own verdict
    compete with being cut off.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    Turns true on the first tick, which is what makes the child declare its failure.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that gives up while its observation is false.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_motion_statechart(nodes=[self.trigger, self.child])
        self.child.fail_condition = self.trigger.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeWithChildSucceedingOnItsOwn(CompositeStatechartNode):
    """
    Composite statechart node whose child declares its own success on the first tick, so
    that a caller ending this node on that same tick makes the child's own verdict
    compete with being cut off.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    Turns true on the first tick, which is what makes the child declare its success.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that declares its success while its observation is false.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_motion_statechart(nodes=[self.trigger, self.child])
        self.child.success_condition = self.trigger.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeWithChildStartingLate(CompositeStatechartNode):
    """
    Composite statechart node whose child waits for a delay before it starts, so the
    child's start is decided while this node is already running and its ending
    conditions have a settled value.
    """

    delay_in_control_cycles: int = field(default=2, kw_only=True)
    """
    How many control cycles pass before the child's start condition turns true.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child whose start is being observed.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        delay = CountControlCycles(control_cycles=self.delay_in_control_cycles)
        self.child = ConstFalseNode()
        self._add_children_to_motion_statechart(nodes=[delay, self.child])
        self.child.start_condition = delay.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeCuttingOffItsUndecidedChild(CompositeStatechartNode):
    """
    Composite statechart node whose child never decides what it observes, so this node
    ending is the only thing that ever ends it.
    """

    child: NodeObservingNothingYet = field(init=False)
    """
    The child that observes nothing until it is ended.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = NodeObservingNothingYet()
        self._add_child_to_motion_statechart(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeCuttingOffItsGrandchild(CompositeStatechartNode):
    """
    Composite statechart node holding another composite statechart node, so that ending
    it reaches a node more than one level below it.
    """

    inner_node: CompositeStatechartNodeCuttingOffItsChild = field(init=False)
    """
    The node between this one and the grandchild.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.inner_node = CompositeStatechartNodeCuttingOffItsChild()
        self._add_child_to_motion_statechart(self.inner_node)

    @property
    def grandchild(self) -> ConstFalseNode:
        """
        :return: The node two levels below this node, which is short of its goal until
            this node ends.
        """
        return self.inner_node.child

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


# %% nodes that record what the statechart does to them


class LifeCycleCallback(Enum):
    """
    A callback the motion statechart runs on a node when its life cycle state changes.
    """

    START = auto()
    """
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_start`.
    """

    PAUSE = auto()
    """
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_pause`.
    """

    UNPAUSE = auto()
    """
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_unpause`.
    """

    END = auto()
    """
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_end`.
    """

    RESET = auto()
    """
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_reset`.
    """


@dataclass(eq=False, repr=False)
class NodeRecordingItsCallbacks(MotionStatechartNode):
    """
    A node that never reaches its goal and records every life cycle callback run on it.
    """

    callbacks: List[LifeCycleCallback] = field(default_factory=list, init=False)
    """
    The callbacks run on this node since :meth:`take_callbacks` was last called, in the
    order they ran.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())

    def take_callbacks(self) -> List[LifeCycleCallback]:
        """
        :return: The callbacks recorded so far, which are forgotten afterwards.
        """
        callbacks = self.callbacks
        self.callbacks = []
        return callbacks

    def on_start(self, context: MotionStatechartContext):
        self.callbacks.append(LifeCycleCallback.START)

    def on_pause(self, context: MotionStatechartContext):
        self.callbacks.append(LifeCycleCallback.PAUSE)

    def on_unpause(self, context: MotionStatechartContext):
        self.callbacks.append(LifeCycleCallback.UNPAUSE)

    def on_end(self, context: MotionStatechartContext):
        self.callbacks.append(LifeCycleCallback.END)

    def on_reset(self, context: MotionStatechartContext):
        self.callbacks.append(LifeCycleCallback.RESET)


@dataclass(eq=False, repr=False)
class NodeObservingTrueOnlyOnTick(MotionStatechartNode):
    """
    A node whose observation expression is False but whose
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick` overrides
    it with True, counting how often it is ticked.
    """

    on_tick_calls: int = field(default=0, init=False)
    """
    How often :meth:`on_tick` was called.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        self.on_tick_calls += 1
        return ObservationStateValues.TRUE


@dataclass(eq=False, repr=False)
class NodeWritingAVariableOnStart(MotionStatechartNode):
    """
    A node that sets a float variable to True when it starts, so what its start callback
    wrote can be observed by another node.
    """

    variable: FloatVariable = field(init=False)
    """
    The variable written when this node starts, False until then.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.variable = FloatVariable(f"{self.name}/written_on_start")
        context.float_variable_data.register_expression(self.variable)
        return NodeArtifacts(observation=sm.Scalar.const_trinary_unknown())

    def on_start(self, context: MotionStatechartContext):
        context.float_variable_data.set_value(
            self.variable, float(ObservationStateValues.TRUE)
        )


@dataclass(eq=False, repr=False)
class NodeObservingAWrittenVariable(MotionStatechartNode):
    """
    A node that observes True once another node's start callback wrote its variable.
    """

    writer: NodeWritingAVariableOnStart = field(kw_only=True)
    """
    The node whose written variable this node observes.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        """
        :return: :attr:`writer`, which creates the variable this node reads.
        """
        return [self.writer]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.writer.variable,
                float(ObservationStateValues.TRUE),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeObservingItsSecondChildRun(CompositeStatechartNode):
    """
    Composite statechart node that observes True once its second child is running, which
    starts once its first child succeeded, so the second child is started and then cut off
    by this node ending.
    """

    first: ConstTrueNode = field(init=False)
    """
    The child that succeeds as soon as it observes its goal.
    """

    second: NodeRecordingItsCallbacks = field(init=False)
    """
    The child that starts once :attr:`first` succeeded and is cut off by this node.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.first = ConstTrueNode()
        self.second = NodeRecordingItsCallbacks()
        self._add_children_to_motion_statechart(nodes=[self.first, self.second])
        self.first.success_condition = self.first.observes_true
        self.second.start_condition = self.first.is_succeeded

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.second.life_cycle_variable,
                int(LifeCycleValues.RUNNING),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeWithARecordingChild(CompositeStatechartNode):
    """
    Composite statechart node holding one child that records its callbacks and would start
    whenever it may.
    """

    child: NodeRecordingItsCallbacks = field(init=False)
    """
    The child whose callbacks are recorded.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = NodeRecordingItsCallbacks()
        self._add_child_to_motion_statechart(self.child)


@dataclass(repr=False, eq=False)
class CompositeStatechartNodeObservingItsCancelMotionRun(CompositeStatechartNode):
    """
    Composite statechart node that observes True once its :class:`CancelMotion` child is
    running, which starts once its other child observes True, so the cancel motion is
    started and then cut off by this node ending.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    The child whose observation starts :attr:`cancel`.
    """

    cancel: CancelMotion = field(init=False)
    """
    The child that is cut off right after starting.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.cancel = CancelMotion(
            exception=TestNodeAssertionError(reason="cancelled right after starting")
        )
        self._add_children_to_motion_statechart(nodes=[self.trigger, self.cancel])
        self.cancel.start_condition = self.trigger.observes_true

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.cancel.life_cycle_variable,
                int(LifeCycleValues.RUNNING),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )
