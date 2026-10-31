from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


import krrood.symbolic_math.symbolic_math as sm
from cramph.node import EndedByOwner, SucceedsOnObservingTrue, FailsOnObservingFalse
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import StatechartError
from cramph.composites import Sequence
from cramph.node import (
    StatechartNode,
    CompositeNode,
    NodeArtifacts,
    CancelStatechart,
    expanded_child_field,
)
from cramph.monitors import CountTicks, Pulse
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.world_entity import WorldEntityWithID


@dataclass
class NodeAssertionError(StatechartError):
    """
    Raised by test statechart nodes when a behaviour they assert on is violated.
    """

    reason: str
    """
    Description of the violated assertion.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return ""


@dataclass(eq=False, repr=False, kw_only=True)
class ConstTrueNode(EndedByOwner, StatechartNode):
    """
    A node that has always reached its goal, so ending it always succeeds it.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False, kw_only=True)
class ConstFalseNode(EndedByOwner, StatechartNode):
    """
    A node that never reaches its goal, so nothing but being released ever ends it.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(eq=False, repr=False, kw_only=True)
class NodeReferringToAWorldEntity(ConstTrueNode):
    """
    A node holding an entity of the world its statechart runs in.
    """

    world_entity: WorldEntityWithID
    """
    The entity of the world this node refers to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeWithOwnStructureCopy(EndedByOwner, StatechartNode):
    """
    A kind of node declared outside of the statechart's own node classes, whose structure
    copy is an instance of this kind.
    """

    def create_structure_copy(self) -> NodeWithOwnStructureCopy:
        return NodeWithOwnStructureCopy(name=self.name)


@dataclass(eq=False, repr=False, kw_only=True)
class SpecializedNodeWithOwnStructureCopy(NodeWithOwnStructureCopy):
    """
    A specialization of :class:`NodeWithOwnStructureCopy` whose structure copy falls back to that kind.
    """

    detail: int = 0
    """
    A value only the specialization has, and its structure copy does not.
    """


@dataclass(repr=False, eq=False, kw_only=True)
class ChangeStateOnEvents(EndedByOwner, StatechartNode):
    """
    Records the name of the life cycle hook that ran last.
    """

    state: str | None = None
    """
    The name of the life cycle hook that ran last, or None before any ran.
    """

    def on_start(self) -> None:
        self.state = "on_start"

    def on_pause(self) -> None:
        self.state = "on_pause"

    def on_unpause(self) -> None:
        self.state = "on_unpause"

    def on_end(self) -> None:
        self.state = "on_end"

    def on_reset(self) -> None:
        self.state = "on_reset"


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChainedChildren(EndedByOwner, CompositeNode):
    """
    A composite node whose second child starts once its first child observes True, and
    which observes what its second child observes.
    """

    sub_node1: ConstTrueNode = expanded_child_field()
    """
    The child that starts right away.
    """

    sub_node2: ConstTrueNode = expanded_child_field()
    """
    The child that starts once :attr:`sub_node1` observes True.
    """

    def expand(self) -> None:
        self.sub_node1 = ConstTrueNode(name="sub muh1")
        self._add_child_to_statechart(self.sub_node1)
        self.sub_node2 = ConstTrueNode(name="sub muh2")
        self._add_child_to_statechart(self.sub_node2)
        self.sub_node1.success_condition = self.sub_node1.observes_true
        self.sub_node2.start_condition = self.sub_node1.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=self.sub_node2.observation_variable)


@dataclass(repr=False, eq=False, kw_only=True)
class SequenceCreatingItsSteps(Sequence):
    """
    A sequence that creates its three steps when it expands, so they are not among the
    fields it is constructed with.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The steps, created when the sequence expands.
    """

    def expand(self) -> None:
        for _ in range(3):
            self.add_node(ConstTrueNode())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithNestedCompositeChild(EndedByOwner, CompositeNode):
    """
    A composite node with a single composite child, observing what that child observes.
    """

    inner: CompositeNodeWithChainedChildren = expanded_child_field()
    """
    The composite child.
    """

    def expand(self) -> None:
        self.inner = CompositeNodeWithChainedChildren(name="inner")
        self._add_child_to_statechart(self.inner)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.inner.observation_variable))


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCancellingIfChildRunsAfterItsEnd(
    SucceedsOnObservingTrue, CompositeNode
):
    """
    A composite node that cancels the statechart if one of its children runs after it
    has ended.

    Uses a CancelStatechart node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountTicks = expanded_child_field()
    """
    Counts three ticks, one more than :attr:`ticking2`, so it only observes True if it
    still runs after this node ended.
    """

    ticking2: CountTicks = expanded_child_field()
    """
    Counts two ticks; this node observes what it observes.
    """

    cancel: CancelStatechart = expanded_child_field()
    """
    Cancels the statechart once :attr:`ticking1` observes True.
    """

    def expand(self) -> None:
        self.ticking1 = CountTicks(name="3ticks", ticks=3)
        self.ticking2 = CountTicks(name="2ticks", ticks=2)
        self.cancel = CancelStatechart(
            name="Cancel_on_tick_after_done",
            exception=NodeAssertionError(reason="Node ticked after template stopped"),
        )

        self._add_children_to_statechart(
            nodes=[
                self.ticking1,
                self.ticking2,
                self.cancel,
            ]
        )
        self.cancel.start_condition = self.ticking1.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking2.observation_variable))


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChildSucceedingBeforeItStarts(EndedByOwner, CompositeNode):
    """
    A composite node whose child has its success condition met before it starts.

    node1 waits 1 tick, then starts node 3. node2 fulfills the success condition of node
    3 immediately. node3 should start when node1 is True and transition to RUNNING with
    Observationstate UNKNOWN. On the next tick, node3 should be ended because its end
    condition is already fulfilled by node2.
    """

    node1: CountTicks = expanded_child_field()
    """
    Counts one tick, then starts :attr:`node3`.
    """

    node2: ConstTrueNode = expanded_child_field()
    """
    Meets the success condition of :attr:`node3` right away.
    """

    node3: ConstTrueNode = expanded_child_field()
    """
    The child whose success condition holds before it starts.
    """

    def expand(self) -> None:
        self.node1 = CountTicks(ticks=1)
        self.node2 = ConstTrueNode()
        self.node3 = ConstTrueNode()

        self._add_children_to_statechart(nodes=[self.node1, self.node2, self.node3])

        self.node3.start_condition = self.node1.observes_true
        self.node3.success_condition = self.node2.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.node3.observation_variable))


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCancellingIfPausedChildResumesAfterItsEnd(
    SucceedsOnObservingTrue, CompositeNode
):
    """
    A composite node that cancels the statechart if a paused child resumes after it has
    ended.

    Uses a CancelStatechart node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountTicks = expanded_child_field()
    """
    Counts three ticks; this node observes what it observes.
    """

    ticking2: CountTicks = expanded_child_field()
    """
    Paused by :attr:`pulse`; it only reaches its four ticks if it resumes after
    this node ended.
    """

    ticking3: CountTicks = expanded_child_field()
    """
    Counts two ticks, then starts :attr:`pulse`.
    """

    pulse: Pulse = expanded_child_field()
    """
    Pauses :attr:`ticking2` while it observes True.
    """

    cancel: CancelStatechart = expanded_child_field()
    """
    Cancels the statechart once :attr:`ticking2` observes True.
    """

    def expand(self) -> None:
        self.ticking1 = CountTicks(name="3ticks", ticks=3)
        self.ticking2 = CountTicks(name="trigger_cancel_after_unpause", ticks=4)
        self.ticking3 = CountTicks(name="2ticks", ticks=2)
        self.pulse = Pulse()
        self.cancel = CancelStatechart(
            name="Cancel_on_tick_after_done",
            exception=NodeAssertionError(reason="Node ticked after template stopped"),
        )

        self._add_children_to_statechart(
            nodes=[self.ticking1, self.ticking2, self.ticking3, self.cancel, self.pulse]
        )
        self.pulse.start_condition = self.ticking3.observes_true
        self.ticking2.pause_condition = self.pulse.observes_true
        self.cancel.start_condition = self.ticking2.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking1.observation_variable))


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeResumingItsPausedChildren(SucceedsOnObservingTrue, CompositeNode):
    """
    A composite node whose children, paused along with it, resume once it resumes while
    their own pause conditions do not hold.
    """

    count_ticks1: CountTicks = expanded_child_field()
    """
    Counts two ticks; this node observes what it observes.
    """

    count_ticks2: CountTicks = expanded_child_field()
    """
    Counts five ticks, then starts :attr:`cancel`.
    """

    cancel: CancelStatechart = expanded_child_field()
    """
    Cancels the statechart if :attr:`count_ticks2` reached its count.
    """

    def expand(self) -> None:
        self.count_ticks1 = CountTicks(ticks=2)
        self.count_ticks2 = CountTicks(ticks=5)
        self.cancel = CancelStatechart(
            name="check_unpause_failed",
            exception=NodeAssertionError(reason="Node did not unpause correctly"),
        )

        self._add_child_to_statechart(self.count_ticks1)
        self._add_child_to_statechart(Sequence(nodes=[self.count_ticks2, self.cancel]))

    def build_artifacts(self) -> NodeArtifacts:
        """
        :attr:`count_ticks1` is read through its observation, which is what it has
        counted while it runs.
        """
        return NodeArtifacts(
            observation=sm.Scalar(self.count_ticks1.observation_variable)
        )


# %% nodes that differ in what they can be judged by


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingNothingYet(EndedByOwner, StatechartNode):
    """
    A node that runs without ever deciding what it observes, so ending it can only
    interrupt it.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_trinary_unknown())


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingAPredicate(EndedByOwner, StatechartNode):
    """
    A node whose observation reads a life cycle predicate, which only a transition
    condition may do.
    """

    watched_node: StatechartNode = None
    """
    The node whose outcome this node tries to observe.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.is_succeeded))


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingLastObservation(EndedByOwner, StatechartNode):
    """
    A node whose observation reads the observation another node took most recently.
    """

    watched_node: StatechartNode = None
    """
    The node whose most recent observation this node observes.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.last_observation))


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingAnObservationPredicate(EndedByOwner, StatechartNode):
    """
    A node that observes whether another node observed True on the previous
    tick.
    """

    watched_node: StatechartNode = None
    """
    The node whose observation this node asks about.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.observes_true))


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingTheOppositeOfAnObservationPredicate(EndedByOwner, StatechartNode):
    """
    A node that observes True while another node does not observe True.
    """

    watched_node: StatechartNode = None
    """
    The node whose observation this node contradicts.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.logic_not(sm.Scalar(self.watched_node.observes_true))
        )


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingAFixedValue(EndedByOwner, StatechartNode):
    """
    A node that observes the same value on every tick and leaves ending it to its
    owner.
    """

    observation: ObservationStateValues
    """
    What this node observes on every tick.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(float(self.observation)))


@dataclass(eq=False, repr=False, kw_only=True)
class NodeFailingOnObservingFalse(FailsOnObservingFalse, NodeObservingAFixedValue):
    """
    A node that fails itself once it observes False and leaves succeeding to its owner.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeSucceedingOnObservingTrue(SucceedsOnObservingTrue, NodeObservingAFixedValue):
    """
    A node that succeeds once it observes True and declares no failure of its own.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeDeclaringNoWayToSucceed(StatechartNode):
    """
    A node class that leaves open how it succeeds.
    """


@dataclass(repr=False, eq=False, kw_only=True)
class NodeDeclaringItsOwnFailure(EndedByOwner, CompositeNode):
    """
    A node short of its goal that declares it cannot continue, which is what a node may
    decide about itself where succeeding is left to its owner.

    Being a composite node is what gives it an :meth:`expand` hook to declare
    the failure in; it has no children of its own.
    """

    def expand(self) -> None:
        self.fail_condition = sm.Scalar.const_true()

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


# %% goals that end their child


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCuttingOffItsChildAtItsGoal(EndedByOwner, CompositeNode):
    """
    Composite node whose child has reached its goal but is never ended on its
    own terms, so the child is only ever taken down by this node ending.
    """

    child: ConstTrueNode = expanded_child_field()
    """
    The child that sits at its goal until it is cut off.
    """

    def expand(self) -> None:
        self.child = ConstTrueNode()
        self._add_child_to_statechart(self.child)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCuttingOffItsChild(EndedByOwner, CompositeNode):
    """
    Composite node whose child is short of its goal and is never ended on its
    own terms, so the child is only ever taken down by this node ending.
    """

    child: ConstFalseNode = expanded_child_field()
    """
    The child that keeps running until it is cut off.
    """

    def expand(self) -> None:
        self.child = ConstFalseNode()
        self._add_child_to_statechart(self.child)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChildInterruptedBySibling(EndedByOwner, CompositeNode):
    """
    Composite node whose child is interrupted by a sibling on the first tick,
    so that a caller ending this node on that same tick makes the two ways of being
    interrupted compete.
    """

    trigger: ConstTrueNode = expanded_child_field()
    """
    Turns true on the first tick, which is what interrupts the child.
    """

    child: ConstFalseNode = expanded_child_field()
    """
    The child that is interrupted while its observation is false.
    """

    def expand(self) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_statechart(nodes=[self.trigger, self.child])
        self.child.interrupt_condition = self.trigger.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChildFailingOnItsOwn(EndedByOwner, CompositeNode):
    """
    Composite node whose child declares its own failure on the first tick, so
    that a caller ending this node on that same tick makes the child's own outcome
    compete with being cut off.
    """

    trigger: ConstTrueNode = expanded_child_field()
    """
    Turns true on the first tick, which is what makes the child declare its failure.
    """

    child: ConstFalseNode = expanded_child_field()
    """
    The child that gives up while its observation is false.
    """

    def expand(self) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_statechart(nodes=[self.trigger, self.child])
        self.child.fail_condition = self.trigger.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChildSucceedingOnItsOwn(EndedByOwner, CompositeNode):
    """
    Composite node whose child declares its own success on the first tick, so
    that a caller ending this node on that same tick makes the child's own outcome
    compete with being cut off.
    """

    trigger: ConstTrueNode = expanded_child_field()
    """
    Turns true on the first tick, which is what makes the child declare its success.
    """

    child: ConstFalseNode = expanded_child_field()
    """
    The child that declares its success while its observation is false.
    """

    def expand(self) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self._add_children_to_statechart(nodes=[self.trigger, self.child])
        self.child.success_condition = self.trigger.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithChildStartingLate(EndedByOwner, CompositeNode):
    """
    Composite node whose child waits for a delay before it starts, so the
    child's start is decided while this node is already running and its ending
    conditions have a settled value.
    """

    delay_in_ticks: int = 2
    """
    How many ticks pass before the child's start condition turns true.
    """

    child: ConstFalseNode = expanded_child_field()
    """
    The child whose start is being observed.
    """

    def expand(self) -> None:
        delay = CountTicks(ticks=self.delay_in_ticks)
        self.child = ConstFalseNode()
        self._add_children_to_statechart(nodes=[delay, self.child])
        self.child.start_condition = delay.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCuttingOffItsUndecidedChild(EndedByOwner, CompositeNode):
    """
    Composite node whose child never decides what it observes, so this node
    ending is the only thing that ever ends it.
    """

    child: NodeObservingNothingYet = expanded_child_field()
    """
    The child that observes nothing until it is ended.
    """

    def expand(self) -> None:
        self.child = NodeObservingNothingYet()
        self._add_child_to_statechart(self.child)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeCuttingOffItsGrandchild(EndedByOwner, CompositeNode):
    """
    Composite node holding another composite node, so that ending
    it reaches a node more than one level below it.
    """

    inner_node: CompositeNodeCuttingOffItsChild = expanded_child_field()
    """
    The node between this one and the grandchild.
    """

    def expand(self) -> None:
        self.inner_node = CompositeNodeCuttingOffItsChild()
        self._add_child_to_statechart(self.inner_node)

    @property
    def grandchild(self) -> ConstFalseNode:
        """
        :return: The node two levels below this node, which is short of its goal until
            this node ends.
        """
        return self.inner_node.child

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


# %% nodes that record what the statechart does to them


class LifeCycleCallback(Enum):
    """
    A callback the statechart runs on a node when its life cycle state changes.
    """

    START = auto()
    """
    :meth:`~cramph.node.StatechartNode.on_start`.
    """

    PAUSE = auto()
    """
    :meth:`~cramph.node.StatechartNode.on_pause`.
    """

    UNPAUSE = auto()
    """
    :meth:`~cramph.node.StatechartNode.on_unpause`.
    """

    END = auto()
    """
    :meth:`~cramph.node.StatechartNode.on_end`.
    """

    RESET = auto()
    """
    :meth:`~cramph.node.StatechartNode.on_reset`.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeRecordingItsCallbacks(EndedByOwner, StatechartNode):
    """
    A node that never reaches its goal and records every life cycle callback run on it.
    """

    callbacks: list[LifeCycleCallback] = field(default_factory=list, init=False)
    """
    The callbacks run on this node since :meth:`take_callbacks` was last called, in the
    order they ran.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())

    def take_callbacks(self) -> list[LifeCycleCallback]:
        """
        :return: The callbacks recorded so far, which are forgotten afterwards.
        """
        callbacks = self.callbacks
        self.callbacks = []
        return callbacks

    def on_start(self) -> None:
        self.callbacks.append(LifeCycleCallback.START)

    def on_pause(self) -> None:
        self.callbacks.append(LifeCycleCallback.PAUSE)

    def on_unpause(self) -> None:
        self.callbacks.append(LifeCycleCallback.UNPAUSE)

    def on_end(self) -> None:
        self.callbacks.append(LifeCycleCallback.END)

    def on_reset(self) -> None:
        self.callbacks.append(LifeCycleCallback.RESET)


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingTrueOnlyOnTick(EndedByOwner, StatechartNode):
    """
    A node whose observation expression is False but whose
    :meth:`~cramph.node.StatechartNode.on_tick` overrides
    it with True, counting how often it is ticked.
    """

    on_tick_calls: int = field(default=0, init=False)
    """
    How often :meth:`on_tick` was called.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())

    def on_tick(self) -> ObservationStateValues:
        self.on_tick_calls += 1
        return ObservationStateValues.TRUE


@dataclass(eq=False, repr=False, kw_only=True)
class NodeWritingAVariableOnStart(EndedByOwner, StatechartNode):
    """
    A node that sets a float variable to True when it starts, so what its start callback
    wrote can be observed by another node.
    """

    variable: FloatVariable = field(init=False)
    """
    The variable written when this node starts, False until then.
    """

    def set_up(self) -> None:
        self.variable = FloatVariable(f"{self.name}/written_on_start")
        self.context.float_variable_data.register_expression(self.variable)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_trinary_unknown())

    def on_start(self) -> None:
        self.context.float_variable_data.set_value(
            self.variable, float(ObservationStateValues.TRUE)
        )


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingAWrittenVariable(EndedByOwner, StatechartNode):
    """
    A node that observes True once another node's start callback wrote its variable.
    """

    writer: NodeWritingAVariableOnStart
    """
    The node whose written variable this node observes.
    """

    @property
    def prerequisite_nodes(self) -> list[StatechartNode]:
        """
        :return: :attr:`writer`, which creates the variable this node reads.
        """
        return [self.writer]

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.writer.variable,
                float(ObservationStateValues.TRUE),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeObservingItsSecondChildRun(EndedByOwner, CompositeNode):
    """
    Composite node that observes True once its second child is running, which
    starts once its first child succeeded, so the second child is started and then cut off
    by this node ending.
    """

    first: ConstTrueNode = expanded_child_field()
    """
    The child that succeeds as soon as it observes its goal.
    """

    second: NodeRecordingItsCallbacks = expanded_child_field()
    """
    The child that starts once :attr:`first` succeeded and is cut off by this node.
    """

    def expand(self) -> None:
        self.first = ConstTrueNode()
        self.second = NodeRecordingItsCallbacks()
        self._add_children_to_statechart(nodes=[self.first, self.second])
        self.first.success_condition = self.first.observes_true
        self.second.start_condition = self.first.is_succeeded

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.second.life_cycle_variable,
                int(LifeCycleValues.RUNNING),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeWithARecordingChild(EndedByOwner, CompositeNode):
    """
    Composite node holding one child that records its callbacks and would start
    whenever it may.
    """

    child: NodeRecordingItsCallbacks = expanded_child_field()
    """
    The child whose callbacks are recorded.
    """

    def expand(self) -> None:
        self.child = NodeRecordingItsCallbacks()
        self._add_child_to_statechart(self.child)


@dataclass(repr=False, eq=False, kw_only=True)
class CompositeNodeObservingItsCancellingChildRun(EndedByOwner, CompositeNode):
    """
    Composite node that observes True once its :class:`CancelStatechart` child is
    running, which starts once its other child observes True, so the cancelling node is
    started and then cut off by this node ending.
    """

    trigger: ConstTrueNode = expanded_child_field()
    """
    The child whose observation starts :attr:`cancel`.
    """

    cancel: CancelStatechart = expanded_child_field()
    """
    The child that is cut off right after starting.
    """

    def expand(self) -> None:
        self.trigger = ConstTrueNode()
        self.cancel = CancelStatechart(
            exception=NodeAssertionError(reason="cancelled right after starting")
        )
        self._add_children_to_statechart(nodes=[self.trigger, self.cancel])
        self.cancel.start_condition = self.trigger.observes_true

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.if_eq(
                self.cancel.life_cycle_variable,
                int(LifeCycleValues.RUNNING),
                sm.Scalar.const_true(),
                sm.Scalar.const_false(),
            )
        )
