from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    Goal,
    NodeArtifacts,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.data_types.exceptions import GiskardException


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
class ConstTrueNode(MotionStatechartNode):
    """
    A node that has always reached its goal, so ending it always succeeds it.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class ConstFalseNode(MotionStatechartNode):
    """
    A node that never reaches its goal, so ending it always fails it.
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
class TestGoal(Goal):
    sub_node1: ConstTrueNode = field(init=False)
    sub_node2: ConstTrueNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.sub_node1 = ConstTrueNode(name="sub muh1")
        self.add_node(self.sub_node1)
        self.sub_node2 = ConstTrueNode(name="sub muh2")
        self.add_node(self.sub_node2)
        self.sub_node1.end_condition = self.sub_node1.observation_variable
        self.sub_node2.start_condition = self.sub_node1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.sub_node2.observation_variable)


@dataclass(repr=False, eq=False)
class TestNestedGoal(Goal):
    sub_node1: TestGoal = field(init=False)
    sub_node2: TestGoal = field(init=False)
    inner: TestGoal = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.inner = TestGoal(name="inner")
        self.add_node(self.inner)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.inner.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStop(Goal):
    """
    Goal that tests if a child node runs after the parent node has stopped.

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

        self.add_nodes(
            nodes=[
                self.ticking1,
                self.ticking2,
                self.cancel,
            ]
        )
        self.cancel.start_condition = self.ticking1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking2.observation_variable))


@dataclass(repr=False, eq=False)
class TestEndBeforeStart(Goal):
    """
    Test if a child node can end before it was started.

    node1 waits 1 tick, then starts node 3. node2 fulfills the end condition of node 3
    immediately. node3 should start when node1 is True and transition to RUNNING with
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

        self.add_nodes(nodes=[self.node1, self.node2, self.node3])

        self.node3.start_condition = self.node1.observation_variable
        self.node3.end_condition = self.node2.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.node3.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStopFromPause(Goal):
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

        self.add_nodes(
            nodes=[self.ticking1, self.ticking2, self.ticking3, self.cancel, self.pulse]
        )
        self.pulse.start_condition = self.ticking3.observation_variable
        self.ticking2.pause_condition = self.pulse.observation_variable
        self.cancel.start_condition = self.ticking2.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking1.observation_variable))


@dataclass(repr=False, eq=False)
class TestUnpauseUnknownFromParentPause(Goal):
    """
    Tests if a child node can transition from PAUSED back to RUNNING when
    child.pause_condition is UNKNOWN.

    Child was paused by parent node being paused and child.pause_condition is UNKNOWN.
    When parent unpauses, child should transition back to RUNNING.
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

        self.add_node(self.count_ticks1)
        self.add_node(Sequence(nodes=[self.count_ticks2, self.cancel]))

        self.count_ticks1.pause_condition = sm.Scalar.const_trinary_unknown()
        self.count_ticks1.end_condition = self.count_ticks1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :attr:`count_ticks1` ends itself once it has counted, so its verdict is what
        still answers for it afterwards.
        """
        return NodeArtifacts(observation=sm.Scalar(self.count_ticks1.goal_reached))


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
class NodeObservingGoalReached(MotionStatechartNode):
    """
    A node whose observation reads whether another node reached its goal.
    """

    watched_node: MotionStatechartNode = field(default=None, kw_only=True)
    """
    The node whose goal this node observes.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.watched_node.goal_reached))


# %% goals that end their child


@dataclass(repr=False, eq=False)
class GoalCuttingOffItsChildAtItsGoal(Goal):
    """
    Goal whose child has reached its goal but is never ended on its own terms, so the
    child is only ever taken down by this goal ending.
    """

    child: ConstTrueNode = field(init=False)
    """
    The child that sits at its goal until it is cut off.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = ConstTrueNode()
        self.add_node(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class GoalCuttingOffItsChild(Goal):
    """
    Goal whose child is short of its goal and is never ended on its own terms, so the
    child is only ever taken down by this goal ending.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that keeps running until it is cut off.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = ConstFalseNode()
        self.add_node(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class GoalWithChildFailingOnItsOwn(Goal):
    """
    Goal whose child is ended on its own terms on the first tick, so that a caller
    ending this goal on that same tick makes the child's own verdict compete with being
    cut off.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    Turns true on the first tick, which is what ends the child.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that is ended while its observation is false.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self.add_nodes(nodes=[self.trigger, self.child])
        self.child.end_condition = self.trigger.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class GoalWithChildStartingLate(Goal):
    """
    Goal whose child waits for a delay before it starts, so the child's start is decided
    while this goal is already running and its end condition has a settled value.
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
        self.add_nodes(nodes=[delay, self.child])
        self.child.start_condition = delay.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False)
class GoalCuttingOffItsUndecidedChild(Goal):
    """
    Goal whose child never decides what it observes, so this goal ending is the only
    thing that ever ends it.
    """

    child: NodeObservingNothingYet = field(init=False)
    """
    The child that observes nothing until it is ended.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = NodeObservingNothingYet()
        self.add_node(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class GoalCuttingOffItsGrandchild(Goal):
    """
    Goal holding another goal, so that ending it reaches a node more than one level
    below it.
    """

    inner_goal: GoalCuttingOffItsChild = field(init=False)
    """
    The goal between this one and the grandchild.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.inner_goal = GoalCuttingOffItsChild()
        self.add_node(self.inner_goal)

    @property
    def grandchild(self) -> ConstFalseNode:
        """
        :return: The node two levels below this goal, which is short of its goal until
            this goal ends.
        """
        return self.inner_goal.child

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())
