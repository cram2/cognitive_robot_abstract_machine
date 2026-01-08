from dataclasses import dataclass, field
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.motion_statechart.context import ExecutionContext, BuildContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    Goal,
    NodeArtifacts,
    CancelMotion,
    EndMotion,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountTicks, Pulse


@dataclass(eq=False, repr=False)
class ConstTrueNode(MotionStatechartNode):
    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class ConstFalseNode(MotionStatechartNode):
    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False)
class ChangeStateOnEvents(MotionStatechartNode):
    state: Optional[str] = None

    def on_start(self, context: ExecutionContext):
        self.state = "on_start"

    def on_pause(self, context: ExecutionContext):
        self.state = "on_pause"

    def on_unpause(self, context: ExecutionContext):
        self.state = "on_unpause"

    def on_end(self, context: ExecutionContext):
        self.state = "on_end"

    def on_reset(self, context: ExecutionContext):
        self.state = "on_reset"


@dataclass(repr=False, eq=False)
class TestGoal(Goal):
    sub_node1: ConstTrueNode = field(init=False)
    sub_node2: ConstTrueNode = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.sub_node1 = ConstTrueNode(name="sub muh1")
        self.add_node(self.sub_node1)
        self.sub_node2 = ConstTrueNode(name="sub muh2")
        self.add_node(self.sub_node2)
        self.sub_node1.end_condition = self.sub_node1.observation_variable
        self.sub_node2.start_condition = self.sub_node1.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.sub_node2.observation_variable)


@dataclass(repr=False, eq=False)
class TestNestedGoal(Goal):
    sub_node1: TestGoal = field(init=False)
    sub_node2: TestGoal = field(init=False)
    inner: TestGoal = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.inner = TestGoal(name="inner")
        self.add_node(self.inner)

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.inner.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStop(Goal):
    """
    Goal that tests if a child node runs after the parent node has stopped.
    Uses a CancelMotion node to raise an exception if the child node runs after the parent has stopped.
    """

    ticking1: CountTicks = field(init=False)
    ticking2: CountTicks = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.ticking1 = CountTicks(name="3ticks", ticks=3)
        self.ticking2 = CountTicks(name="2ticks", ticks=2)
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=GiskardException("Node ticked after template stopped"),
        )

        self.add_nodes(
            nodes=[
                self.ticking1,
                self.ticking2,
                self.cancel,
            ]
        )
        self.cancel.start_condition = self.ticking1.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking2.observation_variable))


@dataclass(repr=False, eq=False)
class TestEndBeforeStart(Goal):
    """
    Test if a child node can end before it was started.
    node1 waits 1 tick, then starts node 3.
    node2 fulfills the end condition of node 3 immediately.
    node3 should start when node1 is True and transition to RUNNING with Observationstate UNKNOWN.
    On the next tick, node3 should end because its end condition is already fulfilled by node2.
    """

    node1: CountTicks = field(init=False)
    node2: ConstTrueNode = field(init=False)
    node3: ConstTrueNode = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.node1 = CountTicks(ticks=1)
        self.node2 = ConstTrueNode()
        self.node3 = ConstTrueNode()

        self.add_nodes(nodes=[self.node1, self.node2, self.node3])

        self.node3.start_condition = self.node1.observation_variable
        self.node3.end_condition = self.node2.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.node3.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStopFromPause(Goal):
    """
    Test if child node can transition to RUNNING from PAUSED after parent node is DONE.
    Uses a CancelMotion node to raise an exception if the child node runs after the parent has stopped.
    """

    ticking1: CountTicks = field(init=False)
    ticking2: CountTicks = field(init=False)
    ticking3: CountTicks = field(init=False)
    pulse: Pulse = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.ticking1 = CountTicks(name="3ticks", ticks=3)
        self.ticking2 = CountTicks(name="trigger_cancel_after_unpause", ticks=4)
        self.ticking3 = CountTicks(name="2ticks", ticks=2)
        self.pulse = Pulse()
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=GiskardException("Node ticked after template stopped"),
        )

        self.add_nodes(
            nodes=[self.ticking1, self.ticking2, self.ticking3, self.cancel, self.pulse]
        )
        self.pulse.start_condition = self.ticking3.observation_variable
        self.ticking2.pause_condition = self.pulse.observation_variable
        self.cancel.start_condition = self.ticking2.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking1.observation_variable))


@dataclass(repr=False, eq=False)
class TestUnpauseUnknownFromParentPause(Goal):
    """
    Tests if a child node can transition from PAUSED back to RUNNING when child.pause_condition is UNKNOWN.
    Child was paused by parent node being paused and child.pause_condition is UNKNOWN.
    When parent unpauses, child should transition back to RUNNING.
    """

    count_ticks1: CountTicks = field(init=False)
    count_ticks2: CountTicks = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.count_ticks1 = CountTicks(ticks=2)
        self.count_ticks2 = CountTicks(ticks=5)
        self.cancel = CancelMotion(
            name="check_unpause_failed",
            exception=GiskardException("Node did not unpause correctly"),
        )

        self.add_node(self.count_ticks1)
        self.add_node(Sequence(nodes=[self.count_ticks2, self.cancel]))

        self.count_ticks1.pause_condition = sm.Scalar.const_trinary_unknown()
        self.count_ticks1.end_condition = self.count_ticks1.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.Scalar(self.count_ticks1.observation_variable)
        )
