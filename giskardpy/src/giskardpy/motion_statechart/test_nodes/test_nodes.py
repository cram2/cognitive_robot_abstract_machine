from dataclasses import dataclass, field
from typing import Optional

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.motion_statechart.context import ExecutionContext, BuildContext
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
        return NodeArtifacts(observation=cas.TrinaryTrue)


@dataclass(eq=False, repr=False)
class ConstFalseNode(MotionStatechartNode):
    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryFalse)


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
        return NodeArtifacts(
            observation=cas.Expression(self.inner.observation_variable)
        )


@dataclass(repr=False, eq=False)
class TestRunAfterStop(Goal):
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
        return NodeArtifacts(
            observation=cas.Expression(self.ticking2.observation_variable)
        )


@dataclass(repr=False, eq=False)
class TestEndBeforeStart(Goal):
    node1: CountTicks = field(init=False)
    node2: ConstTrueNode = field(init=False)
    node3: ConstTrueNode = field(init=False)

    def expand(self, context: BuildContext) -> None:
        self.node1 = CountTicks(ticks=1)
        self.add_node(self.node1)

        self.node2 = ConstTrueNode()
        self.add_node(self.node2)

        self.node3 = ConstTrueNode()
        self.add_node(self.node3)
        self.node3.start_condition = self.node1.observation_variable
        self.node3.end_condition = self.node2.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=cas.Expression(self.node3.observation_variable)
        )


@dataclass(repr=False, eq=False)
class TestRunAfterStopFromPause(Goal):
    ticking1: CountTicks = field(init=False)
    ticking2: CountTicks = field(init=False)
    ticking3: CountTicks = field(init=False)
    pulse: Pulse = field(init=False)
    cancel: CancelMotion = field(init=False)
    constFalse: ConstFalseNode = field(init=False)

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
        return NodeArtifacts(
            observation=cas.Expression(self.ticking1.observation_variable)
        )
