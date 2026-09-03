from __future__ import division

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    if_cases,
    trinary_logic_not,
    trinary_logic_or,
)


@dataclass(repr=False, eq=False)
class MonitoredGoal(Goal, ABC):
    """
    Runs a monitored node next to the monitor observing it.

    The two are siblings, which is what lets the monitor's observation drive the
    monitored node's life cycle: a transition condition may only reference the owning
    node or a sibling of it. Neither node is chained to the other, so the monitor
    observes from the moment this goal starts.
    """

    monitor: MotionStatechartNode = field(kw_only=True)
    """
    The node whose observation controls the monitored node.
    """

    monitored_node: Optional[MotionStatechartNode] = field(default=None, kw_only=True)
    """
    The node placed under the monitor's control.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.add_node(self.monitor)
        self.add_node(self.monitored_node)
        self.wire_monitor()

    @abstractmethod
    def wire_monitor(self) -> None:
        """
        Connect the monitor's observation to the monitored node's life cycle.
        """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.monitored_node.observation_variable)


@dataclass(repr=False, eq=False)
class PausedWhileTrue(MonitoredGoal):
    """
    Holds the monitored node for as long as the monitor observes True, and lets it
    continue once the monitor turns False again.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.pause_condition = trinary_logic_or(self.monitor.observation_variable, self.monitored_node.pause_condition)


@dataclass(repr=False, eq=False)
class PausedUntilTrue(MonitoredGoal):
    """
    Holds the monitored node until the monitor observes True, and lets it continue from
    then on.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.pause_condition = trinary_logic_or(
            self.monitored_node.pause_condition,
            trinary_logic_not(self.monitor.observation_variable),
        )


@dataclass(repr=False, eq=False)
class StoppedWhenTrue(MonitoredGoal):
    """
    Ends the monitored node as soon as the monitor observes True.

    It observes True once the monitored node reached its goal, False once the monitor
    stopped it before that, and Unknown while the monitored node is still running.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.end_condition = trinary_logic_or(
            self.monitored_node.end_condition, self.monitor.observation_variable
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=if_cases(
                [
                    (
                        self.monitored_node.observation_variable == True,
                        Scalar.const_true(),
                    ),
                    (self.monitor.observation_variable == True, Scalar.const_false()),
                ],
                Scalar.const_trinary_unknown(),
            )
        )
