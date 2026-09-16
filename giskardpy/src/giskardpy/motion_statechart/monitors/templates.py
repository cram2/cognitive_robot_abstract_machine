from __future__ import division

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCyclePredicate
from giskardpy.motion_statechart.graph_node import (
    CompositeStatechartNode,
    MaintenanceNode,
    MotionStatechartNode,
    NodeArtifacts,
    SelfFailingNode,
)
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    trinary_if_cases,
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
    logic_or,
    logic_not,
)


@dataclass(repr=False, eq=False)
class MonitoredCompositeStatechartNode(MaintenanceNode, CompositeStatechartNode, ABC):
    """
    Runs a monitored node next to the monitor observing it.

    What it observes is what the monitored node has reached, so nothing here ever
    concludes either: a plan step built from one is an attempt wrapping it.

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
        self._add_child_to_motion_statechart(self.monitor)
        self._add_child_to_motion_statechart(self.monitored_node)
        self.wire_monitor()

    @abstractmethod
    def wire_monitor(self) -> None:
        """
        Connect the monitor's observation to the monitored node's life cycle.
        """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        The monitored node is read through its last observation, which outlasts it,
        because a node that ended observes nothing any more.
        """
        return NodeArtifacts(observation=Scalar(self.monitored_node.last_observation))


@dataclass(repr=False, eq=False)
class PausedWhileTrue(MonitoredCompositeStatechartNode):
    """
    Holds the monitored node for as long as the monitor observes True, and lets it
    continue once the monitor turns False again.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.pause_condition = logic_or(
            self.monitor.observes_true,
            self.monitored_node.pause_condition,
        )


@dataclass(repr=False, eq=False)
class PausedUntilTrue(MonitoredCompositeStatechartNode):
    """
    Holds the monitored node until the monitor observes True, and lets it continue from
    then on.

    A monitor that has not observed anything yet has not turned True either, so it holds
    the monitored node as well.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.pause_condition = logic_or(
            self.monitored_node.pause_condition,
            logic_not(self.monitor.observes_true),
        )


@dataclass(repr=False, eq=False)
class StoppedWhenTrue(SelfFailingNode, MonitoredCompositeStatechartNode):
    """
    Interrupts the monitored node as soon as the monitor observes True.

    It observes True while the monitored node observes True or once it succeeded, False
    once the monitor stopped it, whatever it observed, and Unknown otherwise. Observing
    False is also what it declares its own failure on: the monitored node is down by
    then, so nothing is being held any more, and whoever runs this would otherwise wait
    for a subtree that can no longer arrive.

    The monitor is read through its last observation, which outlasts a monitor that ends
    itself on firing, unlike the pausing goals, which need the reading it takes right
    now.
    """

    def wire_monitor(self) -> None:
        self.monitored_node.interrupt_condition = logic_or(
            self.monitored_node.interrupt_condition,
            self.monitor.last_observed_true,
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        The monitored node's observation counts only while it has not ended; after that,
        only its success does.

        An observation expression reads the observation a node took on the previous
        control cycle, so a node the monitor stopped is told apart from one that
        succeeded by its life cycle rather than by that observation.
        """
        return NodeArtifacts(
            observation=trinary_if_cases(
                [
                    (
                        self._monitored_node_observing_true_or_succeeded,
                        Scalar.const_true(),
                    ),
                    (self.monitor.last_observed_true, Scalar.const_false()),
                ],
                Scalar.const_trinary_unknown(),
            )
        )

    @property
    def _monitored_node_observing_true_or_succeeded(self) -> Scalar:
        """
        :return: True while the monitored node has not ended and observes True, and once
            it succeeded; false otherwise.
        """
        has_ended = LifeCyclePredicate.IS_TERMINATED.expression(
            self.monitored_node.life_cycle_variable
        )
        observing_true_while_running = trinary_logic_and(
            trinary_logic_not(has_ended),
            self.monitored_node.observes_true,
        )
        return trinary_logic_or(
            observing_true_while_running, self.monitored_node.is_succeeded
        )
