from __future__ import annotations
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Callable

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from krrood.entity_query_language.factories import ConditionType, evaluate_condition
from coraplex.plans.plan_node import PlanNode, ActionNode


@dataclass
class ConditionNode(PlanNode):
    """
    Node representing a pre or post condition of an action.
    """

    condition: ConditionType = field(kw_only=True)
    """
    The EQL condition to be evaluated.
    """

    pre_condition: bool = field(kw_only=True)
    """
    If this is a pre or post condition.
    """

    action_node: ActionNode = field(kw_only=True)
    """
    The action node where this condition belongs to.

    Needed to raise the correct exception in case the condition is not satisfied.
    """

    def notify(self):
        pass

    def redirect_node_reference(
        self, replaced_node: PlanNode, replacement_node: ActionNode
    ) -> None:
        if self.action_node is replaced_node:
            self.action_node = replacement_node


def condition_monitor(condition_node: ConditionNode) -> ThreadedPredicateMonitor:
    """
    Build a giskard monitor that evaluates a PyCRAM condition inside a motion state
    chart.

    The EQL condition is wrapped in a plain callable, so giskard never sees any
    PyCRAM/EQL types. The condition is evaluated in a background thread (see
    :class:`~giskardpy.motion_statechart.monitors.payload_monitors.ThreadedPredicateMonitor`),
    its observation state becoming TRUE/FALSE once evaluation finishes.

    :param condition_node: The pre- or post-condition node to evaluate.
    :return: A monitor whose observation reflects the condition's truth value.
    """
    name = "pre_condition" if condition_node.pre_condition else "post_condition"
    return ThreadedPredicateMonitor(
        predicate=lambda: bool(evaluate_condition(condition_node.condition)),
        name=name,
    )


@dataclass(eq=False, repr=False)
class PlanNodeStatusMonitor(MotionStatechartNode):
    """
    A motion-statechart monitor whose observation reflects a boolean predicate over a
    PyCRAM :class:`~pycram.plans.plan_node.PlanNode`'s status (e.g. whether the node is
    interrupted or paused).

    Unlike :class:`~giskardpy.motion_statechart.monitors.payload_monitors.ThreadedPredicateMonitor`,
    the predicate is evaluated synchronously on **every** tick, so the observation
    tracks status changes during execution (needed for pause/resume, where a node
    can be paused and later resumed).
    """

    predicate: Callable[[], bool] = field(kw_only=True)
    """
    The predicate over the originating plan node's status.
    """

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        return (
            ObservationStateValues.TRUE
            if self.predicate()
            else ObservationStateValues.FALSE
        )
