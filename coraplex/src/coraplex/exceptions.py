from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Type, List

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.factories import ConditionType, get_false_statements
from krrood.exceptions import DataclassException
from coraplex.plans.failures import PlanFailure

if TYPE_CHECKING:
    from coraplex.plans.designator import Designator
    from coraplex.robot_plans.actions.base import ActionDescription


@dataclass
class ContextIsUnavailable(DataclassException):
    """
    Raised when an instance that tries to access the context of a plan has no reference to the plan.

    Most likely raised when an action created a subplan without calling `ActionDescription.add_subplan`
    """

    instance: Designator
    """
    The instance where the plan node is None.
    """

    def error_message(self) -> str:
        return f"{self.instance} has no plan node."

    def suggest_correction(self) -> str:
        return (
            "did you forget to call `add_subplan` when creating plans inside actions?"
        )


@dataclass
class ConditionNotSatisfied(PlanFailure):

    pre_condition: bool
    action: Type[ActionDescription]
    condition: ConditionType

    def error_message(self) -> str:
        prefix = "Pre" if self.pre_condition else "Post"
        if isinstance(self.condition, bool):
            return f"{prefix}-Condition for Action '{self.action.__name__}' is not satisfied"
        false_statements = get_false_statements(self.condition)
        return f"{prefix}-Condition for Action '{self.action.__name__}' is not satisfied, following statements are false: {[s._name_ for s in false_statements]}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MotionDidNotFinish(PlanFailure):

    failed_motions: List[MotionStatechartNode]

    def error_message(self) -> str:
        return f"Motion did not finish, following motions failed: {self.failed_motions}"

    def suggest_correction(self) -> str:
        return ""
