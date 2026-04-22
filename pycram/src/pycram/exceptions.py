from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Type, List

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.factories import ConditionType, get_false_statements
from krrood.exceptions import DataclassException
from pycram.plans.failures import PlanFailure

if TYPE_CHECKING:
    from pycram.plans.designator import Designator
    from pycram.robot_plans.actions.base import ActionDescription


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

    def __post_init__(self):
        self.message = (
            f"{self.instance} has no plan node. Did you forget to call `add_subplan` when creating"
            f"plans inside actions?"
        )
        super().__post_init__()


@dataclass
class ConditionNotSatisfied(DataclassException):

    pre_condition: bool
    action: Type[ActionDescription]
    condition: ConditionType

    def __post_init__(self):
        if isinstance(self.condition, bool):
            self.message = f"{"Pre" if self.pre_condition else "Post"}-Condition for Action '{self.action.__name__}' is not satisfied"
        else:
            false_statements = get_false_statements(self.condition)
            self.message = f"{"Pre" if self.pre_condition else "Post"}-Condition for Action '{self.action.__name__}' is not satisfied, following statements are false: {[s._name_ for s in false_statements]}"
        super().__post_init__()


@dataclass
class MotionDidNotFinish(PlanFailure):

    failed_motions: List[MotionStatechartNode]

    def __post_init__(self):
        self.message = (
            f"Motion did not finish, following motions failed: {self.failed_motions}"
        )
