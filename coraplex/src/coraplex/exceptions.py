from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Type, List

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.factories import ConditionType, get_false_statements
from krrood.exceptions import DataclassException
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.failures import PlanFailure

if TYPE_CHECKING:
    from coraplex.plans.designator import Designator
    from coraplex.robot_plans.actions.base import ActionDescription
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


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
class TipLinkDoesNotMatchAnyArm(DataclassException):
    """
    Raised when a reachability validator's tip link is not the tool frame of any arm of the robot,
    so no arm can be selected to reach the requested pose.
    """

    tip_link: KinematicStructureEntity
    """
    The tip link that did not match any arm.
    """

    robot: AbstractRobot
    """
    The robot whose arms were searched.
    """

    def error_message(self) -> str:
        return f"tip_link {self.tip_link} does not match any arm of {self.robot}"

    def suggest_correction(self) -> str:
        return "ensure the tip_link is the tool frame of one of the robot's arms."


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


@dataclass
class UnknownExecutionType(DataclassException):
    """
    Raised when an executable is run with an execution type it does not handle.
    """

    execution_type: ExecutionType
    """
    The execution type that is not supported.
    """

    def error_message(self) -> str:
        return f"Unknown execution type: {self.execution_type}"

    def suggest_correction(self) -> str:
        return ""
