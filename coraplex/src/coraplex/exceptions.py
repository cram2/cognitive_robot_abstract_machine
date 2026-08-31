from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Type, List

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.factories import ConditionType, get_false_statements
from krrood.exceptions import DataclassException
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.plans.failures import PlanFailure

if TYPE_CHECKING:
    from coraplex.plans.designator import Designator
    from coraplex.robot_plans.actions.base import ActionDescription
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
        SemanticAnnotation,
    )


@dataclass
class ContextIsUnavailable(DataclassException):
    """
    Raised when an instance that tries to access the context of a plan has no reference
    to the plan.

    Most likely raised when an action created a subplan without calling
    `ActionDescription.add_subplan`
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
    Raised when a reachability validator's tip link is not the tool frame of any arm of
    the robot, so no arm can be selected to reach the requested pose.
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
class MissingWaypoints(DataclassException):
    """
    Raised when a waypoint motion or tool action produced no waypoints to follow.
    """

    instance: Designator
    """
    The designator that has no waypoints.
    """

    def error_message(self) -> str:
        return f"{self.instance} has no waypoints to follow."

    def suggest_correction(self) -> str:
        return "ensure the motion sequence samples at least one point."


@dataclass
class WipingTargetMissing(DataclassException):
    """
    Raised when a wiping action is created without a surface to wipe.
    """

    instance: Designator
    """
    The wiping action that has no target.
    """

    def error_message(self) -> str:
        return f"{self.instance} has neither a container nor a target pose."

    def suggest_correction(self) -> str:
        return "provide either a container body or a target pose to wipe."


@dataclass
class PerceptionTargetMissing(DataclassException):
    """
    Raised when an action is asked to perceive before grasping but names no object.
    """

    instance: Designator
    """
    The action that has no object to detect.
    """

    def error_message(self) -> str:
        return f"{self.instance} perceives before grasping but names no object."

    def suggest_correction(self) -> str:
        return "provide an object_designator or leave perceive_before_grasp off."


@dataclass
class MissingToolFrame(DataclassException):
    """
    Raised when no tool frame is available for the requested arm.
    """

    arm: Arms
    """
    The arm whose tool frame was requested.
    """

    robot: AbstractRobot
    """
    The robot whose arm was searched.
    """

    def error_message(self) -> str:
        return f"no tool frame available for arm {self.arm} of {self.robot}"

    def suggest_correction(self) -> str:
        return "ensure the arm's end effector defines a tool frame."


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


@dataclass
class PerceptionException(DataclassException, ABC):
    """
    Represents a custom exception specific to perception-related errors.
    """


@dataclass
class PerceptionExceptionWithSemanticAnnotation(PerceptionException, ABC):
    """
    For PerceptionExceptions that name the annotation the perception was about.
    """

    semantic_annotation: Type[SemanticAnnotation]
    """
    The annotation the perception was about.
    """


@dataclass
class PerceivedObjectNotInWorld(PerceptionExceptionWithSemanticAnnotation):
    """
    Raised when a detection names an object the world does not hold, so there is nothing
    to write the perceived pose to.
    """

    def error_message(self) -> str:
        return (
            f"The world holds no {self.semantic_annotation.__name__} the perceived pose "
            f"could be written to."
        )

    def suggest_correction(self) -> str:
        return (
            "spawn the object before detecting it, and annotate it with the semantic "
            "annotation that was queried."
        )


@dataclass
class AmbiguousDetection(PerceptionExceptionWithSemanticAnnotation):
    """
    Raised when a detection's annotation describes several bodies, so the perceived pose
    cannot be assigned to one of them.
    """

    body_count: int
    """
    How many distinct bodies the annotation described.
    """

    def error_message(self) -> str:
        return (
            f"{self.semantic_annotation.__name__} describes {self.body_count} bodies in "
            f"the world."
        )

    def suggest_correction(self) -> str:
        return "narrow the query's semantic annotation so it names a single object."


@dataclass
class NothingDetected(PerceptionExceptionWithSemanticAnnotation):
    """
    Raised when a perception source answers a query without reporting any object.

    Treated as a failure rather than an empty answer: a plan that carried on would act on
    the pose the object was spawned with while believing perception had confirmed it.
    """

    def error_message(self) -> str:
        return f"The perception source reported no {self.semantic_annotation.__name__}."

    def suggest_correction(self) -> str:
        return (
            "check that the object is in view and that the pipeline's crop and plane "
            "parameters cover it."
        )


@dataclass
class UnidentifiedDetections(PerceptionExceptionWithSemanticAnnotation):
    """
    Raised when a perception source reports several candidates it cannot tell apart.

    A pipeline that localizes without classifying gives no way to choose between them,
    so the choice is refused rather than made arbitrarily.
    """

    candidate_count: int
    """
    How many indistinguishable candidates were reported.
    """

    def error_message(self) -> str:
        return (
            f"The perception source reported {self.candidate_count} candidates for "
            f"{self.semantic_annotation.__name__} and none of them carry a class label."
        )

    def suggest_correction(self) -> str:
        return (
            "add a classifying annotator to the perception pipeline, or narrow what is "
            "in view so a single object is reported."
        )


@dataclass
class PerceptionSourceUnavailable(PerceptionException):
    """
    Raised when the perception pipeline does not answer within the configured timeout.
    """

    action_name: str
    """
    The action the source was expected on.
    """

    def error_message(self) -> str:
        return f"No perception source is serving '{self.action_name}'."

    def suggest_correction(self) -> str:
        return "start the perception pipeline before running the plan."
