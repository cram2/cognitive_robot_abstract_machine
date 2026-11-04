from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from coraplex.plans.failures import PlanFailure

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.base import Action
    from cramph.node import StatechartNode
    from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
    from semantic_digital_twin.world_description.world_entity import (
        SemanticAnnotation,
    )


# %% plan execution
@dataclass
class PlanNotCompiled(DataclassException):
    """
    Raised when a plan executor is asked to execute before it compiled a plan.
    """

    def error_message(self) -> str:
        return "No plan was compiled to execute."

    def suggest_correction(self) -> str:
        return "call compile with the plan before calling execute."


@dataclass
class DebuggingWithoutRosNode(DataclassException):
    """
    A plan executor is created in debug mode without a ROS node to show the worlds its
    candidates are tried in.
    """

    def error_message(self) -> str:
        return "Debug mode is selected without a ROS node to publish through."

    def suggest_correction(self) -> str:
        return "Pass a ROS node to the executor, or leave debug mode off."


@dataclass
class CannotMatchOnType(DataclassException):
    """
    Raised when a plan transformation is bound to a type that is neither a plan node nor
    a designator, leaving no rule by which it could select the nodes it rewrites.
    """

    transformation: type
    """
    The transformation class that carries the binding.
    """

    matched_type: type
    """
    The type it is bound to.
    """

    def error_message(self) -> str:
        return (
            f"{self.transformation.__name__} is bound to {self.matched_type}, which is "
            f"neither a plan node nor a designator."
        )

    def suggest_correction(self) -> str:
        return "bind the transformation to a plan node type or a designator type"


@dataclass
class ClosingGoalNotFound(DataclassException):
    """
    Raised when the goal closing the container is asked for, but no such goal lies below
    the close action's node.
    """

    plan_node: StatechartNode
    """
    The node of the close action.
    """

    def error_message(self) -> str:
        return f"{self.plan_node} has no goal closing its container below it."

    def suggest_correction(self) -> str:
        return "ask for the closing goal only once the close action has been expanded"


@dataclass
class ReachHasNoFinalApproach(DataclassException):
    """
    Raised when the final approach of a reach is asked for, but no tool center point
    motion lies below the reach's node.
    """

    plan_node: StatechartNode
    """
    The node of the reach.
    """

    def error_message(self) -> str:
        return f"{self.plan_node} has no tool center point motion below it."

    def suggest_correction(self) -> str:
        return "ask for the final approach only once the reach has been expanded"


@dataclass
class GraspHasNoClosingGoal(DataclassException):
    """
    Raised when the goal a grasp closes its gripper with is asked for, but no goal that
    shuts the gripper lies below the grasp's node.
    """

    plan_node: StatechartNode
    """
    The node of the grasp.
    """

    def error_message(self) -> str:
        return f"{self.plan_node} has no goal closing the gripper below it."

    def suggest_correction(self) -> str:
        return "ask for the closing goal only once the grasp has been expanded"


@dataclass
class ToolPathNotFound(DataclassException):
    """
    Raised when the goal moving a tool along its path is asked for, but no such goal
    lies below the tool action's node.
    """

    plan_node: StatechartNode
    """
    The node of the tool action.
    """

    def error_message(self) -> str:
        return f"{self.plan_node} has no goal moving its tool along a path below it."

    def suggest_correction(self) -> str:
        return "ask for the tool path only once the tool action has been expanded"


@dataclass
class MissingWaypoints(DataclassException):
    """
    Raised when a waypoint motion or tool action produced no waypoints to follow.
    """

    instance: Action
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

    instance: Action
    """
    The wiping action that has no target.
    """

    def error_message(self) -> str:
        return f"{self.instance} has neither a container nor a target pose."

    def suggest_correction(self) -> str:
        return "provide either a container body or a target pose to wipe."


@dataclass
class MissingToolFrame(DataclassException):
    """
    Raised when no tool frame is available for the requested arm.
    """

    arm: Arm
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
class MotionDidNotFinish(PlanFailure):
    """
    Raised when a plan ended without succeeding.
    """

    unfinished_motions: list[StatechartNode]
    """
    The nodes that did not succeed, whether they failed, were interrupted or never
    ended.
    """

    def error_message(self) -> str:
        reports = ", ".join(
            f"{motion.unique_name} ({motion.life_cycle_state.name})"
            for motion in self.unfinished_motions
        )
        return f"Motion did not finish, following motions did not succeed: {reports}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ObjectIsNotHeld(DataclassException):
    """
    Raised when a place is asked for an object that no arm holds and no pick-up before
    it is going to take.
    """

    object_designator: HasGraspCandidates
    """
    The object that was to be placed.
    """

    def error_message(self) -> str:
        return f"no arm holds {self.object_designator.name} to place it."

    def suggest_correction(self) -> str:
        return "place the object after a pick-up of it."


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

    semantic_annotation: type[SemanticAnnotation]
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


@dataclass
class NoFloorBelowRobot(DataclassException):
    """
    Raised when a robot that has to plan its way over a floor stands over none.
    """

    robot: AbstractRobot
    """
    The robot that stands over no floor.
    """

    def error_message(self) -> str:
        return f"'{self.robot.name}' does not stand over any annotated floor."

    def suggest_correction(self) -> str:
        return (
            "annotate the surface the robot drives on as a Floor, or move the robot "
            "onto one that is already annotated."
        )


@dataclass
class NotOnASingleLevelException(DataclassException):
    """
    Raised when an entity is detected to be on None or multiple levels at the same time.
    """

    message: str

    def error_message(self) -> str:
        return self.message

    def suggest_correction(self) -> str:
        return f"Move the robot to a recognized level"
