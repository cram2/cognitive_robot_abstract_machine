from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from coraplex.datastructures.enums import Arms
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from coraplex.validation.goal_validator import MultiJointPositionGoalValidator
    from coraplex.language import LanguageNode
    from semantic_digital_twin.datastructures.definitions import StaticJointState


@dataclass
class PlanFailure(DataclassException):
    """
    Base class for all exceptions that are related to plan errors.
    Can also be raised directly as a generic plan failure.
    """

    def error_message(self) -> str:
        return "Plan failed."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyUnderspecified(PlanFailure):
    """
    Raised when a plan is empty.
    """


@dataclass
class AllChildrenFailed(PlanFailure):
    """
    Thrown when all children of a plan node failed.
    """

    language_node: LanguageNode
    """
    The language node where all children failed.
    """

    def error_message(self) -> str:
        return f"All children of {self.language_node} failed"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class RobotInCollision(PlanFailure):
    """Thrown when the robot is in collision with the environment."""

    def error_message(self) -> str:
        return "The robot is in collision with the environment."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ConfigurationNotReached(PlanFailure):
    """"""

    goal_validator: MultiJointPositionGoalValidator
    """
    The goal validator that was used to check if the goal was reached.
    """
    configuration_type: StaticJointState
    """
    The configuration type that should be reached.
    """

    def error_message(self) -> str:
        return f"Configuration type: {self.configuration_type.name} not reached"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NavigationGoalNotReachedError(PlanFailure):
    """
    Thrown when the navigation goal is not reached.
    """

    current_pose: Pose
    """
    The current pose of the robot.
    """
    goal_pose: Pose
    """
    The goal pose of the robot.
    """

    def error_message(self) -> str:
        return f"Navigation goal not reached. Current pose: {self.current_pose}, goal pose: {self.goal_pose}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class BodyUnfetchable(PlanFailure):
    """
    Raised when a body cannot be fetched from an arm.
    """

    body: Body
    """
    The body that cannot be fetched.
    """

    arm: Arms
    """
    The arm from which the body cannot be fetched.
    """

    def error_message(self) -> str:
        return f"Body {self.body} not fetchable from arm {self.arm}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EndEffectorDidNotReachTarget(PlanFailure):
    """
    Raised when an end effector did not reach its target during a motion
    """

    end_effector: EndEffector
    """
    The end effector that did not reach its target.
    """

    target: Pose
    """
    The target pose that the end effector did not reach.
    """

    def error_message(self) -> str:
        return (
            f"EndEffector {self.end_effector} did not reach target {self.target}"
        )

    def suggest_correction(self) -> str:
        return ""
