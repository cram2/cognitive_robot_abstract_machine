from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from pycram.datastructures.enums import Arms
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from pycram.validation.goal_validator import MultiJointPositionGoalValidator
    from pycram.language import LanguageNode
    from semantic_digital_twin.datastructures.definitions import StaticJointState
    from pycram.robot_plans.actions.composite.searching import SearchAction


@dataclass
class PlanFailure(DataclassException):
    """
    Base class for all exceptions that are related to plan errors.
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

    def __post_init__(self):
        self.message = f"All children of {self.language_node} failed"


@dataclass
class PerceptionObjectNotFound(PlanFailure):
    search_action: SearchAction

    def __post_init__(self):
        self.message = (
            f"Perception object not found in search action {self.search_action}"
        )


@dataclass
class RobotInCollision(PlanFailure):
    """Thrown when the robot is in collision with the environment."""


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

    def __post_init__(self):
        self.message = f"Configuration type: {self.configuration_type.name} not reached"


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

    def __post_init__(self):
        self.message = f"Navigation goal not reached. Current pose: {self.current_pose}, goal pose: {self.goal_pose}"


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

    def __post_init__(self):
        self.message = f"Body {self.body} not fetchable from arm {self.arm}"
