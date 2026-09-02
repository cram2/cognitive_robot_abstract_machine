from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, TYPE_CHECKING, Type

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.factories import ConditionType, get_false_statements
from krrood.exceptions import DataclassException
from coraplex.datastructures.enums import Arms
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from coraplex.validation.goal_validator import MultiJointPositionGoalValidator
    from coraplex.language import LanguageNode
    from semantic_digital_twin.datastructures.definitions import StaticJointState
    from coraplex.datastructures.dataclasses import Context
    from coraplex.failure_handling.failure_handling_strategy import FailureResolution
    from coraplex.plans.plan_node import ActionNode, PlanNode
    from coraplex.robot_plans.actions.base import ActionDescription


@dataclass
class PlanFailure(DataclassException):
    """
    Base class for all exceptions that are related to plan errors.

    Can also be raised directly as a generic plan failure.
    """

    node: PlanNode
    """
    The node the failure occurred at, which gets first refusal at handling it.
    """

    refined_from: Optional[PlanFailure] = field(default=None, kw_only=True)
    """
    The failure this one was refined from by a
    :class:`~coraplex.failure_handling.failure_refiner.FailureDetector`, or None if
    this failure has not been refined.
    """

    resolution: Optional[FailureResolution] = field(default=None, kw_only=True)
    """
    The resolution the failure handler decided on for this failure, or None if it has
    not been handled yet.
    """

    def error_message(self) -> str:
        return "Plan failed."

    def suggest_correction(self) -> str:
        return ""

    @property
    def action_node(self) -> Optional[ActionNode]:
        """
        :return: The node itself if it is an :class:`ActionNode`, otherwise the nearest
            ancestor that is one, or None if there is no such ancestor.
        """
        from coraplex.plans.plan_node import ActionNode

        if isinstance(self.node, ActionNode):
            return self.node
        return self.node.parent_action_node

    @property
    def context(self) -> Context:
        """
        :return: The context of the plan in which this failure occurred.
        """
        return self.node.plan.context


@dataclass
class EmptyUnderspecified(PlanFailure):
    """
    Raised when an underspecified node has no action candidates left to try.
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
    """
    Thrown when the robot is in collision with the environment.
    """

    def error_message(self) -> str:
        return "The robot is in collision with the environment."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ConfigurationNotReached(PlanFailure):
    """
    Raised when a joint configuration the robot was sent to is not reached.
    """

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
class ConditionNotSatisfied(PlanFailure):
    """
    Raised when a pre- or post-condition of an action does not hold.
    """

    pre_condition: bool
    """
    Whether the unsatisfied condition is a pre-condition rather than a post-condition.
    """

    action: Type[ActionDescription]
    """
    The action whose condition is not satisfied.
    """

    condition: ConditionType
    """
    The condition that evaluated to False.
    """

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
    """
    Raised when a motion state chart aborts before every motion reached its goal.
    """

    failed_motions: List[MotionStatechartNode]
    """
    The motion state chart nodes that did not reach their goal.
    """

    def error_message(self) -> str:
        return f"Motion did not finish, following motions failed: {self.failed_motions}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EndEffectorDidNotReachTarget(PlanFailure):
    """
    Raised when an end effector did not reach its target during a motion.
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
        return f"EndEffector {self.end_effector} did not reach target {self.target}"

    def suggest_correction(self) -> str:
        return ""
