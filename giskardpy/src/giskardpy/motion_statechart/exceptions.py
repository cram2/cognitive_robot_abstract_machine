from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    ViolatedCollision,
)

if TYPE_CHECKING:
    from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
    from semantic_digital_twin.datastructures.definitions import GripperState
    from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )
from cramph.exceptions import StatechartError, NodeInitializationError


@dataclass
class CollisionViolatedError(DataclassException):
    """
    Raised when bodies came closer to each other than their collision threshold allows.
    """

    violated_collisions: list[ViolatedCollision]
    """
    Every collision that came closer than its violated distance.
    """

    def error_message(self) -> str:
        violations = "".join(
            f"{str(violation.closest_points.body_a.name), str(violation.closest_points.body_b.name)}: "
            f"{violation.closest_points.distance} < {violation.violated_distance}\n"
            for violation in self.violated_collisions
        )
        return f"Violated collision constraints: \n{violations}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MotionStatechartError(StatechartError, ABC):
    """
    Base class for errors in the motion statechart that concern motion control.
    """


@dataclass
class WorldStateArrayReplacedError(MotionStatechartError):
    """
    Raised when the world replaced its state array while a motion statechart was
    compiled against it.
    """

    compiled_degrees_of_freedom: int
    """
    Number of degrees of freedom the motion statechart was compiled against.
    """

    current_degrees_of_freedom: int
    """
    Number of degrees of freedom the world holds now.
    """

    def error_message(self) -> str:
        return (
            f"The world replaced its state array, which the compiled motion statechart "
            f"still reads through a memory view of the previous one. It was compiled "
            f"against {self.compiled_degrees_of_freedom} degrees of freedom and the "
            f"world now has {self.current_degrees_of_freedom}."
        )

    def suggest_correction(self) -> str:
        return (
            "Adding or removing a degree of freedom replaces the state array. Avoid "
            "such model changes while a motion is running, or re-compile afterwards. "
            "Re-parenting a branch preserves the degrees of freedom and is safe."
        )


@dataclass
class UnexpectedWorldEntityCountError(NodeInitializationError):
    """
    Raised when a node searches the world for entities and finds a different number than
    it can work with.
    """

    expected_count: int | str
    """
    The number of entities the node needs, either as a number or as a textual
    description of the accepted range.
    """

    actual_count: int
    """
    The number of matching entities that were found in the world.
    """

    entity_type: type | str | tuple[type, ...]
    """
    The type of entity that was searched for.
    """

    def error_message(self) -> str:
        return f"Expected {self.expected_count} entities of type {self.entity_type}, but found {self.actual_count}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyGoalStateError(NodeInitializationError):
    """
    Raised when a node is given a goal state that names no degree of freedom.
    """

    def error_message(self) -> str:
        return "Goal state is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyDegreesOfFreedomError(NodeInitializationError):
    """
    Raised when a node is explicitly given an empty list of degrees of freedom.
    """

    def error_message(self) -> str:
        return "Degrees of freedom list is empty."

    def suggest_correction(self) -> str:
        return "Pass at least one degree of freedom, or leave it None to use every active degree of freedom in the world."


@dataclass
class GoalPointsReferenceFrameMismatchError(NodeInitializationError):
    """
    Raised when the goal points of a node are expressed in more than one reference
    frame.
    """

    reference_frame_a: KinematicStructureEntity
    """
    The reference frame of the first goal point.
    """

    reference_frame_b: KinematicStructureEntity
    """
    The reference frame that differs from :attr:`reference_frame_a`.
    """

    def error_message(self) -> str:
        return f"All goal points must have the same reference frame, but got {self.reference_frame_a} and {self.reference_frame_b}."

    def suggest_correction(self) -> str:
        return "Make sure all goal points have the same reference frame."


@dataclass
class MissingErrorSignalError(NodeInitializationError):
    """
    Raised when a converging task builds artifacts that carry no error signal.
    """

    def error_message(self) -> str:
        return (
            f'Converging task "{self.node.unique_name}" built artifacts without an error '
            f"signal, so there is nothing to compare against its threshold."
        )

    def suggest_correction(self) -> str:
        return (
            "Set MotionNodeArtifacts.error in build_artifacts to the error the task's constraints "
            "drive to zero."
        )


@dataclass
class NoProgressError(MotionStatechartError):
    """
    Raised when the watched tasks stopped approaching their goal for too long.
    """

    progress_monitor: StillProgressing
    """
    The monitor that detected the stall and knows which tasks are affected.
    """

    def error_message(self) -> str:
        stalled_tasks = self.progress_monitor.stalled_tasks
        names = ", ".join(task.unique_name for task in stalled_tasks)
        return (
            f"{names or self.progress_monitor.monitored_node.unique_name} stopped "
            f"approaching a goal for {self.progress_monitor.timeout}."
        )

    def suggest_correction(self) -> str:
        return (
            "Check whether the goal is reachable, whether another task of equal or higher "
            "weight is opposing it, or whether the robot is at a joint limit."
        )


@dataclass
class InvalidConstraintExpressionShapeError(MotionStatechartError):
    """
    Raised when a constraint expression is not a scalar.
    """

    actual_shape: list[int]
    """
    The shape of the offending expression.
    """

    def error_message(self) -> str:
        shape_str = " ".join(map(str, self.actual_shape))
        return f"Constraint expression must have shape (1, 1), has ({shape_str})."

    def suggest_correction(self) -> str:
        return "Ensure the expression evaluates to a (1, 1) scalar."


@dataclass
class UnknownGriplinkActionError(MotionStatechartError):
    """
    Raised when a griplink action server task is built for a griplink action it does not
    build goals for.
    """

    message_type: type
    """
    The griplink action the task was built with.
    """

    def error_message(self) -> str:
        return f'Cannot build a goal for the griplink action "{self.message_type.__name__}".'

    def suggest_correction(self) -> str:
        return "Build the task with Grip, Release, Flexgrip or Flexrelease."


@dataclass
class NoGriplinkEndpoint(MotionStatechartError):
    """
    Raised when a gripper state has no griplink action server endpoint to command it on.
    """

    end_effector: EndEffector
    """
    The gripper whose state is to be commanded.
    """

    state_type: GripperState
    """
    The gripper state no griplink endpoint serves.
    """

    def error_message(self) -> str:
        return (
            f"The gripper '{self.end_effector.name}' has no griplink endpoint to "
            f"command the state {self.state_type.name}."
        )

    def suggest_correction(self) -> str:
        return (
            "Command a gripper state the gripper declares an endpoint for, or add an "
            "endpoint for this gripper and state."
        )


@dataclass
class GriplinkParameterOutOfRangeError(MotionStatechartError):
    """
    Raised when a griplink parameter lies outside the interval the gripper accepts.
    """

    parameter_name: str
    """
    The name of the offending parameter.
    """

    value: float
    """
    The value the parameter was given.
    """

    lower_bound: float
    """
    The smallest value the gripper accepts.
    """

    upper_bound: float
    """
    The largest value the gripper accepts.
    """

    def error_message(self) -> str:
        return (
            f"Griplink parameter {self.parameter_name} is {self.value}, outside the "
            f"valid interval [{self.lower_bound}..{self.upper_bound}]."
        )

    def suggest_correction(self) -> str:
        return f"Choose a {self.parameter_name} within the valid interval."


@dataclass
class ActionClientTypeMismatchError(MotionStatechartError):
    """
    Raised when an action topic is requested with a different message type than the one
    its cached action client was created with.
    """

    action_topic: str
    """
    The action topic that was requested with two different message types.
    """

    existing_message_type: type
    """
    The message type the cached action client for this topic was created with.
    """

    requested_message_type: type
    """
    The message type that was requested for this topic instead.
    """

    def error_message(self) -> str:
        return (
            f'Action topic "{self.action_topic}" was already used with message type '
            f'"{self.existing_message_type.__name__}", but is now requested with '
            f'"{self.requested_message_type.__name__}".'
        )

    def suggest_correction(self) -> str:
        return "Use a unique action_topic per message type."


@dataclass
class RobotWithoutMobileBaseError(MotionStatechartError):
    """
    Raised when a goal is to drive the mobile base of a robot that has none.
    """

    robot: AbstractRobot
    """
    The robot without a mobile base.
    """

    def error_message(self) -> str:
        return f"{self.robot.name} has no mobile base to drive."

    def suggest_correction(self) -> str:
        return "Move the base of this robot with a goal that needs no mobile base."


@dataclass
class EmptyDebugExpressionTrajectoryError(MotionStatechartError):
    """
    Raised when a plot is requested but no debug expression samples were recorded.
    """

    def error_message(self) -> str:
        return "Cannot plot: no debug expression samples were recorded."

    def suggest_correction(self) -> str:
        return "Call tick() at least once before plotting, or configure debug expressions to record."
