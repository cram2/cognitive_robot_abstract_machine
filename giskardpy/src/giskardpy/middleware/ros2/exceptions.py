"""
Exceptions raised while executing a trajectory on a robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.data_types.exceptions import GiskardException, SetupException


@dataclass
class ExecutionException(GiskardException):
    """
    Base class for errors that occur while executing a trajectory.
    """


@dataclass
class NoActiveGoalToCancelError(ExecutionException):
    """
    Raised when a goal cancellation is requested but no goal is active.
    """

    def error_message(self) -> str:
        return "Can't cancel goals, because there is no active one."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ExecutionCanceledException(ExecutionException):
    """
    Raised when the execution of a goal is canceled.
    """

    action_server_name: str
    """
    The name of the action server whose goal was canceled.
    """

    goal_id: int
    """
    The id of the canceled goal.
    """

    def error_message(self) -> str:
        return f"'{self.action_server_name}' goal #{self.goal_id} canceled"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ExecutionPreemptedException(ExecutionException):
    """
    Raised when the execution of a goal is preempted.
    """

    namespace: str
    """
    The namespace of the action server that was preempted.
    """

    def error_message(self) -> str:
        return f"'{self.namespace}' preempted. Stopping execution."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ExecutionTimeoutException(ExecutionException):
    """
    Raised when the execution of a goal takes too long.
    """

    namespace: str
    """
    The namespace of the action server that timed out.
    """

    reason: str
    """
    A description of why the execution timed out.
    """

    def error_message(self) -> str:
        return f"'{self.namespace}' timed out. {self.reason}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ExecutionAbortedException(ExecutionException):
    """
    Raised when the execution is aborted by Giskard.
    """

    def error_message(self) -> str:
        return "Execution aborted by Giskard."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ExecutionSucceededPrematurely(ExecutionException):
    """
    Raised when the execution finishes before the minimum execution time.
    """

    namespace: str
    """
    The namespace of the action server that finished too early.
    """

    def error_message(self) -> str:
        return f"'{self.namespace}' executed too quickly, stopping execution."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class FollowJointTrajectoryError(ExecutionException):
    """
    Raised when a follow joint trajectory action server fails to execute a goal.
    """

    namespace: str
    """
    The namespace of the action server that failed.
    """

    error_description: str
    """
    A human-readable description of the action server error code.
    """

    def error_message(self) -> str:
        return f"'{self.namespace}' failed to execute goal. Error: '{self.error_description}'"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class FollowJointTrajectory_INVALID_GOAL(FollowJointTrajectoryError):
    """
    Raised when the action server reports an invalid goal.
    """


@dataclass
class FollowJointTrajectory_INVALID_JOINTS(FollowJointTrajectoryError):
    """
    Raised when the action server reports invalid joints.
    """


@dataclass
class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(FollowJointTrajectoryError):
    """
    Raised when the action server reports an outdated header timestamp.
    """


@dataclass
class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(FollowJointTrajectoryError):
    """
    Raised when the action server reports a path tolerance violation.
    """


@dataclass
class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(FollowJointTrajectoryError):
    """
    Raised when the action server reports a goal tolerance violation.
    """


@dataclass
class FollowJointTrajectoryServerRequiresPlanningModeError(SetupException):
    """
    Raised when a follow joint trajectory server is added outside of planning mode.
    """

    def error_message(self) -> str:
        return "add_follow_joint_trajectory_server only works in planning mode."

    def suggest_correction(self) -> str:
        return ""
