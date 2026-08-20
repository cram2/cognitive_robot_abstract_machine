"""
Exceptions raised while executing a trajectory on a robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type

from giskardpy.data_types.exceptions import (
    DontPrintStackTrace,
    GiskardException,
    SetupException,
)
from semantic_digital_twin.world_description.world_entity import Connection


@dataclass
class UnknownMinimumVelocityJointError(SetupException):
    """
    Raised when a minimum velocity override names a joint that is not commanded.
    """

    joint_name: str
    """
    The joint the override was written for.
    """

    commanded_joint_names: List[str]
    """
    The joints the publisher actually commands.
    """

    def error_message(self) -> str:
        return (
            f'The minimum velocity override for "{self.joint_name}" applies to no '
            f"commanded joint."
        )

    def suggest_correction(self) -> str:
        return f"Use one of {sorted(self.commanded_joint_names)}."


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
class WorldModelModifiedDuringMotionError(ExecutionException, DontPrintStackTrace):
    """
    Raised when another process modified the world model while a motion was running.

    The motion statechart and the quadratic program are compiled against the structure
    of the world, so the modification cannot be applied under a running motion. The
    motion is terminated instead and the modification is applied once Giskard is idle
    again.
    """

    def error_message(self) -> str:
        return "The world model was modified by another process during the motion."

    def suggest_correction(self) -> str:
        return "Send the goal again; the modification is applied by then."


@dataclass
class RequiredWorldUpdateNotReceivedError(ExecutionException, DontPrintStackTrace):
    """
    Raised when a goal names a change of the client's world that never arrived.

    The goal refers to a world the client already changed, so executing it against the
    world Giskard has would act on something else than what was asked for.
    """

    publisher_name: str
    """
    The node whose change was waited for.
    """

    current_sequence_number: int
    """
    The position in the stream that we have.
    """

    awaited_sequence_number: int
    """
    The position in that node's stream that was waited for.
    """

    timeout: float
    """
    Seconds that were spent waiting.
    """

    def error_message(self) -> str:
        return (
            f"Update #{self.awaited_sequence_number} of '{self.publisher_name}' did not "
            f"arrive within {self.timeout}s. Current update is #{self.current_sequence_number}."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the world of the client is still connected to the world sync "
            "topic, then send the goal again."
        )


@dataclass
class GiskardWorldUpdateNotReceivedError(ExecutionException, DontPrintStackTrace):
    """
    Raised when the changes Giskard made during a goal never reached the client.

    Reading the world of the client after such a goal would show a world that Giskard
    has already moved on from.
    """

    awaited_sequence_number: int
    """
    The position in Giskard's stream that was waited for.
    """

    timeout: float
    """
    Seconds that were spent waiting.
    """

    def error_message(self) -> str:
        return (
            f"Update #{self.awaited_sequence_number} of Giskard did not arrive within "
            f"{self.timeout}s."
        )

    def suggest_correction(self) -> str:
        return "Check that this world is still connected to the world sync topic."


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
class AlreadyTrackedByTfFrameError(SetupException):
    """
    Raised when a connection is registered for tf tracking a second time.
    """

    connection_name: str
    """
    The name of the connection that is already tracked.
    """

    tf_parent_frame: str
    """
    The tf parent frame the connection is already tracked with.
    """

    tf_child_frame: str
    """
    The tf child frame the connection is already tracked with.
    """

    def error_message(self) -> str:
        return (
            f"Connection '{self.connection_name}' is already tracked with a tf frame: "
            f"'{self.tf_parent_frame}'<-'{self.tf_child_frame}'"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnboundMessageTypeError(SetupException):
    """
    Raised when a topic synchronizer does not name the type of its messages.
    """

    synchronizer_type: Type
    """
    The synchronizer whose message type is unknown.
    """

    def error_message(self) -> str:
        return (
            f"'{self.synchronizer_type.__name__}' does not name the type of the "
            f"messages it reads."
        )

    def suggest_correction(self) -> str:
        return (
            f"Declare it in the bases of '{self.synchronizer_type.__name__}', as in "
            f"'TopicInputSynchronizer[Odometry]'."
        )


@dataclass
class ConnectionCannotBeTrackedByTfFrameError(SetupException):
    """
    Raised when a connection without 6 degrees of freedom is registered for tf tracking.
    """

    connection: Connection
    """
    The connection that cannot be tracked.
    """

    def error_message(self) -> str:
        return (
            f"Can only sync Connection6DoF with tf, but '{str(self.connection.name)}' is of "
            f"type '{type(self.connection).__name__}'."
        )

    def suggest_correction(self) -> str:
        return ""
