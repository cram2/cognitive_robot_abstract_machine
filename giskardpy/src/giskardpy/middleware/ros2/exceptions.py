"""
Exceptions raised while executing a trajectory on a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.data_types.exceptions import (
    GiskardException,
    SetupException,
)
from semantic_digital_twin.adapters.ros.messages import MetaData
from semantic_digital_twin.world_description.connections import WheeledDrive


@dataclass
class UnknownMinimumVelocityJointError(SetupException):
    """
    Raised when a minimum velocity override names a joint that is not commanded.
    """

    joint_name: str
    """
    The joint the override was written for.
    """

    commanded_joint_names: list[str]
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
class ServiceUnavailableError(SetupException):
    """
    Raised when a service Giskard needs for its setup does not become available.
    """

    service_name: str
    """
    The service that was waited for.
    """

    def error_message(self) -> str:
        return f'The service "{self.service_name}" is not available.'

    def suggest_correction(self) -> str:
        return "Make sure the node offering the service is running."


@dataclass
class ControllersNotLoadedError(SetupException):
    """
    Raised when controllers Giskard was told to use are not loaded by their controller
    manager.
    """

    controller_manager_name: str
    """
    The controller manager that was asked.
    """

    missing_controllers: list[str]
    """
    The requested controllers the manager does not report.
    """

    def error_message(self) -> str:
        return (
            f'The controller manager "{self.controller_manager_name}" has not loaded '
            f"the controllers {self.missing_controllers}."
        )

    def suggest_correction(self) -> str:
        return "Load the controllers or remove them from the whitelist."


@dataclass
class ControllersNotActiveError(SetupException):
    """
    Raised when controllers Giskard was told to use are loaded by their controller
    manager but do not run.
    """

    controller_manager_name: str
    """
    The controller manager that was asked.
    """

    inactive_controllers: list[str]
    """
    The requested controllers the manager reports in another state than active.
    """

    def error_message(self) -> str:
        return (
            f"The controllers {self.inactive_controllers} of the controller manager "
            f'"{self.controller_manager_name}" are not active.'
        )

    def suggest_correction(self) -> str:
        return "Activate the controllers or remove them from the whitelist."


@dataclass
class NoUniqueDriveError(SetupException):
    """
    Raised when Giskard is told to command the drive of a world without being told which
    drive, and the world does not hold exactly one.
    """

    drives: list[WheeledDrive]
    """
    The drives the world holds.
    """

    def error_message(self) -> str:
        return (
            f"Expected exactly one drive to command, but the world holds "
            f"{[str(drive.name) for drive in self.drives]}."
        )

    def suggest_correction(self) -> str:
        return "Name the drive to command."


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

    print_stack_trace: bool = field(default=False, kw_only=True)

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
class ClientDisconnectedError(ExecutionException):
    """
    Raised when the client that sent the running goal disconnected.

    Nobody is waiting for the motion any more, so it is stopped instead of being run to
    its end.
    """

    print_stack_trace: bool = field(default=False, kw_only=True)

    client: MetaData
    """
    The client that sent the goal and is now gone.
    """

    def error_message(self) -> str:
        return (
            f"The client '{self.client.node_name}' (process {self.client.process_id}) "
            f"that sent this goal is gone."
        )

    def suggest_correction(self) -> str:
        return "Restart the client and send the goal again."


@dataclass
class NoWatchedClientError(GiskardException):
    """
    Raised when a presence check is asked about a client while it watches none.
    """

    check_type: type
    """
    The check that was asked.
    """

    def error_message(self) -> str:
        return f"'{self.check_type.__name__}' is not watching a client."

    def suggest_correction(self) -> str:
        return "Only ask a check about a client that it started watching."


@dataclass
class WorldModelModifiedDuringMotionError(ExecutionException):
    """
    Raised when another process modified the world model while a motion was running.

    The motion statechart and the quadratic program are compiled against the structure
    of the world, so the modification cannot be applied under a running motion. The
    motion is terminated instead and the modification is applied once Giskard is idle
    again.
    """

    print_stack_trace: bool = field(default=False, kw_only=True)

    def error_message(self) -> str:
        return "The world model was modified by another process during the motion."

    def suggest_correction(self) -> str:
        return "Send the goal again; the modification is applied by then."


@dataclass
class RequiredWorldUpdateNotReceivedError(ExecutionException):
    """
    Raised when a goal names a change of the client's world that never arrived.

    The goal refers to a world the client already changed, so executing it against the
    world Giskard has would act on something else than what was asked for.
    """

    print_stack_trace: bool = field(default=False, kw_only=True)

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
class GiskardWorldUpdateNotReceivedError(ExecutionException):
    """
    Raised when the changes Giskard made during a goal never reached the client.

    Reading the world of the client after such a goal would show a world that Giskard
    has already moved on from.
    """

    print_stack_trace: bool = field(default=False, kw_only=True)

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
class UnserializableGoalError(ExecutionException):
    """
    Stands in for the error a goal failed with when that error could not be serialized,
    so that the client still learns what went wrong.
    """

    error_class_name: str
    """
    The fully qualified name of the class of the error the goal failed with.
    """

    message: str
    """
    The message of the error the goal failed with.
    """

    def error_message(self) -> str:
        return f"Goal failed with '{self.error_class_name}': {self.message}"

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
class StatechartOutOfStepError(ExecutionException):
    """
    Raised when a client sends a child for a statechart that holds other nodes than the
    client's copy of it.
    """

    node_count: int
    """
    How many nodes the running statechart holds.
    """

    first_sent_node_index: int
    """
    The index the client gave the first node it sent.
    """

    def error_message(self) -> str:
        return (
            f"The client sent nodes from index {self.first_sent_node_index} on, but the "
            f"running statechart holds {self.node_count}."
        )

    def suggest_correction(self) -> str:
        return "Send children only for the goal the client's statechart was sent as."


@dataclass
class StatechartNotEndedByResultError(ExecutionException):
    """
    Raised when the result of a goal that succeeded leaves the client's statechart in a
    state that has not ended.
    """

    def error_message(self) -> str:
        return "The result of the finished goal does not end the client's statechart."

    def suggest_correction(self) -> str:
        return (
            "Check that the statechart Giskard ran holds the same nodes as the "
            "client's copy of it."
        )


@dataclass
class StateUpdateAheadOfStatechartError(ExecutionException):
    """
    Raised when Giskard reports the states of more nodes than the client's copy of the
    statechart holds.
    """

    node_count: int
    """
    How many nodes the client's copy holds.
    """

    reported_node_count: int
    """
    How many nodes Giskard reported a state for.
    """

    def error_message(self) -> str:
        return (
            f"Giskard reported the states of {self.reported_node_count} nodes, but the "
            f"client's statechart holds {self.node_count}."
        )

    def suggest_correction(self) -> str:
        return (
            "Apply the states only to the statechart that was sent as the running goal."
        )


@dataclass
class MotionServerThreadStillRunningError(GiskardException):
    """
    Raised when a motion server's background thread does not stop in time.
    """

    timeout: float
    """
    Seconds :meth:`~giskardpy.middleware.ros2.motion_server.MotionServer.stop` waited
    before giving up on the thread.
    """

    def error_message(self) -> str:
        return (
            f"The motion server's background thread did not stop within "
            f"{self.timeout} seconds."
        )

    def suggest_correction(self) -> str:
        return "Check whether a goal is stuck or the idle loop is blocked."
