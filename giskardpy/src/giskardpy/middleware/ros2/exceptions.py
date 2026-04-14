# errors during execution
from dataclasses import dataclass

from giskardpy.data_types.exceptions import GiskardException


class ExecutionException(GiskardException):
    pass


@dataclass
class ExecutionCanceledException(ExecutionException):
    action_server_name: str
    goal_id: int

    def __post_init__(self):
        super().__init__(f"'{self.action_server_name}' goal #{self.goal_id} canceled")


class ExecutionPreemptedException(ExecutionException):
    pass


class ExecutionTimeoutException(ExecutionException):
    pass


class ExecutionAbortedException(ExecutionException):
    def __init__(self):
        super().__init__("Execution aborted by Giskard.")


class ExecutionSucceededPrematurely(ExecutionException):
    pass


class FollowJointTrajectory_INVALID_GOAL(ExecutionException):
    pass


class FollowJointTrajectory_INVALID_JOINTS(ExecutionException):
    pass


class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(ExecutionException):
    pass


class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(ExecutionException):
    pass


class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(ExecutionException):
    pass
