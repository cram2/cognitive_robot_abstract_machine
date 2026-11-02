from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    NoProgressError,
)
from cramph.exceptions import ExecutionFailure
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from cramph.node import StatechartNode


@dataclass
class PlanFailure(ExecutionFailure):
    """
    Base class for all exceptions that are related to plan errors.

    Can also be raised directly as a generic plan failure.
    """

    def error_message(self) -> str:
        return "Plan failed."

    def suggest_correction(self) -> str:
        return ""


# %% what a plan can recover from


@dataclass
class MotionMadeNoProgress(PlanFailure):
    """
    Raised when a motion stopped approaching its goal.

    The chart cancels itself with a
    :class:`~giskardpy.motion_statechart.exceptions.NoProgressError`, which says this
    attempt did not work rather than that the plan cannot go on. Wrapping it where it
    crosses into a plan is what lets a plan choose an alternative by catching
    :class:`PlanFailure` alone.
    """

    no_progress: NoProgressError
    """
    The stall the motion reported, which names the tasks that stopped converging.
    """

    def error_message(self) -> str:
        return self.no_progress.error_message()

    def suggest_correction(self) -> str:
        return self.no_progress.suggest_correction()


@dataclass
class MotionExceededSimulationTimeLimit(PlanFailure):
    """
    Raised when a simulated plan ran for longer than the actions it started are allowed
    to take together.

    The chart's stall monitor ends a motion that stopped approaching its goal, but one
    that keeps creeping towards it, or that is held and so never counts as stalled,
    would tick forever without this limit.
    """

    time_limit: timedelta
    """
    The simulated time the plan was allowed.
    """

    def error_message(self) -> str:
        return (
            f"The plan did not end within {self.time_limit.total_seconds()} s of "
            f"simulated time."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that every goal of the plan can be reached and that nothing holds "
            "the chart, or raise SimulatedPlanExecutor.simulation_time_per_action for "
            "actions that are meant to take this long."
        )


@dataclass
class MotionViolatedCollisionAvoidance(PlanFailure):
    """
    Raised when a motion brought bodies closer to each other than collision avoidance
    allows.

    Like :class:`MotionMadeNoProgress`, this says the attempt did not work from where it
    started rather than that the plan cannot go on, so a plan can try another candidate
    by catching :class:`PlanFailure` alone.
    """

    violation: CollisionViolatedError
    """
    The violation the motion reported, which names the body pairs that came too close.
    """

    def error_message(self) -> str:
        return self.violation.error_message()

    def suggest_correction(self) -> str:
        return self.violation.suggest_correction()


@dataclass
class EmptyUnderspecified(PlanFailure):
    """
    Raised when an underspecified statement yields no action that succeeds.
    """

    node: StatechartNode
    """
    The node whose statement ran out of actions.
    """

    def error_message(self) -> str:
        return f"{self.node} ran out of actions to try."

    def suggest_correction(self) -> str:
        return "Widen the domains of the statement, or check why its actions fail."


@dataclass
class CandidateLimitReached(EmptyUnderspecified):
    """
    Raised when an underspecified step has tried as many candidates as it may without
    one of them succeeding.
    """

    candidate_limit: int
    """
    How many candidates it was allowed to try.
    """

    def error_message(self) -> str:
        return (
            f"{self.node} tried {self.candidate_limit} candidates without one of them "
            f"succeeding."
        )

    def suggest_correction(self) -> str:
        return (
            "Allow more candidates, through the step's own limit or the context's "
            "candidates_to_try, or check whether any candidate can succeed at all."
        )


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

    arm: Arm
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
