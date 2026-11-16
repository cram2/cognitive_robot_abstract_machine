"""
The parameters motion statechart nodes share, one concept per mixin, so that a parameter
has the same name, type and meaning in every node taking it.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.cruise_velocities import DefaultCruiseVelocity
from giskardpy.motion_statechart.weights import DefaultWeights
from krrood.patterns.mixin import ParameterMixin
from krrood_spatial_math.spatial_types import Point3, Pose, RotationMatrix
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% links of the kinematic chain


@dataclass(eq=False, repr=False, kw_only=True)
class HasRootLinkParameter(ParameterMixin):
    """
    A node acting on a kinematic chain, expressed relative to its root.
    """

    root_link: KinematicStructureEntity
    """
    The root of the kinematic chain, which the motion is expressed relative to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasTipLinkParameter(ParameterMixin):
    """
    A node moving the tip of a kinematic chain.
    """

    tip_link: KinematicStructureEntity
    """
    The tip of the kinematic chain, which is moved.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasKinematicChainParameters(HasRootLinkParameter, HasTipLinkParameter):
    """
    A node moving :attr:`tip_link` relative to :attr:`root_link`.
    """


# %% priority


@dataclass(eq=False, repr=False, kw_only=True)
class HasWeightParameter(ParameterMixin):
    """
    A node whose constraints compete with those of other nodes.
    """

    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value
    """
    Priority of the constraints of this node relative to those of other nodes.
    """


# %% goals


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalPoseParameter(ParameterMixin):
    """
    A node bringing something to a pose.
    """

    goal_pose: Pose
    """
    The pose this node brings what it moves to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalPointParameter(ParameterMixin):
    """
    A node bringing the tip of a kinematic chain to a point, or aiming it at one.
    """

    goal_point: Point3
    """
    The point the tip is brought to or aimed at.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalPointsParameter(ParameterMixin):
    """
    A node moving the tip of a kinematic chain along a sequence of points.
    """

    goal_points: list[Point3]
    """
    The points the tip passes in order, all in one reference frame, ending at the last.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalOrientationParameter(ParameterMixin):
    """
    A node turning the tip of a kinematic chain to an orientation.
    """

    goal_orientation: RotationMatrix
    """
    The orientation the tip is turned to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalStateParameter(ParameterMixin):
    """
    A node moving joints to positions of their own.
    """

    goal_state: JointState
    """
    The position each joint is moved to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalBindingPolicyParameter(ParameterMixin):
    """
    A node whose goal is given relative to a frame that may move.
    """

    goal_binding_policy: GoalBindingPolicy = GoalBindingPolicy.Bind_on_start
    """
    When the pose of the frame the goal is given in is read.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasThresholdParameter(ParameterMixin):
    """
    A node that counts its goal as reached once its error is small enough.
    """

    threshold: float = 0.01
    """
    Error at or below which the goal counts as reached, in the node's own units.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasPositionThresholdParameter(ParameterMixin):
    """
    A node that counts a goal position as reached once it is close enough.
    """

    position_threshold: float = 0.01
    """
    Distance at or below which the position counts as reached, in meters.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasOrientationThresholdParameter(ParameterMixin):
    """
    A node that counts a goal orientation as reached once it is close enough.
    """

    orientation_threshold: float = 0.01
    """
    Angle at or below which the orientation counts as reached, in radians.
    """


# %% velocities


@dataclass(eq=False, repr=False, kw_only=True)
class HasCruiseVelocityParameter(ParameterMixin):
    """
    A node that approaches its goal at a set speed.
    """

    cruise_velocity: float = DefaultCruiseVelocity.TRANSLATION
    """
    How fast the goal is approached, in the units of the node's error per second; not
    strictly enforced, a velocity limit sets a strict one.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasCruiseLinearVelocityParameter(ParameterMixin):
    """
    A node that approaches a goal position at a set speed.
    """

    cruise_linear_velocity: float = DefaultCruiseVelocity.TRANSLATION
    """
    How fast the goal position is approached, in m/s; not strictly enforced, a velocity
    limit sets a strict one.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasCruiseAngularVelocityParameter(ParameterMixin):
    """
    A node that approaches a goal orientation at a set speed.
    """

    cruise_angular_velocity: float = DefaultCruiseVelocity.ROTATION
    """
    How fast the goal orientation is approached, in rad/s; not strictly enforced, a
    velocity limit sets a strict one.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasMaxLinearVelocityParameter(ParameterMixin):
    """
    A node capping how fast the tip of a kinematic chain moves.
    """

    max_linear_velocity: float = 0.1
    """
    The highest linear speed of the tip allowed, in m/s.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasMaxAngularVelocityParameter(ParameterMixin):
    """
    A node capping how fast the tip of a kinematic chain turns.
    """

    max_angular_velocity: float = 0.4
    """
    The highest angular speed of the tip allowed, in rad/s.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasMaxJointVelocityParameter(ParameterMixin):
    """
    A node that may cap how fast the joints it drives move.
    """

    max_joint_velocity: float | None = None
    """
    The highest speed of each joint allowed, in rad/s or m/s; None leaves the joints
    capped only by their own limits.
    """


# %% fingers


@dataclass(eq=False, repr=False, kw_only=True)
class HasTwoFingerGripperParameter(ParameterMixin):
    """
    A node moving the fingers of a two-finger gripper.
    """

    two_finger_gripper: HasTwoFingers
    """
    The gripper whose fingers move.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasWidthParameter(ParameterMixin):
    """
    A node bringing the fingertips of a two-finger gripper to a given width.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces are to stand, in metres; negative
    where they are to overlap, as a fully closed gripper's may.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasTolerateFingerStallParameter(ParameterMixin):
    """
    A node closing fingers that may stop short of their goal on an object between them.
    """

    tolerate_finger_stall: bool = False
    """
    Whether fingers that have stopped approaching their goal count as done, even short
    of it.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasFingerStallMinimumTimeParameter(ParameterMixin):
    """
    A node telling fingers that stopped short of their goal from fingers still moving.
    """

    finger_stall_minimum_time: float | None = None
    """
    How long the fingers must have stopped approaching their goal before a stall counts,
    in seconds.

    None keeps :attr:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached.minimum_time`.
    """
