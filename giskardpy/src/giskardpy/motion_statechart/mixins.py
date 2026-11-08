"""
The parameters motion statechart nodes share, one concept per mixin, so that a parameter
has the same name, type and meaning in every node taking it.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.cruise_velocities import DefaultCruiseVelocity
from giskardpy.motion_statechart.weights import DefaultWeights
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

GoalType = TypeVar("GoalType")
"""
The type of what a node drives towards.
"""

FeatureType = TypeVar("FeatureType")
"""
The type of the geometric features a node compares.
"""

# %% links of the kinematic chain


@dataclass(eq=False, repr=False, kw_only=True)
class HasRootLinkParameter:
    """
    A node acting on a kinematic chain, expressed relative to its root.
    """

    root_link: KinematicStructureEntity
    """
    The root of the kinematic chain, which the motion is expressed relative to.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasTipLinkParameter:
    """
    A node moving the tip of a kinematic chain.
    """

    tip_link: KinematicStructureEntity
    """
    The tip of the kinematic chain, which is moved.
    """


# %% priority


@dataclass(eq=False, repr=False, kw_only=True)
class HasWeightParameter:
    """
    A node whose constraints compete with those of other nodes.
    """

    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value
    """
    Priority of the constraints of this node relative to those of other nodes.
    """


# %% goals


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalParameter(Generic[GoalType], SubClassSafeGeneric):
    """
    A node driving something towards a goal.
    """

    goal: GoalType
    """
    What this node drives towards.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasGoalBindingPolicyParameter:
    """
    A node whose goal is given relative to a frame that may move.
    """

    binding_policy: GoalBindingPolicy = GoalBindingPolicy.Bind_on_start
    """
    When the pose of the frame the goal is given in is read.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasThresholdParameter:
    """
    A node that counts its goal as reached once its error is small enough.
    """

    threshold: float = 0.01
    """
    Error at or below which the goal counts as reached, in the node's own units.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasPoseThresholdParameters:
    """
    A node that counts a pose as reached once both its position and its orientation
    are close enough.
    """

    position_threshold: float = 0.01
    """
    Distance at or below which the position counts as reached, in meters.
    """

    orientation_threshold: float = 0.01
    """
    Angle at or below which the orientation counts as reached, in radians.
    """


# %% velocities


@dataclass(eq=False, repr=False, kw_only=True)
class HasCruiseVelocityParameter:
    """
    A node that approaches its goal at a set speed.
    """

    cruise_velocity: float = DefaultCruiseVelocity.TRANSLATION
    """
    How fast the goal is approached, in the units of the node's error per second; not
    strictly enforced, a velocity limit sets a strict one.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasPoseCruiseVelocityParameters:
    """
    A node that approaches a goal position and a goal orientation at a set speed each.
    """

    cruise_linear_velocity: float = DefaultCruiseVelocity.TRANSLATION
    """
    How fast the goal position is approached, in m/s; not strictly enforced, a velocity
    limit sets a strict one.
    """

    cruise_angular_velocity: float = DefaultCruiseVelocity.ROTATION
    """
    How fast the goal orientation is approached, in rad/s; not strictly enforced, a
    velocity limit sets a strict one.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasLinearVelocityLimitParameter:
    """
    A node capping how fast the tip of a kinematic chain moves.
    """

    max_linear_velocity: float = 0.1
    """
    The highest linear speed of the tip allowed, in m/s.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasAngularVelocityLimitParameter:
    """
    A node capping how fast the tip of a kinematic chain turns.
    """

    max_angular_velocity: float = 0.4
    """
    The highest angular speed of the tip allowed, in rad/s.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasJointVelocityLimitParameter:
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
class HasTwoFingerGripperParameters:
    """
    A node bringing the fingertips of a two-finger gripper to a given width.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers move.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces are to stand, in metres; negative
    where they are to overlap, as a fully closed gripper's may.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class HasFingerStallToleranceParameters:
    """
    A node closing fingers that may stop short of their goal on an object between them.
    """

    tolerate_finger_stall: bool = False
    """
    Whether fingers that have stopped approaching their goal count as done, even short
    of it.
    """

    finger_stall_minimum_time: float | None = None
    """
    How long the fingers must have stopped approaching their goal before a stall counts,
    in seconds.

    None keeps :attr:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached.minimum_time`.
    """


# %% pairs of geometric features


@dataclass(eq=False, repr=False, kw_only=True)
class HasFeaturePairParameters(Generic[FeatureType], SubClassSafeGeneric):
    """
    A node moving one geometric feature relative to another.
    """

    controlled_feature: FeatureType
    """
    The feature that is moved, in the frame of the link it is fixed on.
    """

    reference_feature: FeatureType
    """
    The feature :attr:`controlled_feature` is measured against, in the frame of the
    link it is fixed on.
    """
