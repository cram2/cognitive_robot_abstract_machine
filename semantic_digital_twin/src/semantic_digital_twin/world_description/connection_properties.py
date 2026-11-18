from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from typing_extensions import TYPE_CHECKING, Self

from semantic_digital_twin.exceptions import UnnormalizedTwistError

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
    )


@dataclass
class JointDynamics:
    r"""
    Represents the dynamic properties of a joint.

    .. math::

       \tau = J \ddot{q} + R_\mathrm{c}\mathrm{sign}(\dot{q}) + R_\mathrm{v}\dot{q}

    where

    :math:`\tau` : torque applied to the joint

    :math:`J` : joint armature (inertia)

    :math:`\ddot{q}` : joint acceleration

    :math:`R_\mathrm{c}` : dry friction coefficient

    :math:`R_\mathrm{v}` : damping coefficient

    :math:`\dot{q}` : joint velocity
    """

    armature: float = 0.0
    """
    Additional inertia associated with movement of the joint that is not due to body
    mass.

    This added inertia is usually due to a rotor (a.k.a armature) spinning faster than
    the joint itself due to a geared transmission.
    """

    dry_friction: float = 0.0
    """
    Dry friction coefficient of the joint.
    """

    damping: float = 0.0
    """
    Viscous friction coefficient of the joint.
    """


@dataclass(kw_only=True)
class NormalizedTwist:
    """
    The screw axis of a screw axis connection as a normalized twist: how far its child
    rotates about and translates along the axis per unit of position, one radian if it
    rotates and one meter otherwise. A rate, unlike the displacement
    :class:`~krrood_spatial_math.spatial_types.Twist`.
    """

    rotation_per_unit_position: float
    """
    Radians the child rotates about the axis per unit of position, following the
    right-hand rule: 1 for a motion that rotates, 0 for one that only translates.
    """

    translation_per_unit_position: float
    """
    Meters the child translates along the axis per unit of position: 1 for a motion
    that only translates, the pitch in meters per radian for one that rotates.
    """

    def __post_init__(self):
        """
        :raises UnnormalizedTwistError: if the twist is not normalized.
        """
        rotates_one_radian = self.rotation_per_unit_position == 1
        translates_one_meter_only = (
            self.rotation_per_unit_position == 0
            and self.translation_per_unit_position == 1
        )
        if not (rotates_one_radian or translates_one_meter_only):
            raise UnnormalizedTwistError(
                rotation_per_unit_position=self.rotation_per_unit_position,
                translation_per_unit_position=self.translation_per_unit_position,
            )

    @classmethod
    def create_revolute(cls) -> Self:
        """
        A rotation about the axis without translation, as of a door hinge.
        """
        return cls(rotation_per_unit_position=1.0, translation_per_unit_position=0.0)

    @classmethod
    def create_prismatic(cls) -> Self:
        """
        A translation along the axis without rotation, as of a drawer slide.
        """
        return cls(rotation_per_unit_position=0.0, translation_per_unit_position=1.0)

    @classmethod
    def create_screw(cls, *, screw_pitch: float) -> Self:
        """
        The twist of a single-start thread that advances ``screw_pitch`` meters along
        the axis per full revolution; a negative pitch models a left-handed thread.
        """
        return cls(
            rotation_per_unit_position=1.0,
            translation_per_unit_position=screw_pitch / (2 * np.pi),
        )

    @property
    def behaves_like_a_revolute_connection(self) -> bool:
        """
        Whether the child only rotates about the axis.
        """
        return (
            self.rotation_per_unit_position != 0
            and self.translation_per_unit_position == 0
        )

    @property
    def behaves_like_a_prismatic_connection(self) -> bool:
        """
        Whether the child only translates along the axis.
        """
        return self.rotation_per_unit_position == 0

    @property
    def behaves_like_a_screw_connection(self) -> bool:
        """
        Whether the child rotates about the axis and translates along it at once.
        """
        return (
            self.rotation_per_unit_position != 0
            and self.translation_per_unit_position != 0
        )


@dataclass
class ServoGains:
    r"""
    How hard a position servo pulls its joint towards the position it was given.

    .. math::

       \tau = \min(\tau_\mathrm{max}, K (q_\mathrm{set} - q) - D \dot{q})

    where :math:`K` is the stiffness, :math:`D` the damping, :math:`q_\mathrm{set}` the
    commanded position and :math:`\tau_\mathrm{max}` the torque limit.
    """

    stiffness: float
    """
    Restoring torque per radian, or newton per metre, away from the set point.
    """

    damping: float
    """
    Opposing torque per radian per second the servo itself applies, on top of the
    joint's own passive damping.
    """

    torque_limit: float
    """
    The largest torque, or force, the servo may exert.
    """


@dataclass
class JointServo:
    """
    What a robot part declares for one of its joints to be driven by a position servo
    in a physical simulation: the servo's gains, which become a
    :class:`~semantic_digital_twin.world_description.world_entity.PositionServo`
    actuator on the joint's degree of freedom, and the passive dynamics of the joint
    itself.
    """

    gains: ServoGains
    """
    How hard the servo pulls the joint towards the commanded position.
    """

    dynamics: JointDynamics | None = None
    """
    The joint's own inertia and friction, felt on top of the servo's torque; ``None``
    keeps those the robot's description states.
    """


@dataclass
class JointDrive:
    """
    What drives one joint in a physical simulation: a position servo, and the motor
    turning the joint through its gearbox.
    """

    gains: ServoGains
    """
    How hard the servo pulls the joint towards the commanded position.
    """

    armature: float | None = None
    """
    The rotor inertia the motor adds to the joint through its gearbox; ``None`` keeps
    the armature the description states.
    """

    joint_damping: float | None = None
    """
    The joint's own passive damping, on top of the servo's; ``None`` keeps the damping
    the description states.
    """

    def servo_for(self, connection: ActiveConnection1DOF) -> JointServo:
        """
        :param connection: The joint to drive.
        :return: The servo driving the joint, with the passive dynamics the description
            gives the joint, apart from the :attr:`armature` and :attr:`joint_damping`
            this drive states.
        """
        dynamics = connection.dynamics
        if self.armature is not None:
            dynamics = replace(dynamics, armature=self.armature)
        if self.joint_damping is not None:
            dynamics = replace(dynamics, damping=self.joint_damping)
        return JointServo(gains=self.gains, dynamics=dynamics)


@dataclass
class VelocityServoGains:
    r"""
    How hard a velocity servo pushes its joint towards the velocity it was given.

    .. math::

       \tau = \min(\tau_\mathrm{max}, D (\dot{q}_\mathrm{set} - \dot{q}))

    where :math:`D` is the damping, :math:`\dot{q}_\mathrm{set}` the commanded velocity
    and :math:`\tau_\mathrm{max}` the torque limit.
    """

    damping: float
    """
    Torque per radian per second, or force per metre per second, the servo exerts
    against the difference between the commanded and the actual velocity.
    """

    torque_limit: float
    """
    The largest torque, or force, the servo may exert.
    """


@dataclass
class DriveServos:
    """
    What a mobile base declares for its drive to be moved across the floor by velocity
    servos in a physical simulation: one servo on each translational velocity the drive
    is commanded with, and one on its yaw. The drive's odometry is then what the
    physics makes of those velocities.
    """

    translation_gains: VelocityServoGains
    """
    How hard the servos on the translational velocities push the base towards its
    commanded velocity, in newton per metre per second.
    """

    rotation_gains: VelocityServoGains
    """
    How hard the servo on yaw turns the base towards its commanded angular velocity,
    in newton metre per radian per second.
    """


@dataclass
class DriveVelocityLimits:
    """
    How fast a drive may move a mobile base across the floor.
    """

    translation: float
    """
    The fastest the base may move along the floor, in meter per second.
    """

    rotation: float
    """
    The fastest the base may turn about its yaw, in radian per second.
    """
