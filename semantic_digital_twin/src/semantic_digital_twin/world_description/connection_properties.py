from __future__ import annotations

from dataclasses import dataclass, replace

from typing_extensions import TYPE_CHECKING

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
