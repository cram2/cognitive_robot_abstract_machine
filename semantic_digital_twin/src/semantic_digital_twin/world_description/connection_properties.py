from dataclasses import dataclass


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

    dynamics: JointDynamics
    """
    The joint's own inertia and friction, felt on top of the servo's torque.
    """
