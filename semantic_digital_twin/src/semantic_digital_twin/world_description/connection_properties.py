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
    Additional inertia associated with movement of the joint that is not due to body mass. 
    This added inertia is usually due to a rotor (a.k.a armature) spinning faster than the joint itself due to a geared transmission.
    """

    dry_friction: float = 0.0
    """
    Dry friction coefficient of the joint.
    """

    damping: float = 0.0
    """
    Viscous friction coefficient of the joint.
    """
