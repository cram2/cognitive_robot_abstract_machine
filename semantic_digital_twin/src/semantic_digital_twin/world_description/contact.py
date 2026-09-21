"""
What a shape's surface does in a contact: the friction it offers, and how stiffly and
how hard the contact resolves.

A physics engine reads these off the shape as a simulator property; a shape without them
gets the engine's own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

from typing_extensions import TYPE_CHECKING, Iterable, List, Optional

from semantic_digital_twin.mixin import UniqueSimulatorProperty

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ContactFriction:
    """
    The friction a shape's surface offers in a contact, in the three coefficients a
    physics engine resolves a contact with.
    """

    sliding: float = 1.0
    """
    Friction along both axes of the tangent plane.
    """

    torsional: float = 0.005
    """
    Friction around the contact normal.
    """

    rolling: float = 0.0001
    """
    Friction around both axes of the tangent plane.
    """

    def to_list(self) -> List[float]:
        """
        :return: The coefficients as sliding, torsional and rolling.
        """
        return [self.sliding, self.torsional, self.rolling]


@dataclass
class ContactStiffness:
    """
    How stiff and how damped a contact is, as the time it takes to resolve a penetration
    and the damping of that resolution.
    """

    time_constant: timedelta = timedelta(milliseconds=20)
    """
    Time the contact takes to resolve a penetration; a smaller value is a stiffer
    contact.
    """

    damping_ratio: float = 1.0
    """
    Damping of that resolution; ``1`` is critically damped.
    """

    def to_list(self) -> List[float]:
        """
        :return: The pair as time constant, in seconds, and damping ratio.
        """
        return [self.time_constant.total_seconds(), self.damping_ratio]


@dataclass
class ContactImpedance:
    """
    How hard a contact pushes back as it is penetrated: the impedance rises from a
    minimum at zero penetration to a maximum over a given width.
    """

    minimum: float = 0.9
    """
    Impedance at zero penetration, between 0 and 1; a higher value is a harder contact.
    """

    maximum: float = 0.95
    """
    Impedance once the penetration reaches :attr:`width`.
    """

    width: float = 0.001
    """
    Penetration, in metres, over which the impedance rises from minimum to maximum.
    """

    midpoint: float = 0.5
    """
    Where along that width the rise is steepest, as a fraction of it.
    """

    power: float = 2.0
    """
    How sharply the rise bends around the midpoint.
    """

    def to_list(self) -> List[float]:
        """
        :return: The quintuple as minimum, maximum, width, midpoint and power.
        """
        return [self.minimum, self.maximum, self.width, self.midpoint, self.power]


@dataclass
class ContactParameters(UniqueSimulatorProperty):
    """
    The contact parameters a shape carries for a physical simulation: its friction and,
    optionally, how stiffly and how hard its contacts resolve.

    A physics engine combines the friction of two shapes in contact as the larger of the
    two, so a contact is only as slippery as the grippier of its two sides.
    """

    friction: ContactFriction
    """
    Sliding, torsional and rolling friction.
    """

    stiffness: Optional[ContactStiffness] = None
    """
    How stiff and how damped the contacts are.

    ``None`` is not "unspecified, use a default": it deliberately leaves whatever a
    geometry already declares untouched, so applying friction-only parameters (see
    :meth:`create_for_surface`) never resets a stiffness declared elsewhere.
    """

    impedance: Optional[ContactImpedance] = None
    """
    How hard the contacts push back as they are penetrated.

    ``None`` is not "unspecified, use a default": it deliberately leaves whatever a
    geometry already declares untouched, so applying friction-only parameters (see
    :meth:`create_for_surface`) never resets an impedance declared elsewhere.
    """

    @classmethod
    def create_for_grasped_object(
        cls,
        sliding_friction: float = 0.3,
        torsional_friction: float = 0.05,
        rolling_friction: float = 0.001,
        resolution_time_constant: timedelta = timedelta(milliseconds=8),
        minimum_impedance: float = 0.96,
        maximum_impedance: float = 0.99,
    ) -> ContactParameters:
        """
        The parameters that let a gripper pick an object up and hold it by friction.

        The contacts are stiffer and harder than a physics engine's defaults, since a
        soft contact lets a pinched object sink into the fingers and slip back out as
        the arm lifts. The torsional and rolling friction are raised above the defaults
        to keep a held object from pivoting between the pads.

        :param sliding_friction: The sliding friction coefficient; ``0.3`` approximates
            painted wood or plastic.
        :param torsional_friction: Friction around the contact normal, ten times the
            default so the object does not spin between the pads.
        :param rolling_friction: Friction around the tangent axes, ten times the default
            so the object does not roll out of the pads.
        :param resolution_time_constant: How long a contact takes to resolve a
            penetration; 8ms is a stiff contact the fingers cannot sink into.
        :param minimum_impedance: How hard the contact pushes back at zero penetration.
        :param maximum_impedance: How hard it pushes back once fully penetrated.
        :return: The parameters.
        """
        return cls(
            friction=ContactFriction(
                sliding=sliding_friction,
                torsional=torsional_friction,
                rolling=rolling_friction,
            ),
            stiffness=ContactStiffness(time_constant=resolution_time_constant),
            impedance=ContactImpedance(
                minimum=minimum_impedance, maximum=maximum_impedance
            ),
        )

    @classmethod
    def create_for_surface(cls, sliding_friction: float = 0.3) -> ContactParameters:
        """
        The friction of a surface loose objects rest on, with the default torsional and
        rolling friction, since a surface is never pinched between fingers.

        :param sliding_friction: The sliding friction coefficient.
        :return: The parameters.
        """
        return cls(friction=ContactFriction(sliding=sliding_friction))

    def apply_to(self, bodies: Iterable[Body]) -> None:
        """
        Declare these parameters on every collision geometry of every body that has
        collision geometry, in place: a geometry without contact parameters gets a copy
        of these, one that has some keeps its own stiffness and impedance where these
        leave them open.

        :param bodies: The bodies to modify.
        """
        for body in bodies:
            if not body.has_collision():
                continue
            for geometry in body.collision:
                existing = geometry.get_simulator_property_of_type(ContactParameters)
                if existing is None:
                    geometry.add_simulator_property(replace(self))
                    continue
                existing.friction = self.friction
                if self.stiffness is not None:
                    existing.stiffness = self.stiffness
                if self.impedance is not None:
                    existing.impedance = self.impedance
