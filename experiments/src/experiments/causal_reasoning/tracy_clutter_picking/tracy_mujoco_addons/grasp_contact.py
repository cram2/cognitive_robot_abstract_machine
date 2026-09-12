"""
Contact tuning for a MuJoCo-simulated gripper's grasp on a loose object and for the
surface it rests on, generalized from ``coraplex_panda_demo``'s own reliably-grasped
cube.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterable, Optional, Tuple

from semantic_digital_twin.adapters.multi_sim import MujocoGeom
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.world_entity import Body


def mujoco_geom_for(shape: Shape) -> MujocoGeom:
    """
    ``shape``'s own :class:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom`
    additional property, creating one if it has none yet.

    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoGeomConverter` reads only
    the first ``MujocoGeom`` it finds on a shape, so a second, appended one would be
    silently ignored: callers must modify the returned instance in place rather than
    replacing it.

    :param shape: The shape to find or create a ``MujocoGeom`` on, modified in place if
        none exists yet.
    :return: The shape's ``MujocoGeom``.
    """
    existing = [
        additional_property
        for additional_property in shape.simulator_additional_properties
        if isinstance(additional_property, MujocoGeom)
    ]
    if existing:
        return existing[0]
    mujoco_geom = MujocoGeom()
    shape.simulator_additional_properties.append(mujoco_geom)
    return mujoco_geom


@dataclass(frozen=True)
class ContactParameters:
    """
    The contact parameters one kind of geometry gets in MuJoCo: its friction and,
    optionally, how stiffly its contacts resolve.

    Contact friction is combined by MuJoCo as the element-wise maximum of the two
    participating geoms, so a contact is only as slippery as the grippier of its two
    sides.
    """

    friction: Tuple[float, float, float]
    """
    Sliding, torsional and rolling friction; see
    :attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.friction`.
    """

    solver_reference: Optional[Tuple[float, ...]] = None
    """
    Contact solver reference (see
    :attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_reference`), or
    ``None`` to leave the geometry's own.
    """

    solver_impedance: Optional[Tuple[float, ...]] = None
    """
    Contact solver impedance (see
    :attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_impedance`), or
    ``None`` to leave the geometry's own.
    """

    @classmethod
    def grasped_object(cls, sliding_friction: float = 0.3) -> ContactParameters:
        """
        The parameters that let a gripper pick an object up and hold it.

        The solver reference and impedance are ``coraplex_panda_demo``'s own reliably-
        grasped cube's (``solref="0.008"``, ``solimp="0.96 0.99"``): stiffer and harder
        than MuJoCo's defaults (``0.02`` and ``0.9 0.95``), since a soft contact lets a
        pinched object sink into the fingers and then slip back out as the arm lifts.
        The torsional and rolling friction are lifted above MuJoCo's defaults to keep a
        held object from pivoting between the pads.

        :param sliding_friction: The sliding friction coefficient; ``0.3`` approximates
            painted wood or plastic.
        :return: The parameters.
        """
        return cls(
            friction=(sliding_friction, 0.05, 0.001),
            solver_reference=(0.008, 1.0),
            solver_impedance=(0.96, 0.99, 0.001, 0.5, 2.0),
        )

    @classmethod
    def surface(cls, sliding_friction: float = 0.3) -> ContactParameters:
        """
        The friction of a surface loose objects rest on, with MuJoCo's own torsional and
        rolling defaults, since a surface is never pinched between fingers.

        Made explicit so an object-surface contact can drop below the finger-dominated
        grip instead of being pinned at MuJoCo's own ``1.0`` sliding default.

        :param sliding_friction: The sliding friction coefficient.
        :return: The parameters.
        """
        return cls(friction=(sliding_friction, 0.005, 0.0001))

    @classmethod
    def cube(cls) -> ContactParameters:
        """
        The parameters of ``coraplex_panda_demo/stacking_scene.xml``'s own proven-
        working cube (``friction="1 0.05 0.001"``), for an object that is to be grasped
        without its friction being the question.

        :return: The parameters.
        """
        return cls.grasped_object(sliding_friction=1.0)

    def apply_to(self, bodies: Iterable[Body]) -> None:
        """
        Give every collision geometry of every body these parameters, in place.

        :param bodies: The bodies to modify.
        """
        for body in bodies:
            for geometry in body.collision:
                mujoco_geom = mujoco_geom_for(geometry)
                mujoco_geom.friction = list(self.friction)
                if self.solver_reference is not None:
                    mujoco_geom.solver_reference = list(self.solver_reference)
                if self.solver_impedance is not None:
                    mujoco_geom.solver_impedance = list(self.solver_impedance)
