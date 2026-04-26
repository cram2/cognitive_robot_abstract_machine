"""
Pouring-domain physics classes: differential equations, fill-level mixin, and liquid connection.

Provides the SDT-native building blocks for pouring simulation that carry no
giskardpy dependency. The giskardpy-dependent physics models (PouringMSCModel,
CoupledPouringMSCModel) live in pycram.body_motion_problem.pouring.physics.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.semantic_annotations.mixins import ContainerGeometry
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    HasUpdateState,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PouringEquation(DifferentialEquation):
    """
    Abstract ODE for pouring-domain fill-level dynamics.

    Owns the tilt and fill connections plus the outflow rate constant ``k``.
    Concrete subclasses implement :meth:`symbolic_velocity` and optionally
    override :meth:`symbolic_tilt_floor`.

    :param tilt_connection: Revolute DOF providing the current tilt angle θ.
    :param fill_connection: Prismatic DOF whose position encodes fill level in [0, 1].
    :param k: Outflow rate constant.
    """

    tilt_connection: RevoluteConnection
    fill_connection: PrismaticConnection
    k: float = field(default=1.0, kw_only=True)

    @abstractmethod
    def symbolic_velocity(self) -> Scalar:
        """
        Symbolic d(fill_normalized)/dt as a CasADi expression.

        :return: Symbolic desired fill velocity.
        """

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Symbolic minimum tilt angle at which flow begins for the given fill level.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic tilt floor angle in radians.
        """
        return sm.Scalar(0.0)


@dataclass
class ArticulatedPouringEquation(PouringEquation):
    """
    Pouring ODE derived from the 2-D rectangular-cup model.

    Computes the effective discharge gap from actual cup dimensions (height ``A``,
    half-width ``r``) and the current tilt angle::

        L(h)    = √((A − h)² + r²)
        φ(h)    = atan2(A − h, r)
        d(α, h) = max(0, L(h) · sin(α − φ(h)))
        ḣ       = −k · d(α, h)

    :param container_geometry: Physical dimensions of the container.
    """

    container_geometry: ContainerGeometry

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Returns the geometric tilt offset φ(fill) — the minimum tilt for flow.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic φ(fill) in radians.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        return sm.atan2(A - fill_sym * A, r)

    def symbolic_velocity(self) -> Scalar:
        """
        :return: Symbolic d(fill_normalized)/dt as a CasADi expression.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        h_sym = self.fill_connection.dof.variables.position * A
        L_sym = sm.sqrt((A - h_sym) ** 2 + r**2)
        phi_sym = sm.atan2(A - h_sym, r)
        gap_sym = sm.max(
            sm.Scalar(0.0),
            L_sym * sm.sin(self.tilt_connection.dof.variables.position - phi_sym),
        )
        return -self.k * gap_sym / A


class HasFillLevel:
    """
    Mixin that adds a virtual fill-level DOF to any semantic annotation.

    The fill level is represented as a virtual :class:`LiquidConnection` whose
    position encodes fill in the range ``[0, 1]``. Call :meth:`initialize_fill_level`
    once after the annotation is placed in a world.

    Optionally pass ``tilt_connection`` to :meth:`initialize_fill_level` to also
    create a :attr:`liquid_surface_connection` — a revolute joint that counter-rotates
    by the same DOF so the liquid surface body's XY plane stays parallel to the world
    XY plane regardless of cup tilt.

    After initialisation assign :attr:`fill_equation` (how the cup drains) and
    :attr:`inflow_equation` (how the cup fills from an external source).
    Assign :attr:`container_geometry` to provide physical dimensions for
    geometry-aware fill equations.
    """

    fill_connection: Optional[PrismaticConnection] = field(default=None, kw_only=True)
    """The virtual connection whose position encodes fill level in [0, 1]."""

    liquid_surface_connection: Optional[RevoluteConnection] = field(
        default=None, kw_only=True
    )
    """
    Counter-rotation connection whose child body is always world-frame aligned.

    Only present when ``tilt_connection`` was passed to :meth:`initialize_fill_level`.
    """

    container_geometry: Optional[ContainerGeometry] = field(default=None, kw_only=True)
    """Physical dimensions used by geometry-aware fill equations."""

    @property
    def fill_equation(self) -> Optional[PouringEquation]:
        """ODE governing how this container drains when tilted."""
        return self.__dict__.get("_fill_equation")

    @fill_equation.setter
    def fill_equation(self, equation: Optional[PouringEquation]) -> None:
        self.__dict__["_fill_equation"] = equation
        if self.fill_connection is not None:
            self.fill_connection.outflow_equation = equation

    @property
    def inflow_equation(self) -> Optional[InflowEquation]:
        """ODE governing how this container fills from an external source."""
        return self.__dict__.get("_inflow_equation")

    @inflow_equation.setter
    def inflow_equation(self, equation: Optional[InflowEquation]) -> None:
        self.__dict__["_inflow_equation"] = equation
        if self.fill_connection is not None:
            self.fill_connection.inflow_equation = equation

    def initialize_fill_level(
        self,
        world,
        parent_body: Body,
        initial_fill: float = 1.0,
        tilt_connection: Optional[RevoluteConnection] = None,
    ) -> None:
        """
        Create the virtual fill-level DOF and attach it to the world.

        :param world: The world to add the fill-level DOF to.
        :param parent_body: The body the fill-level DOF is attached to.
        :param initial_fill: Starting fill level in [0, 1].
        :param tilt_connection: When provided, adds a liquid surface body whose
            orientation is always aligned with the world frame.
        """
        phantom = Body(name=PrefixedName(f"{parent_body.name.name}_fill_level_phantom"))
        with world.modify_world():
            world.add_body(phantom)
            connection = LiquidConnection.create_with_dofs(
                world=world,
                parent=parent_body,
                child=phantom,
                axis=Vector3(0, 0, 1),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=-1.0),
                    upper=DerivativeMap(position=1.0, velocity=1.0),
                ),
            )
            world.add_connection(connection)
        self.fill_connection = connection
        world.set_positions_1DOF_connection({connection: initial_fill})

        if tilt_connection is not None:
            surface_phantom = Body(
                name=PrefixedName(f"{parent_body.name.name}_liquid_surface_phantom")
            )
            with world.modify_world():
                world.add_body(surface_phantom)
                surface_connection = RevoluteConnection(
                    parent=phantom,
                    child=surface_phantom,
                    axis=tilt_connection.axis,
                    dof_id=tilt_connection.dof_id,
                    multiplier=-tilt_connection.multiplier,
                )
                world.add_connection(surface_connection)
            self.liquid_surface_connection = surface_connection

    @property
    def fill_level(self) -> float:
        """Current fill level in ``[0, 1]``."""
        return float(self.fill_connection.position)

    @property
    def liquid_surface_body(self) -> Body:
        """The body whose XY plane is always parallel to the world XY plane."""
        return self.liquid_surface_connection.child


@dataclass(eq=False)
class LiquidConnection(ActiveConnection1DOF, HasUpdateState):
    """
    Translating DOF representing the fill level of a container.

    Integrates :attr:`outflow_equation` and :attr:`inflow_equation` each tick.
    Either may be ``None``; the net velocity is their sum.
    """

    outflow_equation: Optional[PouringEquation] = field(
        default=None, kw_only=True, init=False
    )
    """ODE governing how liquid leaves this container (e.g. tilting to pour)."""

    inflow_equation: Optional[InflowEquation] = field(
        default=None, kw_only=True, init=False
    )
    """ODE governing how liquid enters this container from an external source."""

    def add_to_world(self, world):
        super().add_to_world(world)
        translation_axis = self.axis * self.dof.variables.position
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )

    def update_state(self, dt: float):
        state = self._world.state
        old_vel = state[self.dof.id].velocity
        old_acc = state[self.dof.id].acceleration
        velocity = sum(
            eq.symbolic_velocity().evaluate()[0]
            for eq in (self.outflow_equation, self.inflow_equation)
            if eq is not None
        )
        state[self.dof.id].velocity = velocity
        state[self.dof.id].acceleration = (velocity - old_vel) / dt
        state[self.dof.id].jerk = (state[self.dof.id].acceleration - old_acc) / dt
        state[self.dof.id].position = state[self.dof.id].position + velocity * dt


@dataclass
class InflowEquation(DifferentialEquation):
    """
    Fill-level ODE for a container receiving liquid from a source.

    Converts the source's outflow volume rate to a normalised fill velocity
    for this container using its own cross-sectional geometry.

    :param container_geometry: Physical dimensions of the receiving container.
    :param source: The container currently pouring into this one.
    """

    container_geometry: ContainerGeometry
    source: Optional[HasFillLevel] = field(default=None)

    def symbolic_velocity(self) -> Scalar:
        """
        :return: Normalised fill velocity from inflow, or zero when no source is active.
        """
        if self.source is None:
            return sm.Scalar(0.0)
        source_outflow = sm.max(
            sm.Scalar(0.0), -self.source.fill_equation.symbolic_velocity()
        )
        source_volume = (
            self.source.container_geometry.half_width
            * self.source.container_geometry.height
        )
        receiver_volume = (
            self.container_geometry.half_width * self.container_geometry.height
        )
        return source_outflow * source_volume / receiver_volume
