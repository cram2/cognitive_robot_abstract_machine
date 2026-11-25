"""
Tests of the fill-level containers, their transfer coupling and the liquid connection.
"""

# %% imports

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    FillEquation,
    GatedArticulatedPouringEquation,
    GatedInflowEquation,
    InflowEquation,
    SymbolicFillContext,
    TransferGate,
)
from semantic_digital_twin.physics.liquid_transfer import LiquidTransferSettings
from semantic_digital_twin.semantic_annotations.exceptions import (
    FillLevelAlreadyInitializedError,
    MissingFillEquationError,
    ReceiverAlreadyCoupledError,
    ReceiverNotInitializedError,
    SourceAlreadyCoupledError,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasFillLevel,
    LiquidSource,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.liquid_connection import LiquidConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.derivatives import (
    DerivativeMap,
    Derivatives,
)
from semantic_digital_twin.world_description.exceptions import (
    MissingFillLevelLimitsError,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% helpers


@pytest.fixture
def inflow_context() -> SymbolicFillContext:
    """
    A context for inflow equations, whose velocity does not depend on it.
    """
    return SymbolicFillContext(tilt=sm.Scalar(0.0), fill=sm.Scalar(0.0))


def _evaluated_xyz(spatial: Point3 | Vector3) -> np.ndarray:
    """
    The numeric ``x, y, z`` of a symbolic point or vector at the world's current state.
    """
    return np.array(
        [spatial.x.evaluate()[0], spatial.y.evaluate()[0], spatial.z.evaluate()[0]]
    )


def _container_volume(container: HasFillLevel) -> float:
    """
    The liquid volume of a container at full fill, read off its collision geometry.
    """
    extents = container.root.collision.scale
    return extents.x * extents.y * extents.z


def _horizontal_unit(direction: np.ndarray) -> np.ndarray:
    """
    The horizontal part of a numeric direction, scaled to unit length.
    """
    return direction[:2] / np.linalg.norm(direction[:2])


# %% test containers and sources


@dataclass(eq=False, kw_only=True)
class _TranslatingContainer(HasFillLevel):
    """
    A pourable container attached to its parent by a single translating degree of
    freedom.
    """

    @classmethod
    def parent_connection_specification(
        cls,
        axis: Vector3 | None = None,
        dof_limits: DegreeOfFreedomLimits | None = None,
    ) -> PrismaticConnectionSpecification:
        """
        :param axis: Axis of the connection. Defaults to the z axis.
        :param dof_limits: Limits of the generated degree of freedom.
        :return: The connection the container hangs from.
        """
        return PrismaticConnectionSpecification(
            axis=axis if axis is not None else Vector3.Z(), dof_limits=dof_limits
        )


@dataclass(eq=False, kw_only=True)
class _TiltingContainer(HasFillLevel):
    """
    A pourable container attached to its parent by a single tilting degree of freedom.
    """

    @classmethod
    def parent_connection_specification(
        cls,
        axis: Vector3 | None = None,
        dof_limits: DegreeOfFreedomLimits | None = None,
    ) -> RevoluteConnectionSpecification:
        """
        :param axis: Axis of the connection. Defaults to the z axis.
        :param dof_limits: Limits of the generated degree of freedom.
        :return: The connection the container hangs from.
        """
        return RevoluteConnectionSpecification(
            axis=axis if axis is not None else Vector3.Z(), dof_limits=dof_limits
        )


@dataclass(kw_only=True)
class _StaticLiquidSource(LiquidSource):
    """
    A liquid source that is not a cup (a faucet stand-in), with a fixed exit point and a
    constant stream.
    """

    exit_point: Point3
    """
    World-frame point at which the stream leaves the source.
    """

    volume_rate: float
    """
    Constant volume rate of the stream, in cubic metres per second.
    """

    def outflow_volume_rate(self, *, world: World) -> sm.Scalar:
        """
        :return: The constant volume rate of the stream.
        """
        return sm.Scalar(self.volume_rate)

    def liquid_exit_point(self, *, world: World) -> Point3:
        """
        :return: The fixed exit point.
        """
        return self.exit_point

    def opening_direction(self, *, world: World) -> Vector3:
        """
        :return: Straight down, the way the stream leaves.
        """
        return Vector3(x=0.0, y=0.0, z=-1.0, reference_frame=world.root)

    @property
    def pour_tilt_expression(self) -> sm.Scalar:
        """
        :return: Zero, as the source does not tilt.
        """
        return sm.Scalar(0.0)

    def couple_drain_to_gate(self, *, gate: sm.Scalar, world: World) -> None:
        """
        The reservoir is infinite, so being gated does not change the source.
        """

    def validate_can_pour(self) -> None:
        """
        A static source is always ready to pour.
        """


# %% world builders


def _wide_limits() -> DegreeOfFreedomLimits:
    """
    Limits wide enough that no test motion of a container reaches them.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=-2.0, velocity=-1.0),
        upper=DerivativeMap(position=2.0, velocity=1.0),
    )


def _world_with_root() -> World:
    """
    An empty world holding only its root body.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    return world


def _add_container(
    *,
    world: World,
    container_class: type[HasFillLevel],
    name: str,
    axis: Vector3,
    scale: Scale | None = None,
    world_root_T_self: HomogeneousTransformationMatrix | None = None,
) -> HasFillLevel:
    """
    :param world: The world to add the container to.
    :param container_class: The container annotation type.
    :param name: The name of the container.
    :param axis: Axis of the container's single degree of freedom.
    :param scale: Extents of the container's geometry; a 0.1 x 0.1 x 0.2 m box if not
        given.
    :param world_root_T_self: Pose of the container in the world; the origin if not
        given.
    :return: A container without fill level, on one degree of freedom with
        :func:`_wide_limits`.
    """
    with world.modify_world():
        return container_class.create_with_new_body_in_world(
            name=name,
            world=world,
            world_root_T_self=world_root_T_self,
            parent_connection_specification=container_class.parent_connection_specification(
                axis=axis, dof_limits=_wide_limits()
            ),
            scale=scale if scale is not None else Scale(0.1, 0.1, 0.2),
        )


def _lone_tilting_cup() -> tuple[World, _TiltingContainer]:
    """
    A world holding a single cup without fill level that tilts about y.
    """
    world = _world_with_root()
    cup = _add_container(
        world=world,
        container_class=_TiltingContainer,
        name="cup",
        axis=Vector3(0, 1, 0),
    )
    return world, cup


def _build_world(
    *,
    source_class: type[HasFillLevel] = _TranslatingContainer,
    source_axis: Vector3 | None = None,
    source_height: float = 0.3,
    receiver_height: float = 0.2,
    receiver_depth: float = 0.1,
    scale_factor: float = 1.0,
    couple: bool = True,
    initialize_receiver_fill: bool = True,
    settings: LiquidTransferSettings = LiquidTransferSettings(),
) -> tuple[World, HasFillLevel, _TranslatingContainer]:
    """
    Builds a fixed receiver at the origin and a source held above it on a single degree
    of freedom.

    :param source_class: The container annotation type used for the source.
    :param source_axis: Axis of the source's degree of freedom; defaults to x.
    :param source_height: Height of the source's origin above the world root, in metres.
    :param receiver_height: Height of the receiver's collision geometry, in metres.
    :param receiver_depth: Depth of the receiver's collision geometry, in metres, which
        changes its capacity but not the width its pouring model reads.
    :param scale_factor: Factor every length in the scene is multiplied by.
    :param couple: Whether the receiver's inflow is coupled to the source's outflow.
    :param initialize_receiver_fill: Whether the receiver's fill level is initialized.
    :param settings: The transfer settings forwarded to the coupling.
    :return: The world, the source, and the receiver.
    """
    if source_axis is None:
        source_axis = Vector3(1, 0, 0)
    world = _world_with_root()
    receiver = _add_container(
        world=world,
        container_class=_TranslatingContainer,
        name="receiver",
        axis=Vector3(1, 0, 0),
        scale=Scale(
            receiver_depth * scale_factor,
            0.1 * scale_factor,
            receiver_height * scale_factor,
        ),
    )
    source = _add_container(
        world=world,
        container_class=source_class,
        name="source",
        axis=source_axis,
        scale=Scale(0.1 * scale_factor, 0.1 * scale_factor, 0.2 * scale_factor),
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=source_height * scale_factor
        ),
    )
    if initialize_receiver_fill:
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
    source.initialize_fill_level(world=world, initial_fill=1.0)
    if couple:
        receiver.receive_outflow_from(source=source, world=world, settings=settings)
    return world, source, receiver


def _set_source_offset(world: World, source: HasFillLevel, offset: float) -> None:
    """
    Sets the position of the source's single degree of freedom.
    """
    JointState.from_mapping({source.root.parent_connection: offset}).apply_to(world)


def _step_passive_state(world: World, dt: float) -> None:
    """
    Advance the world by ``dt`` with every commanded velocity at zero, so only the
    connections that integrate their own state move.
    """
    world.apply_control_commands(
        commands=np.zeros(len(world.state)), dt=dt, derivative=Derivatives.velocity
    )


# %% gated inflow equation


class TestGatedInflowEquation:
    """
    Validates the volume-conserving, gated inflow conversion.
    """

    def test_the_capacity_defaults_to_the_box_the_extents_describe(self):
        """
        An equation given only the two extents of the 2-D model reads the depth it does
        not carry as the width, so its capacity still grows with the square of the width
        as a real container's does.
        """
        equation = InflowEquation(container_height=0.2, container_width=0.08)
        assert equation.capacity == pytest.approx(0.08 * 0.08 * 0.2)

    def test_gate_scales_the_inflow_velocity(
        self, *, inflow_context: SymbolicFillContext
    ) -> None:
        """
        A half-open gate halves the resulting fill velocity.
        """
        inflow = sm.Scalar(0.006)
        open_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            transfer_gate=TransferGate(height=sm.Scalar(1.0), overlap=sm.Scalar(1.0)),
            exit_speed=LiquidTransferSettings().exit_speed,
        )
        half_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            transfer_gate=TransferGate(height=sm.Scalar(0.5), overlap=sm.Scalar(1.0)),
            exit_speed=LiquidTransferSettings().exit_speed,
        )
        assert half_equation.symbolic_velocity(context=inflow_context).evaluate()[
            0
        ] == pytest.approx(
            0.5 * open_equation.symbolic_velocity(context=inflow_context).evaluate()[0]
        )

    @pytest.mark.parametrize(
        "source_capacity, receiver_capacity",
        [(1e-3, 1e-3), (1e-3, 4e-4), (4e-4, 1e-3)],
    )
    def test_a_transfer_moves_the_share_the_capacities_relate(
        self,
        *,
        source_capacity: float,
        receiver_capacity: float,
        inflow_context: SymbolicFillContext,
    ) -> None:
        """
        The share of itself the receiver gains is the share the source lost, scaled by
        how the two capacities compare, so the volume leaving one is the volume entering
        the other whatever shape either has.
        """
        source = ArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            capacity=source_capacity,
            outflow_rate_constant=1.0,
        )
        source_drain = source.symbolic_velocity(
            context=SymbolicFillContext(tilt=sm.Scalar(1.3), fill=sm.Scalar(0.8))
        ).evaluate()[0]

        receiver = GatedInflowEquation(
            container_height=0.1,
            container_width=0.06,
            capacity=receiver_capacity,
            inflow=-source.symbolic_velocity(
                context=SymbolicFillContext(tilt=sm.Scalar(1.3), fill=sm.Scalar(0.8))
            )
            * source_capacity,
            transfer_gate=TransferGate(height=sm.Scalar(1.0), overlap=sm.Scalar(1.0)),
            exit_speed=LiquidTransferSettings().exit_speed,
        )

        assert receiver.symbolic_velocity(context=inflow_context).evaluate()[
            0
        ] == pytest.approx(-source_drain * source_capacity / receiver_capacity)


# %% the parts of a coupled inflow


class TestCoupledInflowParts:
    """
    The inflow a coupling builds is its gate, the product of a height and an overlap
    factor, times the source's volume rate over the receiver's capacity.
    """

    def _partly_open_transfer(self) -> tuple[World, HasFillLevel, HasFillLevel]:
        """
        A coupled transfer with soft gates and the source tilted so both gate factors
        lie inside their transitions.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            settings=LiquidTransferSettings(
                height_gate_sharpness=1.0, overlap_gate_sharpness=1.0
            ),
        )
        _set_source_offset(world, source, 0.5)
        return world, source, receiver

    def test_the_gate_is_the_product_of_its_factors(self) -> None:
        """
        The gate the inflow is scaled by is the height factor times the overlap factor.
        """
        _, _, receiver = self._partly_open_transfer()
        inflow_equation = receiver.fill_connection.inflow_equation

        assert inflow_equation.transfer_gate.product.evaluate()[0] == pytest.approx(
            inflow_equation.transfer_gate.height.evaluate()[0]
            * inflow_equation.transfer_gate.overlap.evaluate()[0]
        )

    def test_the_height_factor_reads_the_clearance_in_receiver_heights(self) -> None:
        """
        The height factor is the logistic of the lip's clearance above the opening,
        measured in receiver heights.
        """
        world, source, receiver = self._partly_open_transfer()
        clearance = (
            source.liquid_exit_point(world=world).z.evaluate()[0]
            - receiver.opening_point(world=world).z.evaluate()[0]
        )

        sharpness = receiver.inflow_coupling.settings.height_gate_sharpness

        height_gate = receiver.fill_connection.inflow_equation.transfer_gate.height

        assert height_gate.evaluate()[0] == pytest.approx(
            1 / (1 + math.exp(-sharpness * clearance / receiver.root.collision.scale.z))
        )

    def test_the_overlap_factor_reads_the_landing_distance_in_opening_radii(
        self,
    ) -> None:
        """
        The overlap factor is the logistic of the landing point's margin inside the
        opening, measured in opening radii.
        """
        world, source, receiver = self._partly_open_transfer()
        landing = _evaluated_xyz(
            receiver.projectile_landing_point(
                source=source,
                world=world,
                exit_speed=source.current_exit_speed(world=world),
            )
        )
        opening = _evaluated_xyz(receiver.opening_point(world=world))
        landing_distance = np.linalg.norm(landing[:2] - opening[:2])
        margin = (receiver.opening_radius - landing_distance) / receiver.opening_radius
        sharpness = receiver.inflow_coupling.settings.overlap_gate_sharpness

        overlap_gate = receiver.fill_connection.inflow_equation.transfer_gate.overlap

        assert overlap_gate.evaluate()[0] == pytest.approx(
            1 / (1 + math.exp(-sharpness * margin))
        )

    def test_the_ungated_velocity_is_the_volume_rate_over_the_capacity(
        self, *, inflow_context: SymbolicFillContext
    ) -> None:
        """
        With the gate open the receiver fills at the source's volume rate over its own
        capacity.
        """
        world, source, receiver = self._partly_open_transfer()
        inflow_equation = receiver.fill_connection.inflow_equation

        assert inflow_equation.ungated_velocity(context=inflow_context).evaluate()[
            0
        ] == pytest.approx(
            source.outflow_volume_rate(world=world).evaluate()[0] / receiver.capacity
        )


# %% a fill level's rate from all its equations


class TestSymbolicFillVelocity:
    """
    A fill connection's rate sums its drain and its inflow.
    """

    def test_a_receiver_fills_at_its_inflow(self):
        """
        A receiver without a drain changes at the rate its inflow equation gives.
        """
        world, _source, receiver = _build_world()
        connection = receiver.fill_connection
        assert connection.symbolic_fill_velocity().evaluate()[0] == pytest.approx(
            connection.inflow_equation.symbolic_velocity(context=connection).evaluate()[
                0
            ]
        )

    def test_a_source_drains_at_its_outflow(self):
        """
        A source without an inflow changes at the rate its drain equation gives.
        """
        world, source, _receiver = _build_world()
        connection = source.fill_connection
        assert connection.symbolic_fill_velocity().evaluate()[0] == pytest.approx(
            connection.outflow_equation.symbolic_velocity(
                context=connection
            ).evaluate()[0]
        )

    def test_a_connection_without_equations_stands_still(self):
        """
        A fill level with neither equation does not change.
        """
        world, _source, receiver = _build_world(couple=False)
        assert receiver.fill_connection.symbolic_fill_velocity().evaluate()[0] == 0.0


# %% gated source outflow


class TestGatedSourceOutflow:
    """
    Validates the gated source outflow that makes a controlled pour spill- free.
    """

    def test_closed_gate_stops_the_source_draining(self):
        """
        A closed gate zeroes the source outflow, so the source never spills while
        mispositioned.
        """
        equation = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=sm.Scalar(0.0),
        )
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        assert equation.symbolic_velocity(
            context=SymbolicFillContext(tilt=tilt, fill=fill)
        ).evaluate()[0] == pytest.approx(0.0)

    def test_open_gate_matches_ungated_outflow(self):
        """
        A fully open gate leaves the tilt-driven outflow unchanged.
        """
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        ungated = ArticulatedPouringEquation(
            container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
        )
        gated = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=sm.Scalar(1.0),
        )
        assert gated.symbolic_velocity(
            context=SymbolicFillContext(tilt=tilt, fill=fill)
        ).evaluate()[0] == pytest.approx(
            ungated.symbolic_velocity(
                context=SymbolicFillContext(tilt=tilt, fill=fill)
            ).evaluate()[0]
        )

    def test_partial_gate_transfer_is_conserved(
        self, *, inflow_context: SymbolicFillContext
    ) -> None:
        """
        At a partly open gate the source's gated loss equals the receiver's gated gain.
        """
        gate = sm.Scalar(0.5)
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        ungated = ArticulatedPouringEquation(
            container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
        )
        gated_source = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=gate,
        )
        receiver = GatedInflowEquation(
            container_height=0.2,
            container_width=0.08,
            inflow=-ungated.symbolic_velocity(
                context=SymbolicFillContext(tilt=tilt, fill=fill)
            )
            * ungated.capacity,
            transfer_gate=TransferGate(height=gate, overlap=sm.Scalar(1.0)),
            exit_speed=LiquidTransferSettings().exit_speed,
        )
        source_volume_loss = (
            -gated_source.symbolic_velocity(
                context=SymbolicFillContext(tilt=tilt, fill=fill)
            ).evaluate()[0]
            * gated_source.capacity
        )
        receiver_volume_gain = (
            receiver.symbolic_velocity(context=inflow_context).evaluate()[0]
            * receiver.capacity
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)


# %% cross-container volume conversion


class TestCrossContainerVolumeConversion:
    """
    Validates that a transfer moves the same liquid volume out of the source as it moves
    into the receiver, whatever the two containers' shapes.
    """

    def test_containers_of_different_depth_conserve_volume(self):
        """
        A receiver that holds four times the liquid of the source rises four times more
        slowly than the source drops; both rates carry the same gate, so their ratio is
        the volume conversion whatever the gate is worth.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            receiver_depth=0.4,
        )
        _set_source_offset(world, source, 0.6)

        source_fill_loss_rate = (
            -source.fill_connection.outflow_equation.symbolic_velocity(
                context=source.fill_connection
            ).evaluate()[0]
        )
        receiver_fill_gain_rate = (
            receiver.fill_connection.inflow_equation.symbolic_velocity(
                context=receiver.fill_connection
            ).evaluate()[0]
        )
        assert source_fill_loss_rate > 0.0, "the source must be draining"

        assert receiver_fill_gain_rate / source_fill_loss_rate == pytest.approx(
            _container_volume(source) / _container_volume(receiver)
        )

    def test_a_receiver_of_measured_capacity_converts_volume_by_it(self):
        """
        A receiver whose capacity was measured holds that much at full fill, whatever
        the box its collision geometry bounds would hold.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            couple=False,
            initialize_receiver_fill=False,
        )
        receiver.measured_capacity = 3.0 * _container_volume(receiver)
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
        receiver.receive_outflow_from(source=source, world=world)
        _set_source_offset(world, source, 0.6)

        source_fill_loss_rate = (
            -source.fill_connection.outflow_equation.symbolic_velocity(
                context=source.fill_connection
            ).evaluate()[0]
        )
        receiver_fill_gain_rate = (
            receiver.fill_connection.inflow_equation.symbolic_velocity(
                context=receiver.fill_connection
            ).evaluate()[0]
        )
        assert receiver_fill_gain_rate / source_fill_loss_rate == pytest.approx(
            _container_volume(source) / receiver.measured_capacity
        )

    def test_a_measured_capacity_is_the_capacity(self) -> None:
        """
        A measured capacity takes the place of the one the collision geometry bounds.
        """
        _, _, receiver = _build_world(couple=False)
        receiver.measured_capacity = 0.002
        assert receiver.capacity == receiver.measured_capacity

    def test_a_measured_capacity_survives_a_json_round_trip(self):
        """
        The capacity has to reach the process that controls the pour, which rebuilds the
        transfer from the synchronized annotation.
        """
        world, _, receiver = _build_world(couple=False)
        receiver.measured_capacity = 0.002
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)

        restored = _TranslatingContainer.from_json(
            data=receiver.to_json(), **tracker.create_kwargs()
        )

        assert restored.measured_capacity == receiver.measured_capacity


# %% geometric transfer gate


class TestTransferGate:
    """Validates the differentiable geometric gate built by
    ``receive_outflow_from``."""

    def test_gate_is_open_when_source_is_above_receiver(self):
        """
        The gate is essentially fully open when the source is held directly over the
        receiver.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.0)
        gate = receiver.fill_connection.inflow_equation.transfer_gate.product
        assert gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_gate_closes_monotonically_with_horizontal_offset(self):
        """
        Moving the source sideways past the receiver opening closes the gate
        monotonically.
        """
        world, source, receiver = _build_world()
        gate = receiver.fill_connection.inflow_equation.transfer_gate.product
        offsets = [0.0, 0.05, 0.1, 0.3]
        values = []
        for offset in offsets:
            _set_source_offset(world, source, offset)
            values.append(gate.evaluate()[0])
        assert all(earlier >= later for earlier, later in zip(values, values[1:]))
        assert values[0] > values[-1]
        assert values[-1] < 0.1

    def test_gate_has_nonzero_gradient_in_transition(self):
        """
        The gate is differentiable: its slope with respect to the source position is
        non-zero at the rim.
        """
        world, source, receiver = _build_world()
        gate = receiver.fill_connection.inflow_equation.transfer_gate.product
        source_position = source.root.parent_connection.dof.variables.position
        _set_source_offset(world, source, 0.05)
        gradient = gate.jacobian(variables=[source_position])[0, 0].evaluate()[0]
        assert abs(gradient) > 1e-3

    def test_height_gate_is_unchanged_by_a_uniform_scene_scaling(self):
        """
        Whether the source's lip clears the receiver's opening depends on the shape of
        the arrangement alone, so two scenes that differ by a uniform scaling read the
        same height gate.
        """
        gates = []
        for scale_factor in (1.0, 4.0):
            world, source, receiver = _build_world(
                receiver_height=0.76,
                scale_factor=scale_factor,
            )
            _set_source_offset(world, source, 0.0)
            gates.append(
                receiver.fill_connection.inflow_equation.transfer_gate.height.evaluate()[
                    0
                ]
            )
        assert 0.01 < gates[0] < 0.99, (
            "the setup must hold the lip in the gate's transition, not at either end"
        )

        assert gates[1] == pytest.approx(gates[0])

    def test_gate_closes_when_source_tilts(self):
        """
        Tilting a source held in place sends the liquid's projectile past the receiver,
        closing the gate — the property that forces the optimizer to reposition the
        gripper while pouring.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate = receiver.fill_connection.inflow_equation.transfer_gate.product
        _set_source_offset(world, source, 0.0)
        upright_gate = gate.evaluate()[0]
        _set_source_offset(world, source, 1.2)
        tilted_gate = gate.evaluate()[0]
        assert upright_gate > 0.9
        assert tilted_gate < upright_gate
        assert tilted_gate < 0.5

    def test_gate_closes_when_the_source_lip_is_below_the_receiver_opening(self):
        """
        Liquid enters over the rim, so a receiver whose opening reaches above the
        source's lip cannot be poured into -- even while the lip is still well above the
        receiver's base.
        """
        world, source, receiver = _build_world(receiver_height=1.0)
        _set_source_offset(world, source, 0.0)
        source_lip = source.liquid_exit_point(world=world).z.evaluate()[0]
        receiver_base = world.compose_forward_kinematics_expression(
            world.root, receiver.root
        ).position.z.evaluate()[0]
        receiver_opening = receiver.opening_point(world=world).z.evaluate()[0]
        assert receiver_base < source_lip < receiver_opening, (
            "the setup must hold the lip above the receiver's base but below its opening"
        )

        gate = receiver.fill_connection.inflow_equation.transfer_gate.product

        assert gate.evaluate()[0] < 0.1


# %% live exit speed


class TestCurrentExitSpeed:
    """
    Validates the discharge-scaled Torricelli exit speed derived from the live pour
    head.
    """

    def _tilted_source(
        self,
        *,
        tilt: float,
        discharge_coefficient: float = ArticulatedPouringEquation.discharge_coefficient,
    ) -> tuple[World, HasFillLevel]:
        """
        A source tilted by ``tilt`` that drains with the given discharge coefficient.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        source.fill_equation.discharge_coefficient = discharge_coefficient
        _set_source_offset(world, source, tilt)
        return world, source

    def test_matches_discharge_scaled_torricelli(self):
        """
        The exit speed is ``C_d * sqrt(2 g h_head)`` for the current head above the lip.
        """
        world, source = self._tilted_source(tilt=1.2, discharge_coefficient=0.4)
        head = source.fill_equation.head_above_lip(
            context=source.fill_connection
        ).evaluate()[0]
        velocity = source.current_exit_speed(world=world)
        assert velocity is not None
        assert velocity.evaluate()[0] == pytest.approx(
            0.4 * math.sqrt(2 * FillEquation.standard_gravity * head)
        )

    def test_discharge_coefficient_scales_speed_linearly(self):
        """
        The discharge coefficient scales the exit speed proportionally.
        """
        low_world, low = self._tilted_source(tilt=1.2, discharge_coefficient=0.2)
        high_world, high = self._tilted_source(tilt=1.2, discharge_coefficient=0.6)
        ratio = (
            high.current_exit_speed(world=high_world).evaluate()[0]
            / low.current_exit_speed(world=low_world).evaluate()[0]
        )
        assert ratio == pytest.approx(3.0)

    def test_more_tilt_pours_faster(self):
        """
        A steeper tilt lifts more liquid above the lip, so the exit speed grows.
        """
        gentle_world, gentle = self._tilted_source(tilt=0.6)
        steep_world, steep = self._tilted_source(tilt=1.2)
        assert (
            steep.current_exit_speed(world=steep_world).evaluate()[0]
            > gentle.current_exit_speed(world=gentle_world).evaluate()[0]
        )

    def test_head_is_floored_when_barely_pouring(self):
        """
        With no head above the lip the head is floored so the exit-speed gradient stays
        finite.
        """
        world, source = self._tilted_source(tilt=0.0, discharge_coefficient=0.4)
        assert source.current_exit_speed(world=world).evaluate()[0] == pytest.approx(
            0.4
            * math.sqrt(
                2 * FillEquation.standard_gravity * source.fill_equation.minimum_head
            )
        )

    def test_source_without_pour_head_has_no_velocity(self):
        """
        A source whose dynamics expose no head reports no exit speed.
        """
        source = _StaticLiquidSource(exit_point=Point3(), volume_rate=0.01)
        assert source.current_exit_speed(world=World()) is None

    def test_initialize_fill_level_threads_discharge_coefficient(self):
        """
        The discharge coefficient passed to ``initialize_fill_level`` reaches the pour
        model.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(
            world=world, initial_fill=1.0, discharge_coefficient=0.5
        )
        assert cup.fill_equation.discharge_coefficient == 0.5

    def test_initialize_fill_level_threads_repose_angle(self):
        """
        The repose angle passed to ``initialize_fill_level`` reaches the drain the fill
        connection integrates.
        """
        world, cup = _lone_tilting_cup()
        repose_angle = 0.4
        cup.initialize_fill_level(
            world=world, initial_fill=1.0, repose_angle=repose_angle
        )
        assert cup.fill_connection.outflow_equation.repose_angle == repose_angle


# %% liquid exit point


class TestLiquidExitPoint:
    """
    Validates that liquid leaves from the rim edge on the pour side, not the rim centre.
    """

    def _horizontal_half_extent(self, *, source: HasFillLevel) -> float:
        """
        Half the source's extent along its own x axis, the rim radius the exit point
        moves out to.
        """
        collision = source.root.collision
        return (collision.max_point.x - collision.min_point.x) / 2

    def _rim_center_world(self, *, world: World, source: HasFillLevel) -> Point3:
        """
        The centre of the source's rim in the world frame.
        """
        return (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )

    def test_exit_point_at_rim_centre_when_upright(self):
        """
        An upright cup has no pour direction, so the exit point stays at the rim centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.0)

        exit_point = source.liquid_exit_point(world=world)
        rim_center = self._rim_center_world(world=world, source=source)

        assert exit_point.x.evaluate()[0] == pytest.approx(
            rim_center.x.evaluate()[0], abs=1e-3
        )
        assert exit_point.y.evaluate()[0] == pytest.approx(
            rim_center.y.evaluate()[0], abs=1e-3
        )

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self):
        """
        A tilted cup pours over the rim edge on the pour side, a full rim radius from
        centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 1.2)

        exit_point = source.liquid_exit_point(world=world)
        rim_center = self._rim_center_world(world=world, source=source)
        offset = exit_point - rim_center
        offset_x = offset.x.evaluate()[0]
        offset_y = offset.y.evaluate()[0]
        offset_z = offset.z.evaluate()[0]

        distance = math.sqrt(offset_x**2 + offset_y**2 + offset_z**2)
        assert distance == pytest.approx(
            self._horizontal_half_extent(source=source), abs=1e-3
        )

        pour_direction = source.opening_direction(world=world)
        horizontal_alignment = (
            offset_x * pour_direction.x.evaluate()[0]
            + offset_y * pour_direction.y.evaluate()[0]
        )
        assert horizontal_alignment > 0


# %% projectile landing point


class TestProjectileLandingPoint:
    """
    Validates the projectile model that locates where poured liquid lands.
    """

    def test_rim_source_launches_horizontally(self) -> None:
        """
        Liquid spilling over a rim has no vertical launch: it travels towards where the
        opening faces, horizontally.
        """
        world, source, _receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)

        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(world=world, exit_speed=0.2)
        )

        opening_direction = _evaluated_xyz(source.opening_direction(world=world))
        assert exit_velocity[2] == 0.0
        assert _horizontal_unit(exit_velocity) == pytest.approx(
            _horizontal_unit(opening_direction)
        )

    def test_a_rim_source_launches_at_the_horizontal_share_of_the_exit_speed(
        self,
    ) -> None:
        """
        The stream's horizontal speed is the exit speed scaled by how far the opening
        leans from vertical.
        """
        world, source, _receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)
        exit_speed = 0.2

        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(world=world, exit_speed=exit_speed)
        )

        opening_direction = _evaluated_xyz(source.opening_direction(world=world))
        assert exit_velocity[:2] == pytest.approx(exit_speed * opening_direction[:2])

    def test_upright_source_lands_below_its_rim(self):
        """
        With no tilt the liquid has no horizontal velocity, so it lands directly below
        the rim.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)
        landing = receiver.projectile_landing_point(
            source=source, world=world, exit_speed=0.2
        )
        source_rim = (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )
        assert landing.x.evaluate()[0] == pytest.approx(
            source_rim.x.evaluate()[0], abs=1e-3
        )
        assert landing.y.evaluate()[0] == pytest.approx(
            source_rim.y.evaluate()[0], abs=1e-3
        )

    def test_tilting_moves_landing_forward(self):
        """
        Tilting the source gives the liquid horizontal velocity, moving the landing
        forward.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.0)
        upright_landing = receiver.projectile_landing_point(
            source=source, world=world, exit_speed=0.2
        ).x.evaluate()[0]
        _set_source_offset(world, source, 0.6)
        tilted_landing = receiver.projectile_landing_point(
            source=source, world=world, exit_speed=0.2
        ).x.evaluate()[0]
        assert tilted_landing > upright_landing

    def test_higher_source_lands_farther(self):
        """
        A higher source gives the liquid a longer flight time, so it lands farther
        forward.
        """
        low_world, low_source, low_receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.3,
        )
        high_world, high_source, high_receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.7,
        )
        JointState.from_mapping({low_source.root.parent_connection: 0.5}).apply_to(
            low_world
        )
        JointState.from_mapping({high_source.root.parent_connection: 0.5}).apply_to(
            high_world
        )
        low_landing = low_receiver.projectile_landing_point(
            source=low_source, world=low_world, exit_speed=0.2
        ).x.evaluate()[0]
        high_landing = high_receiver.projectile_landing_point(
            source=high_source, world=high_world, exit_speed=0.2
        ).x.evaluate()[0]
        assert high_landing > low_landing

    def test_arc_terminates_at_the_opening_not_the_base(self):
        """
        The arc ends where the liquid arrives, the receiver's opening, not the
        receiver's origin at its base.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)

        landing = receiver.projectile_landing_point(
            source=source, world=world, exit_speed=0.2
        )

        assert landing.z.evaluate()[0] == pytest.approx(
            receiver.opening_point(world=world).z.evaluate()[0], abs=1e-9
        )

    def test_throw_distance_follows_the_fall_to_the_opening(self):
        """
        The horizontal throw is the exit velocity carried over the time the liquid takes
        to fall from the source's lip to the receiver's opening.
        """
        exit_speed = 0.2
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)

        landing = receiver.projectile_landing_point(
            source=source, world=world, exit_speed=exit_speed
        )

        exit_point = source.liquid_exit_point(world=world)
        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(world=world, exit_speed=exit_speed)
        )
        fall_height = (
            exit_point.z.evaluate()[0]
            - receiver.opening_point(world=world).z.evaluate()[0]
        )
        flight_time = math.sqrt(2 * fall_height / FillEquation.standard_gravity)
        expected_x = exit_point.x.evaluate()[0] + exit_velocity[0] * flight_time
        assert landing.x.evaluate()[0] == pytest.approx(expected_x, abs=1e-9)


# %% coupling guards


class TestReceiveOutflowGuard:
    """
    Validates the guards protecting ``receive_outflow_from`` against illegal couplings.
    """

    def test_raises_when_source_has_no_fill_equation(self):
        """
        Coupling from a source that was never initialized raises a meaningful error.
        """
        source = _TranslatingContainer(
            name=PrefixedName("dry_source"), root=Body(name=PrefixedName("dry_source"))
        )
        receiver = _TranslatingContainer(
            name=PrefixedName("receiver"), root=Body(name=PrefixedName("receiver"))
        )
        with pytest.raises(MissingFillEquationError):
            receiver.receive_outflow_from(source=source, world=World())

    def test_raises_when_source_already_coupled(self):
        """
        Coupling a source whose outflow is already gated onto another receiver raises.
        """
        world, source, receiver = _build_world()
        second_receiver = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="second_receiver",
            axis=Vector3(1, 0, 0),
        )
        second_receiver.initialize_fill_level(world=world, initial_fill=0.0)
        with pytest.raises(SourceAlreadyCoupledError):
            second_receiver.receive_outflow_from(source=source, world=world)

    def test_raises_when_receiver_has_no_fill_connection(self):
        """
        Coupling into a receiver that was never initialized blames the receiver, not the
        source.
        """
        world, source, receiver = _build_world(
            couple=False, initialize_receiver_fill=False
        )
        with pytest.raises(ReceiverNotInitializedError) as error_info:
            receiver.receive_outflow_from(source=source, world=world)
        assert error_info.value.receiver is receiver

    def test_raises_when_receiver_already_coupled(self):
        """
        Coupling a second source into an already-coupled receiver raises instead of
        silently overwriting the first source's transfer.
        """
        world, source, receiver = _build_world()
        second_source = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="second_source",
            axis=Vector3(1, 0, 0),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3),
        )
        second_source.initialize_fill_level(world=world, initial_fill=1.0)
        first_inflow_equation = receiver.fill_connection.inflow_equation
        with pytest.raises(ReceiverAlreadyCoupledError) as error_info:
            receiver.receive_outflow_from(source=second_source, world=world)
        assert error_info.value.receiver is receiver
        assert receiver.fill_connection.inflow_equation is first_inflow_equation


# %% fill-level initialization guard


class TestFillLevelInitializationGuard:
    """
    Validates that a container's fill level can only be initialized once.
    """

    def test_second_initialization_raises(self):
        """
        A second ``initialize_fill_level`` call raises instead of stacking a second
        phantom body and fill connection onto the container.
        """
        world, source, receiver = _build_world(couple=False)
        first_fill_connection = source.fill_connection
        with pytest.raises(FillLevelAlreadyInitializedError) as error_info:
            source.initialize_fill_level(world=world, initial_fill=1.0)
        assert error_info.value.container is source
        assert source.fill_connection is first_fill_connection


# %% coupling reconstruction


class TestCouplingReconstruction:
    """
    A transfer coupling travels as a serializable
    :class:`~semantic_digital_twin.physics.liquid_transfer.LiquidTransferCoupling`, from
    which a world that lacks the symbolic inflow rebuilds it against its own symbols.
    """

    def _coupled_world(self) -> tuple[World, HasFillLevel, HasFillLevel]:
        """
        A world whose tilting source is coupled to the receiver below it.
        """
        return _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )

    def test_receive_outflow_records_serializable_descriptor(self):
        """
        The coupling descriptor names the source and survives a JSON round trip.
        """
        world, source, receiver = self._coupled_world()
        coupling = receiver.inflow_coupling
        assert coupling is not None
        assert coupling.source_id == source.id

        assert from_json(to_json(coupling)) == coupling

    def test_rebuild_reconstructs_inflow_when_symbolic_state_absent(self):
        """
        A receiver holding the coupling description but no symbolic inflow, as in a
        synchronized world, rebuilds an inflow bound to its own world.
        """
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world=world)

        inflow_equation = receiver.fill_connection.inflow_equation
        assert inflow_equation is not None
        # The gate is a symbolic function of the source's DOF in this world; evaluating it proves
        # the rebuilt coupling is bound to this world's symbols, not the ones it was first built in.
        JointState.from_mapping({source.root.parent_connection: 0.0}).apply_to(world)
        assert inflow_equation.transfer_gate.product.evaluate()[0] == pytest.approx(
            1.0, abs=1e-2
        )
        # The inflow tracks the source's outflow: zero while upright, its drain once tilted.
        assert inflow_equation.inflow.evaluate()[0] == pytest.approx(0.0)
        JointState.from_mapping({source.root.parent_connection: 1.0}).apply_to(world)
        assert inflow_equation.inflow.evaluate()[0] == pytest.approx(
            source.outflow_volume_rate(world=world).evaluate()[0]
        )

    def test_a_rebuilt_coupling_moves_the_volume_the_source_loses(self) -> None:
        """
        Rebuilding the coupling of a source whose drain is already gated feeds the
        receiver the volume the source loses, not that volume gated a second time.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            settings=LiquidTransferSettings(
                height_gate_sharpness=1.0, overlap_gate_sharpness=1.0
            ),
        )
        _set_source_offset(world, source, 0.5)
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world=world)

        gate = (
            receiver.fill_connection.inflow_equation.transfer_gate.product.evaluate()[0]
        )
        assert 0.01 < gate < 0.99, "the setup must hold the gate partly open"
        source_volume_loss = (
            -source.fill_connection.symbolic_fill_velocity().evaluate()[0]
            * source.capacity
        )
        receiver_volume_gain = (
            receiver.fill_connection.symbolic_fill_velocity().evaluate()[0]
            * receiver.capacity
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)

    def test_rebuild_regates_source_outflow(self):
        """
        Rebuilding re-establishes the source's gated outflow, keeping the transfer
        spill-free.
        """
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world=world)

        source_outflow = source.fill_connection.outflow_equation
        assert isinstance(source_outflow, GatedArticulatedPouringEquation)
        JointState.from_mapping({source.root.parent_connection: 0.0}).apply_to(world)
        assert source_outflow.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_rebuild_is_noop_when_inflow_already_present(self):
        """
        A receiver already carrying a symbolic inflow equation is left untouched.
        """
        world, source, receiver = self._coupled_world()
        original_inflow = receiver.fill_connection.inflow_equation
        assert original_inflow is not None

        receiver.ensure_inflow_coupling(world=world)

        assert receiver.fill_connection.inflow_equation is original_inflow

    def test_rebuild_rebinds_fill_connection_to_the_target_world(self):
        """
        An annotation holding a detached copy of its fill connection, as a synchronized
        one does, is re-pointed at the connection resident in the world, so the coupling
        evaluates against the world's own symbols.
        """
        world, source, receiver = self._coupled_world()
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        detached_connection = LiquidConnection.from_json(
            data=receiver.fill_connection.to_json(), **tracker.create_kwargs()
        )
        assert detached_connection._world is not world
        synchronized_receiver = HasFillLevel(
            name=receiver.name,
            root=receiver.root,
            fill_connection=detached_connection,
            inflow_coupling=receiver.inflow_coupling,
        )

        synchronized_receiver.ensure_inflow_coupling(world=world)

        assert synchronized_receiver.fill_connection is receiver.fill_connection
        assert synchronized_receiver.fill_connection._world is world


# %% coupling model switch


class TestCouplingModelSwitch:
    """
    A new ungated drain published for the source makes the symbolic coupling another
    process built stale, so it is rebuilt from the new drain.
    """

    def _coupled_world(self) -> tuple[World, HasFillLevel, HasFillLevel]:
        """
        A world whose tilting source is coupled to the receiver below it.
        """
        return _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )

    def _drain_swapped_like_a_synchronized_diff(
        self, *, source: HasFillLevel
    ) -> ArticulatedPouringEquation:
        """
        Assign the source a fresh ungated drain of doubled outflow rate, as a
        synchronized attribute diff does.
        """
        swapped = ArticulatedPouringEquation(
            container_height=source.fill_equation.container_height,
            container_width=source.fill_equation.container_width,
            outflow_rate_constant=source.fill_equation.outflow_rate_constant * 2,
        )
        source.fill_equation = swapped
        return swapped

    def test_ensure_rebuilds_when_synchronized_source_equation_changed(self):
        """
        A synchronized source-equation swap marks the coupling stale: the next ensure
        call rebuilds the inflow and re-gates the drain from the new equation.
        """
        world, source, receiver = self._coupled_world()
        stale_inflow = receiver.fill_connection.inflow_equation
        swapped = self._drain_swapped_like_a_synchronized_diff(source=source)

        receiver.ensure_inflow_coupling(world=world)

        assert receiver.fill_connection.inflow_equation is not stale_inflow
        regated_drain = source.fill_equation
        assert isinstance(regated_drain, GatedArticulatedPouringEquation)
        assert regated_drain.outflow_rate_constant == swapped.outflow_rate_constant
        assert source.fill_connection.outflow_equation is regated_drain

    def test_a_synchronized_copy_of_the_same_drain_is_gated_again(self) -> None:
        """
        A synchronized world receives the drain without its gate; even with unchanged
        parameters the next ensure call gates it again.
        """
        world, source, receiver = self._coupled_world()
        source.fill_equation = source.fill_equation.drop_gate()

        receiver.ensure_inflow_coupling(world=world)

        assert isinstance(source.fill_equation, GatedArticulatedPouringEquation)
        assert source.fill_connection.outflow_equation is source.fill_equation

    def test_rebuilt_inflow_reflects_the_new_drain(self):
        """
        The rebuilt inflow is derived from the swapped drain: doubling the source's
        outflow rate constant doubles the inflow volume rate at the same pose.
        """
        world, source, receiver = self._coupled_world()
        stale_inflow = receiver.fill_connection.inflow_equation
        self._drain_swapped_like_a_synchronized_diff(source=source)

        receiver.ensure_inflow_coupling(world=world)

        JointState.from_mapping({source.root.parent_connection: 1.0}).apply_to(world)
        rebuilt_inflow = receiver.fill_connection.inflow_equation
        assert rebuilt_inflow.inflow.evaluate()[0] == pytest.approx(
            2 * stale_inflow.inflow.evaluate()[0]
        )

    def test_ensure_is_stable_after_a_rebuild(self):
        """
        A rebuild settles: ensuring again without another synchronized swap is a no-op.
        """
        world, source, receiver = self._coupled_world()
        self._drain_swapped_like_a_synchronized_diff(source=source)
        receiver.ensure_inflow_coupling(world=world)
        rebuilt_inflow = receiver.fill_connection.inflow_equation

        receiver.ensure_inflow_coupling(world=world)

        assert receiver.fill_connection.inflow_equation is rebuilt_inflow


# %% non-cup liquid source


class TestNonCupLiquidSource:
    """
    A receiver fills from a :class:`LiquidSource` that is not a :class:`HasFillLevel`
    cup.
    """

    def test_receiver_fills_from_static_source(
        self, *, inflow_context: SymbolicFillContext
    ) -> None:
        """
        A static faucet-like source with no fill level and no tilt above the receiver
        opens the gate and fills the receiver at its volume rate.
        """
        world = _world_with_root()
        receiver = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="receiver",
            axis=Vector3(1, 0, 0),
        )
        receiver.initialize_fill_level(world=world, initial_fill=0.0)

        source = _StaticLiquidSource(
            exit_point=Point3(x=0.0, y=0.0, z=0.5, reference_frame=world.root),
            volume_rate=0.001,
        )
        receiver.receive_outflow_from(source=source, world=world)

        inflow_equation = receiver.fill_connection.inflow_equation
        gate = inflow_equation.transfer_gate.product.evaluate()[0]
        assert gate == pytest.approx(1.0, abs=1e-2)
        assert inflow_equation.symbolic_velocity(context=inflow_context).evaluate()[
            0
        ] == pytest.approx(gate * source.volume_rate / receiver.capacity)


# %% container that receives and pours


class TestReceivingAndPouringContainer:
    """
    A container filled from one source can pour into another, so a synchronized change
    to it must not stumble over the process-local inflow equation it carries.
    """

    def test_coupled_receiver_serializes_without_its_symbolic_inflow(self) -> None:
        """
        The symbolic inflow stays in its process; the coupling description travels.
        """
        world, _source, receiver = _build_world()
        assert receiver.fill_connection.inflow_equation is not None

        payload = receiver.to_json()

        assert "inflow_equation" not in payload
        assert (
            HasFillLevel.from_json(
                data=payload,
                **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
            ).inflow_coupling
            == receiver.inflow_coupling
        )

    def test_receiver_can_pour_into_a_third_container(self) -> None:
        """
        A container that receives from a stream can be coupled as the source of a third
        container while it keeps its own inflow.
        """
        world, _source, can = _build_world(couple=False)
        stream = _StaticLiquidSource(
            exit_point=Point3(x=0.0, y=0.0, z=0.5, reference_frame=world.root),
            volume_rate=0.001,
        )
        can.receive_outflow_from(source=stream, world=world)
        pot = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="pot",
            axis=Vector3(1, 0, 0),
            scale=Scale(0.2, 0.2, 0.15),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
        )
        pot.initialize_fill_level(world=world, initial_fill=0.2)

        pot.receive_outflow_from(source=can, world=world)

        assert isinstance(can.fill_equation, GatedArticulatedPouringEquation)
        assert pot.inflow_coupling.source_id == can.id
        assert isinstance(pot.fill_connection.inflow_equation, GatedInflowEquation)
        assert can.fill_connection.inflow_equation is can.inflow_equation


# %% fill-level integration limits


class TestFillLevelIntegrationLimits:
    """
    Validates that physics integration keeps the fill level inside the DOF limits.
    """

    def test_tilted_empty_source_does_not_drain_below_empty(self):
        """
        A fully tilted cup with no liquid left must stay at fill 0, not integrate
        negative.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(world=world, initial_fill=0.0)
        JointState.from_mapping({cup.root.parent_connection: 1.5}).apply_to(world)

        for _ in range(50):
            _step_passive_state(world, 0.05)

        assert cup.fill_level == pytest.approx(0.0)

    def test_receiver_saturates_at_full(self):
        """
        A receiver with a steady inflow must stop at fill 1, not fill past the brim.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(world=world, initial_fill=0.9)
        with world.modify_world():
            cup.set_inflow_equation(
                inflow_equation=InflowEquation(
                    container_height=0.2,
                    container_width=0.1,
                    inflow=sm.Scalar(0.005),
                )
            )

        for _ in range(50):
            _step_passive_state(world, 0.05)

        assert cup.fill_level == pytest.approx(1.0)


# %% coupling serialization


class TestLiquidCouplingSerialization:
    """
    Validates that a serialized coupling never resurrects as a silently open transfer.
    """

    def test_serialized_drain_is_ungated(self):
        """
        The source drain crosses the process boundary ungated, so a deserialized world
        cannot carry an always-open gate; the receiving world re-gates it when
        rebuilding the coupling.
        """
        world, source, receiver = _build_world()
        assert isinstance(source.fill_equation, GatedArticulatedPouringEquation)

        payload = source.fill_connection.to_json()
        restored_drain = from_json(payload["outflow_equation"])

        assert restored_drain == source.fill_equation.drop_gate()

    def test_a_coupled_source_round_trips_ungated_and_ready_to_pour(self) -> None:
        """
        A coupled source annotation comes back with its drain ungated, so the world that
        reads it can couple it again instead of finding it already gated.
        """
        world, source, _receiver = _build_world()
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)

        restored_drain = from_json(
            source.to_json()["fill_equation"], **tracker.create_kwargs()
        )

        assert restored_drain == source.fill_equation.drop_gate()
        _TranslatingContainer(
            name=source.name,
            root=source.root,
            fill_connection=source.fill_connection,
            fill_equation=restored_drain,
        ).validate_can_pour()

    def test_symbolic_inflow_is_not_serialized(self):
        """
        The symbolic inflow cannot cross a process boundary; only the coupling
        descriptor does, so the receiving world rebuilds the inflow via
        ensure_inflow_coupling.
        """
        world, source, receiver = _build_world()
        assert receiver.fill_connection.inflow_equation is not None

        payload = receiver.fill_connection.to_json()

        assert "inflow_equation" not in payload


# %% fill-level connection builders


def build_world_with_liquid_connection(
    dof_limits: DegreeOfFreedomLimits | None = None,
) -> tuple[World, LiquidConnection]:
    """
    Build a world containing a container body with a fill-level connection.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    phantom = Body(name=PrefixedName("container_fill_level_phantom"))
    with world.modify_world():
        world.add_body(container)
    with world.modify_world():
        connection = LiquidConnection.create_with_dofs(
            world=world,
            parent=container,
            child=phantom,
            axis=Vector3.Z(),
            dof_limits=dof_limits,
        )
        world.add_connection(connection)
    return world, connection


def build_fill_level_limits() -> DegreeOfFreedomLimits:
    """
    The [0, 1] fill-level limits a fill DOF is normally created with.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=0.0, velocity=-1.0),
        upper=DerivativeMap(position=1.0, velocity=1.0),
    )


# %% passive fill degree of freedom


def test_fill_dof_is_passive() -> None:
    """
    The fill level is a passive degree of freedom the QP never commands.
    """
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    assert connection.active_dofs == []
    assert connection.passive_dofs == [connection.raw_dof]
    assert connection.raw_dof in world.passive_degrees_of_freedom
    assert connection.raw_dof not in world.active_degrees_of_freedom
    assert connection.is_controlled is False


# %% fill-level connection clamping


def test_update_state_without_position_limits_raises() -> None:
    """
    A fill level without position limits cannot be clamped, so stepping it raises.
    """
    _, connection = build_world_with_liquid_connection(dof_limits=None)
    with pytest.raises(MissingFillLevelLimitsError) as error_info:
        connection.update_state(0.05)
    assert error_info.value.connection_name == connection.name


# %% fill-level connection integration


def test_repeated_steps_compile_the_fill_velocity_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Stepping the fill level reuses the fill velocity compiled at the first step.
    """
    _, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = ArticulatedPouringEquation(
        container_height=0.2, container_width=0.1
    )
    connection.update_state(0.05)
    compilations = []
    original_compile = sm.Scalar.compile

    def counting_compile(expression: sm.Scalar, **kwargs) -> sm.CompiledFunction:
        """
        Compiles ``expression`` and records that it was compiled.
        """
        compilations.append(expression)
        return original_compile(expression, **kwargs)

    monkeypatch.setattr(sm.Scalar, "compile", counting_compile)
    for _ in range(3):
        connection.update_state(0.05)

    assert compilations == []


def test_a_new_inflow_takes_effect_at_the_next_step() -> None:
    """
    Replacing an equation of a fill level that was stepped before integrates the new one
    from the next step on.
    """
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.update_state(0.05)
    inflow_equation = InflowEquation(
        container_height=0.2, container_width=0.1, inflow=sm.Scalar(0.001)
    )
    connection.inflow_equation = inflow_equation
    dt = 0.05

    connection.update_state(dt)

    assert world.state[connection.raw_dof.id].position == pytest.approx(
        0.001 / inflow_equation.capacity * dt
    )


# %% fill-level connection serialization


def test_json_round_trip_restores_ungated_outflow() -> None:
    """
    A connection with a gated drain comes back with the same drain ungated and no
    inflow.
    """
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = GatedArticulatedPouringEquation(
        container_height=0.2,
        container_width=0.1,
        outflow_rate_constant=2.0,
        gate=sm.Scalar(0.5),
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        data=connection.to_json(), **tracker.create_kwargs()
    )

    assert type(restored) is LiquidConnection
    assert restored.outflow_equation == connection.outflow_equation.drop_gate()
    assert restored.inflow_equation is None
    assert restored.raw_dof.id == connection.raw_dof.id


def test_json_round_trip_without_outflow_equation() -> None:
    """
    A connection without equations comes back without equations.
    """
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        data=connection.to_json(), **tracker.create_kwargs()
    )
    assert restored.outflow_equation is None
    assert restored.inflow_equation is None


# %% fill-level connection copy


def test_copy_for_world_evaluates_against_its_own_world() -> None:
    """
    A connection copied into another world carries its drain ungated and no inflow, so
    nothing it integrates reads the original world's symbols.
    """
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    original_fill = connection.dof.variables.position
    connection.outflow_equation = GatedArticulatedPouringEquation(
        container_height=0.2, container_width=0.1, gate=original_fill
    )
    connection.inflow_equation = InflowEquation(
        container_height=0.2, container_width=0.1, inflow=original_fill
    )
    world_copy = deepcopy(world)

    copied_connection = connection.copy_for_world(world_copy)
    with world_copy.modify_world():
        world_copy.remove_connection(
            world_copy.get_connection(copied_connection.parent, copied_connection.child)
        )
        world_copy.add_connection(copied_connection)

    copy_fill = world_copy.get_degree_of_freedom_by_id(
        connection.raw_dof.id
    ).variables.position
    assert copied_connection.outflow_equation == connection.outflow_equation.drop_gate()
    assert copied_connection.inflow_equation is None
    assert copied_connection.symbolic_fill_velocity().free_variables() == [copy_fill]
