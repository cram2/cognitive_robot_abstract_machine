from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood.ormatic.utils import classproperty

from semantic_digital_twin.exceptions import (
    MissingFillEquationError,
    SourceAlreadyCoupledError,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    DEFAULT_POUR_EXIT_SPEED,
    GatedArticulatedPouringEquation,
    GatedInflowEquation,
    InflowEquation,
    STANDARD_GRAVITY,
    SymbolicFillContext,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

_INFLOW_CONTEXT = SymbolicFillContext(sm.Scalar(0.0), sm.Scalar(0.0))
"""Placeholder context for inflow equations, whose velocity does not depend on the context."""


@dataclass(eq=False)
class _TranslatingContainer(HasFillLevel):
    """A pourable container attached to its parent by a single translating DOF."""

    @classproperty
    def _parent_connection_type(self):
        return PrismaticConnection


@dataclass(eq=False)
class _TiltingContainer(HasFillLevel):
    """A pourable container attached to its parent by a single tilting DOF."""

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@dataclass
class _StaticLiquidSource(LiquidSource):
    """A non-cup liquid source (a faucet stand-in) with a fixed exit point and constant stream."""

    exit_point: Point3
    """World-frame point at which the stream leaves the source."""

    volume_rate: float
    """Constant volume rate of the stream, in cubic metres per second."""

    def outflow_volume_rate(self, world):
        return sm.Scalar(self.volume_rate)

    def liquid_exit_point(self, world):
        return self.exit_point

    def liquid_exit_direction(self, world):
        return Vector3.Z()

    @property
    def pour_tilt_expression(self):
        return sm.Scalar(0.0)

    def couple_drain_to_gate(self, gate, world):
        """The reservoir is infinite, so being gated does not change the source."""

    def validate_can_pour(self):
        """A static source is always ready to pour."""


class TestGatedInflowEquation:
    """Validates the volume-conserving, gated inflow conversion."""

    def test_cross_section_volume_matches_rectangular_area(self):
        """The 2-D cup volume is half-width times height."""
        equation = InflowEquation(container_height=0.2, container_width=0.08)
        assert equation.cross_section_volume == pytest.approx(0.04 * 0.2)

    def test_gate_scales_the_inflow_velocity(self):
        """A half-open gate halves the resulting fill velocity."""
        inflow = sm.Scalar(0.006)
        open_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            gate=sm.Scalar(1.0),
        )
        half_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            gate=sm.Scalar(0.5),
        )
        assert half_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[
            0
        ] == pytest.approx(
            0.5 * open_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
        )

    @pytest.mark.parametrize(
        "source_size, receiver_size",
        [((0.2, 0.08), (0.2, 0.08)), ((0.2, 0.08), (0.1, 0.06))],
    )
    def test_transfer_is_volume_conserving(self, source_size, receiver_size):
        """
        The volume the receiver gains per second equals the volume the source loses, for both
        equal and unequal cups, while the gate is fully open.
        """
        source_height, source_width = source_size
        receiver_height, receiver_width = receiver_size
        source = ArticulatedPouringEquation(
            container_height=source_height,
            container_width=source_width,
            outflow_rate_constant=1.0,
        )
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        source_normalized_loss = source.symbolic_velocity(
            SymbolicFillContext(tilt, fill)
        )
        source_volume_rate = -source_normalized_loss * source.cross_section_volume

        receiver = GatedInflowEquation(
            container_height=receiver_height,
            container_width=receiver_width,
            inflow=source_volume_rate,
            gate=sm.Scalar(1.0),
        )
        receiver_volume_gain = (
            receiver.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
            * receiver.cross_section_volume
        )
        source_volume_loss = (
            -source_normalized_loss.evaluate()[0] * source.cross_section_volume
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)


class TestGatedSourceOutflow:
    """Validates the gated source outflow that makes a controlled pour spill-free."""

    def test_closed_gate_stops_the_source_draining(self):
        """A closed gate zeroes the source outflow, so the source never spills while mispositioned."""
        equation = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=sm.Scalar(0.0),
        )
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        assert equation.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
            0
        ] == pytest.approx(0.0)

    def test_open_gate_matches_ungated_outflow(self):
        """A fully open gate leaves the tilt-driven outflow unchanged."""
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
        assert gated.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
            0
        ] == pytest.approx(
            ungated.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[0]
        )

    def test_partial_gate_transfer_is_conserved(self):
        """At a partly open gate the source's gated loss equals the receiver's gated gain."""
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
            inflow=-ungated.symbolic_velocity(SymbolicFillContext(tilt, fill))
            * ungated.cross_section_volume,
            gate=gate,
        )
        source_volume_loss = (
            -gated_source.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
                0
            ]
            * gated_source.cross_section_volume
        )
        receiver_volume_gain = (
            receiver.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
            * receiver.cross_section_volume
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)


class TestTransferGate:
    """Validates the differentiable geometric gate built by ``receive_outflow_from``."""

    def _build_world(
        self,
        source_class=_TranslatingContainer,
        source_axis=Vector3(1, 0, 0),
        source_height=0.3,
    ) -> tuple[World, HasFillLevel, _TranslatingContainer]:
        """Builds a fixed receiver at the origin and a source held above it on a single DOF."""
        wide_limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=-2.0, velocity=-1.0),
            upper=DerivativeMap(position=2.0, velocity=1.0),
        )
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))

        with world.modify_world():
            receiver = _TranslatingContainer.create_with_new_body_in_world(
                name=PrefixedName("receiver"),
                world=world,
                active_axis=Vector3(1, 0, 0),
                connection_limits=wide_limits,
                scale=Scale(0.1, 0.1, 0.2),
            )
            source = source_class.create_with_new_body_in_world(
                name=PrefixedName("source"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=source_height
                ),
                active_axis=source_axis,
                connection_limits=wide_limits,
                scale=Scale(0.1, 0.1, 0.2),
            )
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
        source.initialize_fill_level(world=world, initial_fill=1.0)
        receiver.receive_outflow_from(source=source, world=world)
        return world, source, receiver

    def _set_source_offset(
        self, world: World, source: _TranslatingContainer, offset: float
    ):
        world.set_positions_1DOF_connection({source.root.parent_connection: offset})

    def test_gate_is_open_when_source_is_above_receiver(self):
        """The gate is essentially fully open when the source is held directly over the receiver."""
        world, source, receiver = self._build_world()
        self._set_source_offset(world, source, 0.0)
        gate = receiver.fill_connection.inflow_equation.gate
        assert gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_gate_closes_monotonically_with_horizontal_offset(self):
        """Moving the source sideways past the receiver opening closes the gate monotonically."""
        world, source, receiver = self._build_world()
        gate = receiver.fill_connection.inflow_equation.gate
        offsets = [0.0, 0.05, 0.1, 0.3]
        values = []
        for offset in offsets:
            self._set_source_offset(world, source, offset)
            values.append(gate.evaluate()[0])
        assert all(earlier >= later for earlier, later in zip(values, values[1:]))
        assert values[0] > values[-1]
        assert values[-1] < 0.1

    def test_gate_has_nonzero_gradient_in_transition(self):
        """The gate is differentiable: its slope w.r.t. the source position is non-zero at the rim."""
        world, source, receiver = self._build_world()
        gate = receiver.fill_connection.inflow_equation.gate
        source_position = source.root.parent_connection.dof.variables.position
        self._set_source_offset(world, source, 0.05)
        gradient = gate.jacobian([source_position])[0, 0].evaluate()[0]
        assert abs(gradient) > 1e-3

    def test_gate_closes_when_source_tilts(self):
        """
        Tilting a source held in place sends the liquid's projectile past the receiver, closing
        the gate — the property that forces the optimizer to reposition the gripper while pouring.
        """
        world, source, receiver = self._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate = receiver.fill_connection.inflow_equation.gate
        self._set_source_offset(world, source, 0.0)
        upright_gate = gate.evaluate()[0]
        self._set_source_offset(world, source, 1.2)
        tilted_gate = gate.evaluate()[0]
        assert upright_gate > 0.9
        assert tilted_gate < upright_gate
        assert tilted_gate < 0.5


class TestCurrentOutflowVelocity:
    """Validates the Torricelli exit speed derived from the live pour head."""

    def _tilted_source(self, tilt: float) -> tuple[World, HasFillLevel]:
        gate_world = TestTransferGate()
        world, source, _ = gate_world._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate_world._set_source_offset(world, source, tilt)
        return world, source

    def test_matches_torricelli_of_pour_head(self):
        """The exit speed is ``sqrt(2 g h_head)`` for the current head above the pouring lip."""
        world, source = self._tilted_source(1.2)
        head = source.fill_equation.head_above_lip(source.fill_connection).evaluate()[0]
        velocity = source.current_outflow_velocity(world)
        assert velocity is not None
        assert velocity.evaluate()[0] == pytest.approx(
            math.sqrt(2 * STANDARD_GRAVITY * head)
        )
        assert velocity.evaluate()[0] > 0

    def test_more_tilt_pours_faster(self):
        """A steeper tilt lifts more liquid above the lip, so the exit speed grows."""
        gentle_world, gentle = self._tilted_source(0.6)
        steep_world, steep = self._tilted_source(1.2)
        assert (
            steep.current_outflow_velocity(steep_world).evaluate()[0]
            > gentle.current_outflow_velocity(gentle_world).evaluate()[0]
        )

    def test_floors_at_nominal_speed_when_barely_pouring(self):
        """With no head above the lip the speed floors at the nominal default, not zero."""
        world, source = self._tilted_source(0.0)
        assert source.current_outflow_velocity(world).evaluate()[0] == pytest.approx(
            DEFAULT_POUR_EXIT_SPEED
        )

    def test_source_without_pour_head_has_no_velocity(self):
        """A source whose dynamics expose no head reports no exit speed."""
        source = _StaticLiquidSource(exit_point=Point3(), volume_rate=0.01)
        assert source.current_outflow_velocity(world=None) is None


class TestLiquidExitPoint:
    """Validates that liquid leaves from the rim edge on the pour side, not the rim centre."""

    def _horizontal_half_extent(self, source: HasFillLevel) -> float:
        collision = source.root.collision
        return (collision.max_point.x - collision.min_point.x) / 2

    def _rim_center_world(self, world: World, source: HasFillLevel) -> Point3:
        return (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )

    def test_exit_point_at_rim_centre_when_upright(self):
        """An upright cup has no pour direction, so the exit point stays at the rim centre."""
        gate_world = TestTransferGate()
        world, source, _ = gate_world._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate_world._set_source_offset(world, source, 0.0)

        exit_point = source.liquid_exit_point(world)
        rim_center = self._rim_center_world(world, source)

        assert exit_point.x.evaluate()[0] == pytest.approx(
            rim_center.x.evaluate()[0], abs=1e-3
        )
        assert exit_point.y.evaluate()[0] == pytest.approx(
            rim_center.y.evaluate()[0], abs=1e-3
        )

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self):
        """A tilted cup pours over the rim edge on the pour side, a full rim radius from centre."""
        gate_world = TestTransferGate()
        world, source, _ = gate_world._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate_world._set_source_offset(world, source, 1.2)

        exit_point = source.liquid_exit_point(world)
        rim_center = self._rim_center_world(world, source)
        offset = exit_point - rim_center
        offset_x = offset.x.evaluate()[0]
        offset_y = offset.y.evaluate()[0]
        offset_z = offset.z.evaluate()[0]

        distance = math.sqrt(offset_x**2 + offset_y**2 + offset_z**2)
        assert distance == pytest.approx(self._horizontal_half_extent(source), abs=1e-3)

        pour_direction = source.liquid_exit_direction(world)
        horizontal_alignment = (
            offset_x * pour_direction.x.evaluate()[0]
            + offset_y * pour_direction.y.evaluate()[0]
        )
        assert horizontal_alignment > 0


class TestProjectileLandingPoint:
    """Validates the projectile model that locates where poured liquid lands."""

    def test_upright_source_lands_below_its_rim(self):
        """With no tilt the liquid has no horizontal velocity, so it lands directly below the rim."""
        gate_world = TestTransferGate()
        world, source, receiver = gate_world._build_world()
        gate_world._set_source_offset(world, source, 0.1)
        landing = receiver.projectile_landing_point(source, world, exit_speed=0.2)
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
        """Tilting the source gives the liquid horizontal velocity, moving the landing forward."""
        gate_world = TestTransferGate()
        world, source, receiver = gate_world._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate_world._set_source_offset(world, source, 0.0)
        upright_landing = receiver.projectile_landing_point(
            source, world, exit_speed=0.2
        ).x.evaluate()[0]
        gate_world._set_source_offset(world, source, 0.6)
        tilted_landing = receiver.projectile_landing_point(
            source, world, exit_speed=0.2
        ).x.evaluate()[0]
        assert tilted_landing > upright_landing

    def test_higher_source_lands_farther(self):
        """A higher source gives the liquid a longer flight time, so it lands farther forward."""
        gate_world = TestTransferGate()
        low_world, low_source, low_receiver = gate_world._build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.3,
        )
        high_world, high_source, high_receiver = gate_world._build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.7,
        )
        low_world.set_positions_1DOF_connection(
            {low_source.root.parent_connection: 0.5}
        )
        high_world.set_positions_1DOF_connection(
            {high_source.root.parent_connection: 0.5}
        )
        low_landing = low_receiver.projectile_landing_point(
            low_source, low_world, exit_speed=0.2
        ).x.evaluate()[0]
        high_landing = high_receiver.projectile_landing_point(
            high_source, high_world, exit_speed=0.2
        ).x.evaluate()[0]
        assert high_landing > low_landing


class TestReceiveOutflowGuard:
    """Validates the guard against coupling from a source without outflow physics."""

    def test_raises_when_source_has_no_fill_equation(self):
        """Coupling from a source that was never initialized raises a meaningful error."""
        source = _TranslatingContainer(
            name=PrefixedName("dry_source"), root=Body(name=PrefixedName("dry_source"))
        )
        receiver = _TranslatingContainer(
            name=PrefixedName("receiver"), root=Body(name=PrefixedName("receiver"))
        )
        with pytest.raises(MissingFillEquationError):
            receiver.receive_outflow_from(source=source, world=World())

    def test_raises_when_source_already_coupled(self):
        """Coupling a source whose outflow is already gated onto another receiver raises."""
        world, source, receiver = TestTransferGate()._build_world()
        with world.modify_world():
            second_receiver = _TranslatingContainer.create_with_new_body_in_world(
                name=PrefixedName("second_receiver"),
                world=world,
                active_axis=Vector3(1, 0, 0),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-2.0, velocity=-1.0),
                    upper=DerivativeMap(position=2.0, velocity=1.0),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        second_receiver.initialize_fill_level(world=world, initial_fill=0.0)
        with pytest.raises(SourceAlreadyCoupledError):
            second_receiver.receive_outflow_from(source=source, world=world)


class TestCouplingReconstruction:
    """
    Validates that a transfer coupling is transmitted as a serializable parametric descriptor and
    rebuilt against the receiving world.

    The gate and inflow of a coupling are symbolic expressions bound to the world they were built
    in, so they cannot be serialized to another process. ``receive_outflow_from`` therefore records
    a :class:`~semantic_digital_twin.world_description.connections.LiquidTransferCoupling` descriptor
    on the fill connection, which survives synchronization and lets the receiving world rebuild the
    symbolic coupling locally.
    """

    def _coupled_world(self):
        return TestTransferGate()._build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )

    def test_receive_outflow_records_serializable_descriptor(self):
        """The coupling descriptor names the source and survives a JSON round trip."""
        world, source, receiver = self._coupled_world()
        coupling = receiver.inflow_coupling
        assert coupling is not None
        assert coupling.source_id == source.id

        restored = from_json(to_json(coupling))
        assert restored.source_id == source.id
        assert restored.exit_speed == coupling.exit_speed
        assert restored.height_gate_sharpness == coupling.height_gate_sharpness
        assert restored.overlap_gate_sharpness == coupling.overlap_gate_sharpness

    def test_rebuild_reconstructs_inflow_when_symbolic_state_absent(self):
        """
        Given the state a synchronized world holds - the descriptor present but the symbolic inflow
        side effect absent - the receiver rebuilds a working, world-bound inflow equation.
        """
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world)

        inflow_equation = receiver.fill_connection.inflow_equation
        assert inflow_equation is not None
        # The gate is a symbolic function of the source's DOF in this world; evaluating it proves
        # the rebuilt coupling is bound to this world's symbols, not the ones it was first built in.
        world.set_positions_1DOF_connection({source.root.parent_connection: 0.0})
        assert inflow_equation.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)
        # The inflow tracks the source's outflow: zero while upright, positive once the source tilts.
        assert inflow_equation.inflow.evaluate()[0] == pytest.approx(0.0)
        world.set_positions_1DOF_connection({source.root.parent_connection: 1.0})
        assert inflow_equation.inflow.evaluate()[0] > 0.0

    def test_rebuild_regates_source_outflow(self):
        """Rebuilding re-establishes the source's gated outflow, keeping the transfer spill-free."""
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world)

        source_outflow = source.fill_connection.outflow_equation
        assert isinstance(source_outflow, GatedArticulatedPouringEquation)
        world.set_positions_1DOF_connection({source.root.parent_connection: 0.0})
        assert source_outflow.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_rebuild_is_noop_when_inflow_already_present(self):
        """A receiver already carrying a symbolic inflow equation is left untouched."""
        world, source, receiver = self._coupled_world()
        original_inflow = receiver.fill_connection.inflow_equation
        assert original_inflow is not None

        receiver.ensure_inflow_coupling(world)

        assert receiver.fill_connection.inflow_equation is original_inflow


class TestNonCupLiquidSource:
    """A receiver fills from a :class:`LiquidSource` that is not a :class:`HasFillLevel` cup."""

    def test_receiver_fills_from_static_source(self):
        """
        Coupling a static faucet-like source (no fill level, no tilt) opens the gate and gives the
        receiver a positive inflow — a transfer the cup-only API could not express.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        with world.modify_world():
            receiver = _TranslatingContainer.create_with_new_body_in_world(
                name=PrefixedName("receiver"),
                world=world,
                active_axis=Vector3(1, 0, 0),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-2.0, velocity=-1.0),
                    upper=DerivativeMap(position=2.0, velocity=1.0),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        receiver.initialize_fill_level(world=world, initial_fill=0.0)

        source = _StaticLiquidSource(
            exit_point=Point3(x=0.0, y=0.0, z=0.5, reference_frame=world.root),
            volume_rate=0.001,
        )
        receiver.receive_outflow_from(source=source, world=world)

        inflow_equation = receiver.fill_connection.inflow_equation
        assert inflow_equation is not None
        assert inflow_equation.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)
        assert inflow_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0] > 0.0
