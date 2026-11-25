"""
Tests of the fill-level containers, the transfer between them and the liquid connection.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood.patterns.classproperty import classproperty
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
    PouringEquation,
)
from semantic_digital_twin.semantic_annotations.exceptions import (
    FillLevelAlreadyInitializedError,
    MissingFillEquationError,
    ReceiverAlreadyCoupledError,
    ReceiverNotInitializedError,
    SourceAlreadyCoupledError,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.world import World
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
from semantic_digital_twin.world_description.liquid_connection import LiquidConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% helpers


def _value(expression: sm.Scalar) -> float:
    """
    The numeric value of a symbolic expression at the world's current state.
    """
    return float(expression.evaluate()[0])


def _evaluated_xyz(spatial: Point3 | Vector3) -> np.ndarray:
    """
    The numeric ``x, y, z`` of a symbolic point or vector at the world's current state.
    """
    return np.array([_value(spatial.x), _value(spatial.y), _value(spatial.z)])


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


# %% test containers


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


@dataclass(eq=False, kw_only=True)
class _SoftGateContainer(_TranslatingContainer):
    """
    A receiver whose transfer gate opens and closes gently, so ordinary poses hold it
    partly open.
    """

    @classproperty
    def height_gate_sharpness(cls) -> float:
        """
        A gentle height gate.
        """
        return 1.0

    @classproperty
    def overlap_gate_sharpness(cls) -> float:
        """
        A gentle overlap gate.
        """
        return 1.0


@dataclass(eq=False, kw_only=True)
class _SteadilyFillingContainer(_TiltingContainer):
    """
    A container whose fill rises at a constant rate whatever its pose.
    """

    @classproperty
    def steady_fill_velocity(cls) -> float:
        """
        The constant rate of the normalized fill level, per second.
        """
        return 0.1

    def fill_velocity(self) -> sm.Scalar:
        """
        :return: The constant fill rate.
        """
        return sm.Scalar(self.steady_fill_velocity)


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


def _lone_tilting_cup(
    *, container_class: type[HasFillLevel] = _TiltingContainer
) -> tuple[World, HasFillLevel]:
    """
    A world holding a single cup without fill level that tilts about y.
    """
    world = _world_with_root()
    cup = _add_container(
        world=world,
        container_class=container_class,
        name="cup",
        axis=Vector3(0, 1, 0),
    )
    return world, cup


def _build_world(
    *,
    source_class: type[HasFillLevel] = _TranslatingContainer,
    source_axis: Vector3 | None = None,
    receiver_class: type[HasFillLevel] = _TranslatingContainer,
    source_height: float = 0.3,
    receiver_height: float = 0.2,
    receiver_depth: float = 0.1,
    scale_factor: float = 1.0,
    couple: bool = True,
    initialize_receiver_fill: bool = True,
) -> tuple[World, HasFillLevel, HasFillLevel]:
    """
    Builds a fixed receiver at the origin and a source held above it on a single degree
    of freedom.

    :param source_class: The container annotation type used for the source.
    :param source_axis: Axis of the source's degree of freedom; defaults to x.
    :param receiver_class: The container annotation type used for the receiver.
    :param source_height: Height of the source's origin above the world root, in metres.
    :param receiver_height: Height of the receiver's collision geometry, in metres.
    :param receiver_depth: Depth of the receiver's collision geometry, in metres, which
        changes its capacity but not the width its pouring model reads.
    :param scale_factor: Factor every length in the scene is multiplied by.
    :param couple: Whether the source pours into the receiver.
    :param initialize_receiver_fill: Whether the receiver's fill level is initialized.
    :return: The world, the source, and the receiver.
    """
    if source_axis is None:
        source_axis = Vector3(1, 0, 0)
    world = _world_with_root()
    receiver = _add_container(
        world=world,
        container_class=receiver_class,
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
        receiver.initialize_fill_level(initial_fill=0.0)
    source.initialize_fill_level(initial_fill=1.0)
    if couple:
        receiver.receive_outflow_from(source=source)
    return world, source, receiver


def _partly_open_transfer() -> tuple[World, HasFillLevel, HasFillLevel]:
    """
    A tilted source pouring into a receiver with soft gates, so both gate factors lie
    inside their transitions.
    """
    world, source, receiver = _build_world(
        source_class=_TiltingContainer,
        source_axis=Vector3(0, 1, 0),
        receiver_class=_SoftGateContainer,
    )
    _set_source_offset(world, source, 0.5)
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


def _replay_latest_modification(*, world: World, onto: World) -> None:
    """
    Apply the latest model modification of ``world`` to ``onto`` through JSON, as a
    world synchronizer does.
    """
    block = world.get_world_model_manager().model_modification_blocks[-1]
    tracker = WorldEntityWithIDKwargsTracker.from_world(onto)
    replayed = from_json(to_json(block), **tracker.create_kwargs())
    with onto.modify_world():
        replayed.apply(world=onto)


def _synchronized_copy(world: World) -> World:
    """
    A fresh world built by replaying every model modification of ``world`` through JSON,
    as a world synchronizer does, at the same state.
    """
    copy = World()
    for block in world.get_world_model_manager().model_modification_blocks:
        tracker = WorldEntityWithIDKwargsTracker.from_world(copy)
        replayed = from_json(to_json(block), **tracker.create_kwargs())
        with copy.modify_world():
            replayed.apply(world=copy)
    for degree_of_freedom in world.degrees_of_freedom:
        copy.state[degree_of_freedom.id].position = world.state[
            degree_of_freedom.id
        ].position
    copy.notify_state_change()
    return copy


def _counterpart(container: HasFillLevel, *, world: World) -> HasFillLevel:
    """
    The annotation of ``world`` that has the id of ``container``.
    """
    return world.get_semantic_annotation_by_id(container.id)


# %% fill velocity


class TestFillVelocity:
    """
    A container's fill rate is its drain, gated by the container it pours into, plus the
    gated volume its source loses over its own capacity.
    """

    def test_a_receiver_fills_at_the_gated_volume_its_source_loses(self) -> None:
        """
        The receiver gains the source's ungated volume rate, gated and converted by its
        own capacity.
        """
        _, source, receiver = _partly_open_transfer()

        assert _value(receiver.fill_velocity()) == pytest.approx(
            _value(receiver.transfer_gate(source=source))
            * _value(source.outflow_volume_rate())
            / receiver.capacity
        )

    def test_a_source_drains_at_its_gated_drain(self) -> None:
        """
        The source loses its drain, gated by where its stream lands.
        """
        _, source, receiver = _partly_open_transfer()

        assert _value(source.fill_velocity()) == pytest.approx(
            _value(receiver.transfer_gate(source=source))
            * _value(
                source.fill_equation.fill_velocity(
                    tilt=source.tilt, fill=source.fill_position
                )
            )
        )

    def test_a_container_pouring_into_nothing_drains_freely(self) -> None:
        """
        Without a receiver the fill falls at the drain the equation gives.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.8)
        JointState.from_mapping({cup.root.parent_connection: 1.0}).apply_to(world)

        assert _value(cup.fill_velocity()) == pytest.approx(
            _value(cup.fill_equation.fill_velocity(tilt=1.0, fill=0.8))
        )

    def test_a_partly_open_transfer_conserves_volume(self) -> None:
        """
        The volume the source loses is the volume the receiver gains.
        """
        _, source, receiver = _partly_open_transfer()
        gate = _value(receiver.transfer_gate(source=source))
        assert 0.01 < gate < 0.99, "the setup must hold the gate partly open"

        assert _value(receiver.fill_velocity()) * receiver.capacity == pytest.approx(
            -_value(source.fill_velocity()) * source.capacity
        )

    def test_containers_of_different_depth_conserve_volume(self) -> None:
        """
        A receiver that holds four times the liquid of the source rises four times more
        slowly than the source drops.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            receiver_depth=0.4,
        )
        _set_source_offset(world, source, 0.6)
        source_fill_loss_rate = -_value(source.fill_velocity())
        assert source_fill_loss_rate > 0.0, "the source must be draining"

        assert _value(
            receiver.fill_velocity()
        ) / source_fill_loss_rate == pytest.approx(
            _container_volume(source) / _container_volume(receiver)
        )

    def test_a_receiver_of_measured_capacity_converts_volume_by_it(self) -> None:
        """
        A receiver whose capacity was measured holds that much at full fill, whatever
        the box its collision geometry bounds would hold.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        receiver.measured_capacity = 3.0 * _container_volume(receiver)
        _set_source_offset(world, source, 0.6)

        assert _value(receiver.fill_velocity()) / -_value(
            source.fill_velocity()
        ) == pytest.approx(_container_volume(source) / receiver.measured_capacity)


# %% capacity


class TestCapacity:
    """
    A container holds the box its collision geometry bounds, unless it was measured.
    """

    def test_the_capacity_is_the_box_of_the_collision_geometry(self) -> None:
        """
        Without a measurement the capacity is the collision geometry's box.
        """
        _, _, receiver = _build_world(couple=False)
        assert receiver.capacity == pytest.approx(_container_volume(receiver))

    def test_a_measured_capacity_is_the_capacity(self) -> None:
        """
        A measured capacity takes the place of the one the collision geometry bounds.
        """
        _, _, receiver = _build_world(couple=False)
        receiver.measured_capacity = 0.002
        assert receiver.capacity == receiver.measured_capacity

    def test_a_measured_capacity_survives_a_json_round_trip(self) -> None:
        """
        The capacity reaches the process that controls the pour with the annotation.
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
    """
    The transfer is open while the source's lip is above the receiver's opening and its
    stream lands within the opening.
    """

    def test_the_gate_is_the_product_of_its_factors(self) -> None:
        """
        The gate is the height factor times the overlap factor.
        """
        _, source, receiver = _partly_open_transfer()

        assert _value(receiver.transfer_gate(source=source)) == pytest.approx(
            _value(receiver.height_gate(source=source))
            * _value(receiver.overlap_gate(source=source))
        )

    def test_the_height_factor_reads_the_clearance_in_receiver_heights(self) -> None:
        """
        The height factor is the logistic of the lip's clearance above the opening,
        measured in receiver heights.
        """
        _, source, receiver = _partly_open_transfer()
        clearance = _value(source.liquid_exit_point().z) - _value(
            receiver.opening_point().z
        )
        sharpness = receiver.height_gate_sharpness

        assert _value(receiver.height_gate(source=source)) == pytest.approx(
            1 / (1 + math.exp(-sharpness * clearance / receiver.root.collision.scale.z))
        )

    def test_the_overlap_factor_reads_the_landing_distance_in_opening_radii(
        self,
    ) -> None:
        """
        The overlap factor is the logistic of the landing point's margin inside the
        opening, measured in opening radii.
        """
        _, source, receiver = _partly_open_transfer()
        landing = _evaluated_xyz(
            receiver.projectile_landing_point(
                source=source, exit_speed=source.exit_speed()
            )
        )
        opening = _evaluated_xyz(receiver.opening_point())
        landing_distance = np.linalg.norm(landing[:2] - opening[:2])
        margin = (receiver.opening_radius - landing_distance) / receiver.opening_radius
        sharpness = receiver.overlap_gate_sharpness

        assert _value(receiver.overlap_gate(source=source)) == pytest.approx(
            1 / (1 + math.exp(-sharpness * margin))
        )

    def test_gate_is_open_when_source_is_above_receiver(self) -> None:
        """
        The gate is essentially fully open when the source is held directly over the
        receiver.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.0)
        assert _value(receiver.transfer_gate(source=source)) == pytest.approx(
            1.0, abs=1e-2
        )

    def test_gate_closes_monotonically_with_horizontal_offset(self) -> None:
        """
        Moving the source sideways past the receiver opening closes the gate
        monotonically.
        """
        world, source, receiver = _build_world()
        gate = receiver.transfer_gate(source=source)
        values = []
        for offset in [0.0, 0.05, 0.1, 0.3]:
            _set_source_offset(world, source, offset)
            values.append(_value(gate))
        assert all(earlier >= later for earlier, later in zip(values, values[1:]))
        assert values[0] > values[-1]
        assert values[-1] < 0.1

    def test_gate_has_nonzero_gradient_in_transition(self) -> None:
        """
        The gate is differentiable: its slope with respect to the source position is
        non-zero at the rim.
        """
        world, source, receiver = _build_world()
        gate = receiver.transfer_gate(source=source)
        source_position = source.root.parent_connection.dof.variables.position
        _set_source_offset(world, source, 0.05)
        gradient = _value(gate.jacobian(variables=[source_position])[0, 0])
        assert abs(gradient) > 1e-3

    def test_height_gate_is_unchanged_by_a_uniform_scene_scaling(self) -> None:
        """
        Whether the source's lip clears the receiver's opening depends on the shape of
        the arrangement alone, so two scenes that differ by a uniform scaling read the
        same height gate.
        """
        gates = []
        for scale_factor in (1.0, 4.0):
            world, source, receiver = _build_world(
                receiver_height=0.76, scale_factor=scale_factor
            )
            _set_source_offset(world, source, 0.0)
            gates.append(_value(receiver.height_gate(source=source)))
        assert 0.01 < gates[0] < 0.99, (
            "the setup must hold the lip in the gate's transition, not at either end"
        )

        assert gates[1] == pytest.approx(gates[0])

    def test_gate_closes_when_source_tilts(self) -> None:
        """
        Tilting a source held in place sends the stream past the receiver, closing the
        gate, which forces a controller to reposition the source while pouring.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate = receiver.transfer_gate(source=source)
        _set_source_offset(world, source, 0.0)
        upright_gate = _value(gate)
        _set_source_offset(world, source, 1.2)
        tilted_gate = _value(gate)
        assert upright_gate > 0.9
        assert tilted_gate < upright_gate
        assert tilted_gate < 0.5

    def test_gate_closes_when_the_source_lip_is_below_the_receiver_opening(
        self,
    ) -> None:
        """
        Liquid enters over the rim, so a receiver whose opening reaches above the
        source's lip cannot be poured into, even while the lip is still well above the
        receiver's base.
        """
        world, source, receiver = _build_world(receiver_height=1.0)
        _set_source_offset(world, source, 0.0)
        source_lip = _value(source.liquid_exit_point().z)
        receiver_base = _value(
            world.compose_forward_kinematics_expression(
                world.root, receiver.root
            ).position.z
        )
        receiver_opening = _value(receiver.opening_point().z)
        assert receiver_base < source_lip < receiver_opening, (
            "the setup must hold the lip above the receiver's base but below its opening"
        )

        assert _value(receiver.transfer_gate(source=source)) < 0.1


# %% tilt and exit speed


class TestTilt:
    """
    The tilt is the angle between a container's up axis and the vertical.
    """

    @pytest.mark.parametrize("angle", [0.7, -0.4, 0.0])
    def test_the_tilt_is_the_angle_from_the_vertical(self, *, angle: float) -> None:
        """
        Turning a container about a horizontal axis by an angle tilts it by that angle.
        """
        world, cup = _lone_tilting_cup()
        JointState.from_mapping({cup.root.parent_connection: angle}).apply_to(world)

        assert _value(cup.tilt) == pytest.approx(abs(angle), abs=1e-6)


class TestExitSpeed:
    """
    The stream leaves at the discharge-scaled Torricelli speed of the current head.
    """

    def _tilted_source(
        self,
        *,
        tilt: float,
        discharge_coefficient: float = PouringEquation.discharge_coefficient,
    ) -> HasFillLevel:
        """
        A full source tilted by ``tilt`` that drains with the given discharge
        coefficient.
        """
        world, source = _lone_tilting_cup()
        source.initialize_fill_level(
            initial_fill=1.0, discharge_coefficient=discharge_coefficient
        )
        JointState.from_mapping({source.root.parent_connection: tilt}).apply_to(world)
        return source

    def test_matches_discharge_scaled_torricelli(self) -> None:
        """
        The exit speed is ``C_d * sqrt(2 g h)`` for the current head above the lip.
        """
        source = self._tilted_source(tilt=1.2, discharge_coefficient=0.4)
        head = _value(
            source.fill_equation.head_above_lip(
                tilt=source.tilt, fill=source.fill_position
            )
        )

        assert _value(source.exit_speed()) == pytest.approx(
            0.4 * math.sqrt(2 * PouringEquation.standard_gravity * head)
        )

    def test_discharge_coefficient_scales_speed_linearly(self) -> None:
        """
        The discharge coefficient scales the exit speed proportionally.
        """
        low = self._tilted_source(tilt=1.2, discharge_coefficient=0.2)
        high = self._tilted_source(tilt=1.2, discharge_coefficient=0.6)

        assert _value(high.exit_speed()) / _value(low.exit_speed()) == pytest.approx(
            3.0
        )

    def test_more_tilt_pours_faster(self) -> None:
        """
        A steeper tilt lifts more liquid above the lip, so the exit speed grows.
        """
        gentle = self._tilted_source(tilt=0.6)
        steep = self._tilted_source(tilt=1.2)

        assert _value(steep.exit_speed()) > _value(gentle.exit_speed())

    def test_head_is_floored_when_barely_pouring(self) -> None:
        """
        With no head above the lip the head is floored so the exit-speed gradient stays
        finite.
        """
        source = self._tilted_source(tilt=0.0, discharge_coefficient=0.4)

        assert _value(source.exit_speed()) == pytest.approx(
            0.4
            * math.sqrt(
                2 * PouringEquation.standard_gravity * source.fill_equation.minimum_head
            )
        )

    def test_initialize_fill_level_threads_the_drain_parameters(self) -> None:
        """
        The parameters passed to ``initialize_fill_level`` reach the drain.
        """
        _, cup = _lone_tilting_cup()

        cup.initialize_fill_level(
            initial_fill=1.0,
            outflow_rate_constant=2.0,
            discharge_coefficient=0.5,
            repose_angle=0.4,
        )

        assert cup.fill_equation == PouringEquation(
            container_height=cup.root.collision.scale.z,
            container_width=cup.root.collision.scale.y,
            outflow_rate_constant=2.0,
            discharge_coefficient=0.5,
            repose_angle=0.4,
        )


# %% liquid exit point


class TestLiquidExitPoint:
    """
    Liquid leaves from the rim edge on the pour side, not the rim centre.
    """

    def _rim_center_world(self, *, world: World, source: HasFillLevel) -> Point3:
        """
        The centre of the source's rim in the world frame.
        """
        return (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )

    def test_exit_point_at_rim_centre_when_upright(self) -> None:
        """
        An upright cup has no pour direction, so the exit point stays at the rim centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.0)

        exit_point = _evaluated_xyz(source.liquid_exit_point())
        rim_center = _evaluated_xyz(self._rim_center_world(world=world, source=source))

        assert exit_point[:2] == pytest.approx(rim_center[:2], abs=1e-3)

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self) -> None:
        """
        A tilted cup pours over the rim edge on the pour side, a full rim radius from
        centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 1.2)

        offset = _evaluated_xyz(source.liquid_exit_point()) - _evaluated_xyz(
            self._rim_center_world(world=world, source=source)
        )
        collision = source.root.collision

        assert np.linalg.norm(offset) == pytest.approx(
            (collision.max_point.x - collision.min_point.x) / 2, abs=1e-3
        )
        pour_direction = _evaluated_xyz(source.opening_direction())
        assert offset[:2] @ pour_direction[:2] > 0


# %% projectile landing point


class TestProjectileLandingPoint:
    """
    The projectile model locates where poured liquid lands.
    """

    def test_rim_source_launches_horizontally(self) -> None:
        """
        Liquid spilling over a rim has no vertical launch: it travels towards where the
        opening faces, horizontally.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)

        exit_velocity = _evaluated_xyz(source.liquid_exit_velocity(exit_speed=0.2))

        assert exit_velocity[2] == 0.0
        assert _horizontal_unit(exit_velocity) == pytest.approx(
            _horizontal_unit(_evaluated_xyz(source.opening_direction()))
        )

    def test_a_rim_source_launches_at_the_horizontal_share_of_the_exit_speed(
        self,
    ) -> None:
        """
        The stream's horizontal speed is the exit speed scaled by how far the opening
        leans from vertical.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)
        exit_speed = 0.2

        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(exit_speed=exit_speed)
        )

        assert exit_velocity[:2] == pytest.approx(
            exit_speed * _evaluated_xyz(source.opening_direction())[:2]
        )

    def test_upright_source_lands_below_its_rim(self) -> None:
        """
        With no tilt the liquid has no horizontal velocity, so it lands directly below
        the rim.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)

        landing = _evaluated_xyz(
            receiver.projectile_landing_point(source=source, exit_speed=0.2)
        )
        source_rim = _evaluated_xyz(
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )

        assert landing[:2] == pytest.approx(source_rim[:2], abs=1e-3)

    def test_tilting_moves_landing_forward(self) -> None:
        """
        Tilting the source gives the liquid horizontal velocity, moving the landing
        forward.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        landing = receiver.projectile_landing_point(source=source, exit_speed=0.2)
        _set_source_offset(world, source, 0.0)
        upright_landing = _value(landing.x)
        _set_source_offset(world, source, 0.6)

        assert _value(landing.x) > upright_landing

    def test_higher_source_lands_farther(self) -> None:
        """
        A higher source gives the liquid a longer flight time, so it lands farther
        forward.
        """
        landings = []
        for source_height in (0.3, 0.7):
            world, source, receiver = _build_world(
                source_class=_TiltingContainer,
                source_axis=Vector3(0, 1, 0),
                source_height=source_height,
            )
            _set_source_offset(world, source, 0.5)
            landings.append(
                _value(
                    receiver.projectile_landing_point(source=source, exit_speed=0.2).x
                )
            )

        assert landings[1] > landings[0]

    def test_arc_terminates_at_the_opening_not_the_base(self) -> None:
        """
        The arc ends where the liquid arrives, the receiver's opening, not the
        receiver's origin at its base.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)

        landing = receiver.projectile_landing_point(source=source, exit_speed=0.2)

        assert _value(landing.z) == pytest.approx(
            _value(receiver.opening_point().z), abs=1e-9
        )

    def test_throw_distance_follows_the_fall_to_the_opening(self) -> None:
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
            source=source, exit_speed=exit_speed
        )

        exit_point = _evaluated_xyz(source.liquid_exit_point())
        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(exit_speed=exit_speed)
        )
        fall_height = exit_point[2] - _value(receiver.opening_point().z)
        flight_time = math.sqrt(2 * fall_height / PouringEquation.standard_gravity)
        assert _value(landing.x) == pytest.approx(
            exit_point[0] + exit_velocity[0] * flight_time, abs=1e-9
        )


# %% coupling guards


class TestReceiveOutflowGuard:
    """
    The guards of ``receive_outflow_from`` refuse transfers that cannot work.
    """

    def test_raises_when_source_has_no_fill_level(self) -> None:
        """
        A source that was never initialized cannot pour.
        """
        world, source, receiver = _build_world(couple=False)
        dry_source = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="dry_source",
            axis=Vector3(1, 0, 0),
        )

        with pytest.raises(MissingFillEquationError) as error_info:
            receiver.receive_outflow_from(source=dry_source)
        assert error_info.value.source is dry_source

    def test_raises_when_source_already_pours(self) -> None:
        """
        A source that already pours into one container cannot pour into a second.
        """
        world, source, receiver = _build_world()
        second_receiver = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="second_receiver",
            axis=Vector3(1, 0, 0),
        )
        second_receiver.initialize_fill_level(initial_fill=0.0)

        with pytest.raises(SourceAlreadyCoupledError) as error_info:
            second_receiver.receive_outflow_from(source=source)
        assert error_info.value.source is source
        assert source.pours_into is receiver

    def test_raises_when_receiver_has_no_fill_level(self) -> None:
        """
        Pouring into a receiver that was never initialized blames the receiver.
        """
        _, source, receiver = _build_world(couple=False, initialize_receiver_fill=False)

        with pytest.raises(ReceiverNotInitializedError) as error_info:
            receiver.receive_outflow_from(source=source)
        assert error_info.value.receiver is receiver

    def test_raises_when_receiver_already_fed(self) -> None:
        """
        A second source cannot pour into a receiver that already has one.
        """
        world, source, receiver = _build_world()
        second_source = _add_container(
            world=world,
            container_class=_TranslatingContainer,
            name="second_source",
            axis=Vector3(1, 0, 0),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3),
        )
        second_source.initialize_fill_level(initial_fill=1.0)

        with pytest.raises(ReceiverAlreadyCoupledError) as error_info:
            receiver.receive_outflow_from(source=second_source)
        assert error_info.value.receiver is receiver
        assert receiver.poured_from() is source
        assert second_source.pours_into is None


class TestFillLevelInitializationGuard:
    """
    A container's fill level can only be initialized once.
    """

    def test_second_initialization_raises(self) -> None:
        """
        A second ``initialize_fill_level`` call raises instead of stacking a second
        fill-level body and connection onto the container.
        """
        _, source, _ = _build_world(couple=False)
        first_fill_connection = source.fill_connection

        with pytest.raises(FillLevelAlreadyInitializedError) as error_info:
            source.initialize_fill_level(initial_fill=1.0)
        assert error_info.value.container is source
        assert source.fill_connection is first_fill_connection


# %% synchronized worlds


class TestSynchronizedTransfer:
    """
    A world rebuilt from another's model modifications, as a world synchronizer does,
    pours the same way without any extra step.
    """

    def test_the_copy_knows_which_container_pours_into_which(self) -> None:
        """
        The transfer arrives as a reference between the copy's own annotations.
        """
        world, source, receiver = _partly_open_transfer()

        copy = _synchronized_copy(world)

        copied_source = _counterpart(source, world=copy)
        assert copied_source.pours_into is _counterpart(receiver, world=copy)
        assert copied_source.fill_connection._world is copy

    def test_the_copy_pours_like_the_original(self) -> None:
        """
        Stepping both worlds alike moves the same volumes in both.
        """
        world, source, receiver = _partly_open_transfer()
        copy = _synchronized_copy(world)

        for _ in range(20):
            _step_passive_state(world, 0.05)
            _step_passive_state(copy, 0.05)

        assert receiver.fill_level > 0.0, "the setup must transfer liquid"
        assert _counterpart(source, world=copy).fill_level == pytest.approx(
            source.fill_level
        )
        assert _counterpart(receiver, world=copy).fill_level == pytest.approx(
            receiver.fill_level
        )

    def test_a_changed_drain_reaches_the_copy(self) -> None:
        """
        A drain replaced in one world drains the source of the other at the next step.
        """
        world, source, receiver = _partly_open_transfer()
        copy = _synchronized_copy(world)
        copied_source = _counterpart(source, world=copy)
        _step_passive_state(copy, 0.05)
        faster_drain = PouringEquation(
            container_height=source.fill_equation.container_height,
            container_width=source.fill_equation.container_width,
            outflow_rate_constant=2 * source.fill_equation.outflow_rate_constant,
        )

        with world.modify_world():
            source.set_fill_equation(fill_equation=faster_drain)
        _replay_latest_modification(world=world, onto=copy)

        assert copied_source.fill_equation == faster_drain
        compiled = copied_source.fill_connection.current_compiled_fill_velocity()
        assert compiled.evaluate() == pytest.approx(
            _value(copied_source.fill_velocity())
        )


# %% fill-level integration


class TestFillLevelIntegration:
    """
    Each physics step integrates the fill level from the velocity of the container it
    belongs to, inside its limits.
    """

    def test_tilted_empty_source_does_not_drain_below_empty(self) -> None:
        """
        A fully tilted cup with no liquid left stays at fill 0, not negative.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.0)
        JointState.from_mapping({cup.root.parent_connection: 1.5}).apply_to(world)

        for _ in range(50):
            _step_passive_state(world, 0.05)

        assert cup.fill_level == pytest.approx(0.0)

    def test_a_filling_container_saturates_at_full(self) -> None:
        """
        A container with a steady inflow stops at fill 1, not past the brim.
        """
        world, cup = _lone_tilting_cup(container_class=_SteadilyFillingContainer)
        cup.initialize_fill_level(initial_fill=0.9)

        for _ in range(50):
            _step_passive_state(world, 0.05)

        assert cup.fill_level == pytest.approx(1.0)

    def test_a_step_advances_the_fill_by_its_velocity(self) -> None:
        """
        One step moves the fill level by its velocity times the step length.
        """
        world, cup = _lone_tilting_cup(container_class=_SteadilyFillingContainer)
        cup.initialize_fill_level(initial_fill=0.2)
        dt = 0.05

        _step_passive_state(world, dt)

        assert cup.fill_level == pytest.approx(
            0.2 + _SteadilyFillingContainer.steady_fill_velocity * dt
        )

    def test_repeated_steps_compile_the_fill_velocity_once(
        self, *, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Stepping the fill level reuses the fill velocity compiled at the first step.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.8)
        _step_passive_state(world, 0.05)
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
            _step_passive_state(world, 0.05)

        assert compilations == []


# %% the liquid connection on its own


def _world_with_bare_liquid_connection(
    dof_limits: DegreeOfFreedomLimits | None,
) -> tuple[World, LiquidConnection]:
    """
    A world holding a body with a fill-level connection that no container owns.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    fill_level_body = Body(name=PrefixedName("container_fill_level"))
    with world.modify_world():
        world.add_body(container)
    with world.modify_world():
        connection = LiquidConnection.create_with_dofs(
            world=world,
            parent=container,
            child=fill_level_body,
            axis=Vector3.Z(),
            dof_limits=dof_limits,
        )
        world.add_connection(connection)
    return world, connection


def _fill_level_limits() -> DegreeOfFreedomLimits:
    """
    The [0, 1] fill-level limits a fill degree of freedom is created with.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=0.0, velocity=-1.0),
        upper=DerivativeMap(position=1.0, velocity=1.0),
    )


class TestLiquidConnection:
    """
    The fill level is a passive degree of freedom that only the physics step moves.
    """

    def test_the_fill_degree_of_freedom_is_passive(self) -> None:
        """
        The QP never commands the fill level.
        """
        world, connection = _world_with_bare_liquid_connection(_fill_level_limits())

        assert connection.active_dofs == []
        assert connection.passive_dofs == [connection.raw_dof]
        assert connection.raw_dof in world.passive_degrees_of_freedom
        assert connection.raw_dof not in world.active_degrees_of_freedom
        assert connection.is_controlled is False

    def test_a_connection_no_container_owns_stands_still(self) -> None:
        """
        Without a container there is nothing to fill or drain.
        """
        world, connection = _world_with_bare_liquid_connection(_fill_level_limits())

        _step_passive_state(world, 0.05)

        assert connection.position == 0.0

    def test_update_state_without_position_limits_raises(self) -> None:
        """
        A fill level without position limits cannot be clamped, so stepping it raises.
        """
        _, connection = _world_with_bare_liquid_connection(dof_limits=None)

        with pytest.raises(MissingFillLevelLimitsError) as error_info:
            connection.update_state(0.05)
        assert error_info.value.connection_name == connection.name
