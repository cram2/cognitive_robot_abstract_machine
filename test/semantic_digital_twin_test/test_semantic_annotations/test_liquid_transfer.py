"""
Tests of the fill-level containers, the transfer between them and the liquid connection.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
import pytest
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


# %% test containers


@dataclass(eq=False, kw_only=True)
class _Container(HasFillLevel):
    """
    A pourable container.
    """


@dataclass(eq=False, kw_only=True)
class _SoftGateContainer(_Container):
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
class _SteadilyFillingContainer(_Container):
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
    world_root_T_self: HomogeneousTransformationMatrix | None = None,
) -> HasFillLevel:
    """
    :param world: The world to add the container to.
    :param container_class: The container annotation type.
    :param name: The name of the container.
    :param axis: Axis the container tilts about.
    :param world_root_T_self: Pose of the container in the world; the origin if not
        given.
    :return: A 0.1 x 0.1 x 0.2 m container without fill level, tilting on one degree of
        freedom whose limits no test motion reaches.
    """
    with world.modify_world():
        return container_class.create_with_new_body_in_world(
            name=name,
            world=world,
            world_root_T_self=world_root_T_self,
            parent_connection_specification=RevoluteConnectionSpecification(
                axis=axis,
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-2.0, velocity=-1.0),
                    upper=DerivativeMap(position=2.0, velocity=1.0),
                ),
            ),
            scale=Scale(0.1, 0.1, 0.2),
        )


def _lone_tilting_cup(
    *, container_class: type[HasFillLevel] = _Container
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
    receiver_class: type[HasFillLevel] = _Container,
    couple: bool = True,
    initialize_receiver_fill: bool = True,
) -> tuple[World, HasFillLevel, HasFillLevel]:
    """
    Builds a receiver at the origin and a full source held 0.3 m above it that tilts
    about y.

    :param receiver_class: The container annotation type used for the receiver.
    :param couple: Whether the source pours into the receiver.
    :param initialize_receiver_fill: Whether the receiver's fill level is initialized.
    :return: The world, the source, and the receiver.
    """
    world = _world_with_root()
    receiver = _add_container(
        world=world,
        container_class=receiver_class,
        name="receiver",
        axis=Vector3(1, 0, 0),
    )
    source = _add_container(
        world=world,
        container_class=_Container,
        name="source",
        axis=Vector3(0, 1, 0),
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3),
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
    world, source, receiver = _build_world(receiver_class=_SoftGateContainer)
    _tilt_source(world, source, 0.5)
    return world, source, receiver


def _tilt_source(world: World, source: HasFillLevel, tilt: float) -> None:
    """
    Tilts the source to ``tilt`` about its tilt axis.
    """
    JointState.from_mapping({source.root.parent_connection: tilt}).apply_to(world)


def _step_passive_state(world: World, dt: float) -> None:
    """
    Advance the world by ``dt`` with every commanded velocity at zero, so only the
    connections that integrate their own state move.
    """
    world.apply_control_commands(
        commands=np.zeros(len(world.state)), dt=dt, derivative=Derivatives.velocity
    )


def _replay(blocks: list, *, onto: World) -> None:
    """
    Apply model modification blocks to ``onto`` through JSON, as a world synchronizer
    does.
    """
    for block in blocks:
        tracker = WorldEntityWithIDKwargsTracker.from_world(onto)
        replayed = from_json(to_json(block), **tracker.create_kwargs())
        with onto.modify_world():
            replayed.apply(world=onto)


def _synchronized_copy(world: World) -> World:
    """
    A fresh world built by replaying every model modification of ``world``, at the same
    state.
    """
    copy = World()
    _replay(world.get_world_model_manager().model_modification_blocks, onto=copy)
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

    def test_a_receiver_of_measured_capacity_converts_volume_by_it(self) -> None:
        """
        A receiver whose capacity was measured holds that much at full fill, whatever
        the box its collision geometry bounds would hold.
        """
        world, source, receiver = _build_world()
        receiver.measured_capacity = 3.0 * _container_volume(receiver)
        _tilt_source(world, source, 0.6)

        assert _value(receiver.fill_velocity()) / -_value(
            source.fill_velocity()
        ) == pytest.approx(_container_volume(source) / receiver.measured_capacity)


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


# %% tilt


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

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self) -> None:
        """
        A tilted cup pours over the rim edge on the pour side, a full rim radius from
        centre.
        """
        world, source, _ = _build_world()
        _tilt_source(world, source, 1.2)

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

    def test_a_rim_source_launches_at_the_horizontal_share_of_the_exit_speed(
        self,
    ) -> None:
        """
        The stream's horizontal speed is the exit speed scaled by how far the opening
        leans from vertical.
        """
        world, source, _ = _build_world()
        _tilt_source(world, source, 0.6)
        exit_speed = 0.2

        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(exit_speed=exit_speed)
        )

        assert exit_velocity[:2] == pytest.approx(
            exit_speed * _evaluated_xyz(source.opening_direction())[:2]
        )

    def test_throw_distance_follows_the_fall_to_the_opening(self) -> None:
        """
        The stream lands at the receiver's opening, thrown horizontally by the exit
        velocity over the time the liquid takes to fall from the source's lip to it.
        """
        exit_speed = 0.2
        world, source, receiver = _build_world()
        _tilt_source(world, source, 0.6)

        landing = receiver.projectile_landing_point(
            source=source, exit_speed=exit_speed
        )

        exit_point = _evaluated_xyz(source.liquid_exit_point())
        exit_velocity = _evaluated_xyz(
            source.liquid_exit_velocity(exit_speed=exit_speed)
        )
        opening_height = _value(receiver.opening_point().z)
        fall_height = exit_point[2] - opening_height
        flight_time = math.sqrt(2 * fall_height / PouringEquation.standard_gravity)
        assert _value(landing.x) == pytest.approx(
            exit_point[0] + exit_velocity[0] * flight_time, abs=1e-9
        )
        assert _value(landing.z) == pytest.approx(opening_height, abs=1e-9)


# %% coupling guards


class TestReceiveOutflowGuard:
    """
    The guards of ``receive_outflow_from`` refuse transfers that cannot work.
    """

    def test_raises_when_source_has_no_fill_level(self) -> None:
        """
        A source that was never initialized cannot pour.
        """
        world, _, receiver = _build_world(couple=False)
        dry_source = _add_container(
            world=world,
            container_class=_Container,
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
            container_class=_Container,
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
            container_class=_Container,
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


class TestFillLevelInitialization:
    """
    Initializing a container's fill level gives it its drain, once.
    """

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

    def test_the_copy_pours_like_the_original(self) -> None:
        """
        The copy knows which container pours into which, and stepping both worlds alike
        moves the same volumes in both.
        """
        world, source, receiver = _partly_open_transfer()
        copy = _synchronized_copy(world)

        copied_source = _counterpart(source, world=copy)
        assert copied_source.pours_into is _counterpart(receiver, world=copy)
        for _ in range(20):
            _step_passive_state(world, 0.05)
            _step_passive_state(copy, 0.05)

        assert receiver.fill_level > 0.0, "the setup must transfer liquid"
        assert copied_source.fill_level == pytest.approx(source.fill_level)
        assert _counterpart(receiver, world=copy).fill_level == pytest.approx(
            receiver.fill_level
        )

    def test_a_changed_drain_reaches_the_copy(self) -> None:
        """
        A drain replaced in one world drains the source of the other at the next step.
        """
        world, source, _ = _partly_open_transfer()
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
        _replay(
            world.get_world_model_manager().model_modification_blocks[-1:], onto=copy
        )

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


# %% the liquid connection on its own


def _world_with_unlimited_liquid_connection() -> tuple[World, LiquidConnection]:
    """
    A world holding a body with a fill-level connection without position limits.
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
        )
        world.add_connection(connection)
    return world, connection


class TestLiquidConnection:
    """
    The fill level is a passive degree of freedom that only the physics step moves.
    """

    def test_the_fill_degree_of_freedom_is_passive(self) -> None:
        """
        The QP never commands the fill level.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.5)
        connection = cup.fill_connection

        assert connection.active_dofs == []
        assert connection.passive_dofs == [connection.raw_dof]
        assert connection.raw_dof in world.passive_degrees_of_freedom
        assert connection.raw_dof not in world.active_degrees_of_freedom
        assert connection.is_controlled is False

    def test_update_state_without_position_limits_raises(self) -> None:
        """
        A fill level without position limits cannot be clamped, so stepping it raises.
        """
        _, connection = _world_with_unlimited_liquid_connection()

        with pytest.raises(MissingFillLevelLimitsError) as error_info:
            connection.update_state(0.05)
        assert error_info.value.connection_name == connection.name
