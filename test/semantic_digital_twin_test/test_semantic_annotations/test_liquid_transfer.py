"""
Tests of the fill-level containers, the transfer between them and the fill-level
connection.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from krrood.adapters.json_serializer import from_json, to_json
from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
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
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.specifications.connections import (
    ScrewAxisConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.derivatives import (
    DerivativeMap,
    Derivatives,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

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
            parent_connection_specification=ScrewAxisConnectionSpecification(
                screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=axis),
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
) -> tuple[World, HasFillLevel, HasFillLevel]:
    """
    Builds an empty receiver at the origin and a full source 0.3 m above it, tilting
    about y and pouring into the receiver.

    :param receiver_class: The container annotation type used for the receiver.
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
    receiver.initialize_fill_level(initial_fill=0.0)
    source.initialize_fill_level(initial_fill=1.0)
    receiver.receive_outflow_from(source=source)
    return world, source, receiver


def _partly_open_transfer() -> tuple[World, HasFillLevel, HasFillLevel]:
    """
    A tilted source pouring into a receiver with soft gates, so both gate factors lie
    inside their transitions.
    """
    world, source, receiver = _build_world(receiver_class=_SoftGateContainer)
    _tilt(world, source, 0.5)
    return world, source, receiver


def _tilt(world: World, container: HasFillLevel, tilt: float) -> None:
    """
    Tilts the container to ``tilt`` about its tilt axis.
    """
    JointState.from_mapping({container.root.parent_connection: tilt}).apply_to(world)


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
        gate = receiver.incoming_stream().gate().evaluate()[0]
        assert 0.01 < gate < 0.99, "the setup must hold the gate partly open"

        assert receiver.fill_velocity().evaluate()[
            0
        ] * receiver.capacity == pytest.approx(
            -source.fill_velocity().evaluate()[0] * source.capacity
        )


# %% geometric transfer gate


class TestTransferGate:
    """
    The transfer is open while the source's lip is above the receiver's opening and its
    stream lands within the opening.
    """

    def test_the_height_factor_reads_the_clearance_in_receiver_heights(self) -> None:
        """
        The height factor is the logistic of the lip's clearance above the opening,
        measured in receiver heights.
        """
        _, source, receiver = _partly_open_transfer()
        clearance = (
            source.liquid_exit_point().z.evaluate()[0]
            - receiver.opening_point().z.evaluate()[0]
        )
        sharpness = receiver.height_gate_sharpness

        assert receiver.incoming_stream().height_gate().evaluate()[0] == pytest.approx(
            1 / (1 + math.exp(-sharpness * clearance / receiver.root.collision.scale.z))
        )

    def test_the_overlap_factor_reads_the_landing_distance_in_opening_radii(
        self,
    ) -> None:
        """
        The overlap factor is the logistic of the landing point's margin inside the
        opening, measured in opening radii.
        """
        _, _, receiver = _partly_open_transfer()
        landing = receiver.incoming_stream().landing_point().evaluate()[:3]
        opening = receiver.opening_point().evaluate()[:3]
        landing_distance = np.linalg.norm(landing[:2] - opening[:2])
        margin = (receiver.opening_radius - landing_distance) / receiver.opening_radius
        sharpness = receiver.overlap_gate_sharpness

        assert receiver.incoming_stream().overlap_gate().evaluate()[0] == pytest.approx(
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
        _tilt(world, cup, angle)

        assert cup.tilt.evaluate()[0] == pytest.approx(abs(angle), abs=1e-6)


# %% liquid exit point


class TestLiquidExitPoint:
    """
    Liquid leaves from the rim edge on the pour side, not the rim centre.
    """

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self) -> None:
        """
        A tilted cup pours over the rim edge on the pour side, a full rim radius from
        centre.
        """
        world, source, _ = _build_world()
        _tilt(world, source, 1.2)

        offset = (
            source.liquid_exit_point().evaluate()[:3]
            - source.opening_point().evaluate()[:3]
        )

        assert np.linalg.norm(offset) == pytest.approx(
            source.root.collision.scale.x / 2, abs=1e-3
        )
        pour_direction = source.opening_direction().evaluate()[:3]
        assert offset[:2] @ pour_direction[:2] > 0


# %% liquid exit velocity


class TestLiquidExitVelocity:
    """
    Liquid leaves over the rim with the horizontal share of the exit speed.
    """

    def test_a_rim_source_launches_at_the_horizontal_share_of_the_exit_speed(
        self,
    ) -> None:
        """
        The stream's horizontal speed is the exit speed scaled by how far the opening
        leans from vertical.
        """
        world, source, _ = _build_world()
        _tilt(world, source, 0.6)
        exit_speed = 0.2

        exit_velocity = source.liquid_exit_velocity(exit_speed=exit_speed).evaluate()[
            :3
        ]

        assert exit_velocity[:2] == pytest.approx(
            exit_speed * source.opening_direction().evaluate()[:2]
        )


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
            container_scale=source.fill_equation.container_scale,
            outflow_rate_constant=2 * source.fill_equation.outflow_rate_constant,
        )

        with world.modify_world():
            source.set_fill_equation(fill_equation=faster_drain)
        _replay(
            world.get_world_model_manager().model_modification_blocks[-1:], onto=copy
        )

        assert copied_source.fill_equation == faster_drain
        assert copied_source.fill_connection.current_fill_velocity() == pytest.approx(
            copied_source.fill_velocity().evaluate()[0]
        )


# %% fill-level integration


class TestFillLevelIntegration:
    """
    Each physics step integrates the fill level from the velocity of the container it
    belongs to, inside its limits.
    """

    def test_a_step_advances_the_fill_by_its_velocity(self) -> None:
        """
        One step moves the fill level by its velocity times the step length.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.8)
        _tilt(world, cup, 1.0)
        fill_velocity = cup.fill_velocity().evaluate()[0]
        dt = 0.05

        _step_passive_state(world, dt)

        assert cup.fill_level == pytest.approx(0.8 + fill_velocity * dt)

    def test_a_draining_container_stops_at_empty(self) -> None:
        """
        A tilted container drains to fill 0, not below.
        """
        world, cup = _lone_tilting_cup()
        cup.initialize_fill_level(initial_fill=0.1)
        _tilt(world, cup, 1.5)

        for _ in range(50):
            _step_passive_state(world, 0.05)

        assert cup.fill_level == 0.0
