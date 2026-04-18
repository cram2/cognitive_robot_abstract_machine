"""
Tests for the coupled pouring chain (D_pour → receiver).

Verifies that CoupledPouringCauses correctly models:
  source fill decrease → receiver fill increase (via QP volume conservation constraint)

The source cup has a kinematic chain: world → PrismaticX → PrismaticZ → RevoluteY.
The QP drives the x/z DOFs to keep the spilling rim above the receiver at every tick.
The receiver is a fixed container placed below the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.body_motion_problem import Motion
from semantic_digital_twin.reasoning.body_motion_problem.pouring import (
    ArticulatedPouringEquation,
    ContainerGeometry,
    CoupledPouringCauses,
    CoupledPouringMSCModel,
    PouringMSCModel,
    ReceiverFillEffect,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap


@dataclass(eq=False)
class PourableContainer(HasRootBody, HasFillLevel):
    """Source cup: root body connected via RevoluteConnection (tilt joint)."""

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@dataclass(eq=False)
class ReceiverContainer(HasRootBody, HasFillLevel):
    """Receiver cup: fixed in world, initially empty."""

    @classproperty
    def _parent_connection_type(self):
        return FixedConnection


SOURCE_GEOMETRY = ContainerGeometry(height=0.12, half_width=0.04)
RECEIVER_GEOMETRY = ContainerGeometry(height=0.10, half_width=0.05)

RECEIVER_TOP_Z = 0.70
HEIGHT_ABOVE_RECEIVER = 0.10

_R = SOURCE_GEOMETRY.half_width
_A = SOURCE_GEOMETRY.height


@pytest.fixture
def world_with_source_and_receiver():
    """
    World with source cup (kinematic chain: PrismaticX → PrismaticZ → RevoluteY)
    positioned so its rim starts directly above the receiver opening.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))

    # Receiver: fixed below source
    with world.modify_world():
        receiver = ReceiverContainer.create_with_new_body_in_world(
            name=PrefixedName("receiver"),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2,
                y=0.0,
                z=RECEIVER_TOP_Z - RECEIVER_GEOMETRY.height / 2,
                reference_frame=world.root,
            ),
            scale=Scale(
                2 * RECEIVER_GEOMETRY.half_width,
                2 * RECEIVER_GEOMETRY.half_width,
                RECEIVER_GEOMETRY.height,
            ),
        )
    receiver.container_geometry = RECEIVER_GEOMETRY
    receiver.initialize_fill_level(
        world=world, parent_body=receiver.root, initial_fill=0.0
    )

    receiver_center = receiver.root.global_pose.to_position().to_np()[:3]
    receiver_opening = receiver_center.copy()
    receiver_opening[2] += RECEIVER_GEOMETRY.height / 2
    x_dof_initial = float(receiver_opening[0]) - _R
    z_dof_initial = float(receiver_opening[2]) + HEIGHT_ABOVE_RECEIVER - _A

    x_dof_initial = x_dof_initial * -2
    z_dof_initial = z_dof_initial * 1.2

    # Source cup kinematic chain: world.root → x_body → z_body → cup_body
    x_body = Body(name=PrefixedName("source_x_translation"))
    with world.modify_world():
        world.add_body(x_body)
        x_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=x_body,
            axis=Vector3(1, 0, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-1.0, velocity=-2.0),
                upper=DerivativeMap(position=1.0, velocity=2.0),
            ),
        )
        world.add_connection(x_connection)
    world.set_positions_1DOF_connection({x_connection: x_dof_initial})

    z_body = Body(name=PrefixedName("source_z_translation"))
    with world.modify_world():
        world.add_body(z_body)
        z_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=x_body,
            child=z_body,
            axis=Vector3(0, 0, 1),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=2.0, velocity=2.0),
            ),
        )
        world.add_connection(z_connection)
    world.set_positions_1DOF_connection({z_connection: z_dof_initial})

    cup_body = Body(name=PrefixedName("source_cup"))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=cup_body),
        scale=Scale(2 * _R, 2 * _R, _A),
    )
    cup_body.visual = ShapeCollection(shapes=[cup_shape])
    cup_body.collision = ShapeCollection(shapes=[cup_shape])
    with world.modify_world():
        world.add_body(cup_body)
        tilt_connection = RevoluteConnection.create_with_dofs(
            world=world,
            parent=z_body,
            child=cup_body,
            axis=Vector3(0, 1, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
        )
        world.add_connection(tilt_connection)

    source = PourableContainer(name=PrefixedName("source_cup"), root=cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source)
    source.container_geometry = SOURCE_GEOMETRY
    source.initialize_fill_level(world=world, parent_body=cup_body, initial_fill=0.8)
    source.fill_equation = ArticulatedPouringEquation(
        tilt_connection=tilt_connection,
        fill_connection=source.fill_connection,
        container_geometry=SOURCE_GEOMETRY,
    )

    return world, source, receiver, receiver_opening, x_connection, z_connection


def test_receiver_fill_increases(world_with_source_and_receiver, rclpy_node):
    """CoupledPouringCauses produces a tilt trajectory that fills the receiver to setpoint."""
    world, source, receiver, receiver_opening, x_connection, z_connection = (
        world_with_source_and_receiver
    )
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()

    goal_receiver_fill = 0.2

    receiver_effect = ReceiverFillEffect(
        target_object=receiver,
        property_getter=lambda c: c.fill_level,
        goal_value=goal_receiver_fill,
        tolerance=0.05,
    )

    source_physics = PouringMSCModel(
        fill_equation=source.fill_equation, theta_max=math.pi / 2
    )
    coupled_model = CoupledPouringMSCModel(
        source_model=source_physics,
        source_geometry=SOURCE_GEOMETRY,
        receiver_geometry=RECEIVER_GEOMETRY,
        x_translation_connection=x_connection,
        z_translation_connection=z_connection,
        receiver_opening_world=receiver_opening,
        height_above_receiver=HEIGHT_ABOVE_RECEIVER,
    )

    motion = Motion(
        trajectory=[],
        actuator=source.fill_equation.tilt_connection,
        motion_model=coupled_model,
    )

    causes = CoupledPouringCauses(
        effect=receiver_effect,
        environment=world,
        motion=motion,
        source=source,
    )

    result = causes()
    assert result, "CoupledPouringCauses must confirm the receiver fill is achieved"
    assert len(motion.trajectory) > 0, "Tilt trajectory must have been generated"

    receiver_fill_traj = coupled_model.last_receiver_fill_trajectory
    assert len(receiver_fill_traj) > 0
    assert receiver_fill_traj[-1] >= goal_receiver_fill - receiver_effect.tolerance

    causes.replay(step_delay=0.2)


def test_world_state_reset_after_coupled_model(
    world_with_source_and_receiver, rclpy_node
):
    """Source and receiver fill levels are restored after CoupledPouringMSCModel.run()."""
    world, source, receiver, receiver_opening, x_connection, z_connection = (
        world_with_source_and_receiver
    )
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()

    source_fill_before = source.fill_level
    receiver_fill_before = receiver.fill_level

    receiver_effect = ReceiverFillEffect(
        target_object=receiver,
        property_getter=lambda c: c.fill_level,
        goal_value=0.3,
        tolerance=0.05,
    )

    coupled_model = CoupledPouringMSCModel(
        source_model=PouringMSCModel(fill_equation=source.fill_equation),
        source_geometry=SOURCE_GEOMETRY,
        receiver_geometry=RECEIVER_GEOMETRY,
        x_translation_connection=x_connection,
        z_translation_connection=z_connection,
        receiver_opening_world=receiver_opening,
        height_above_receiver=HEIGHT_ABOVE_RECEIVER,
    )
    coupled_model.run(receiver_effect, world)

    assert source.fill_level == pytest.approx(
        source_fill_before
    ), "Source fill must be restored after simulation"
    assert receiver.fill_level == pytest.approx(
        receiver_fill_before
    ), "Receiver fill must be restored after simulation"
