from copy import deepcopy

import numpy as np
import pytest

import experiments.orm.ormatic_interface  # type: ignore
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from experiments.sage_10k.sage10k_actions import Sage10kOpenDoor
from krrood.entity_query_language.backends import ProbabilisticBackend
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Wall,
    Door,
    Handle,
    Hinge,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color, Scale


@pytest.fixture
def wall_door_handle_world():
    world = World.create_with_root_body("map")

    with world.modify_world():
        wall = Wall.create_with_new_body_in_world(
            name="wall",
            world=world,
            scale=Scale(0.1, 4, 2),
        )
        wall.root.visual.dye_shapes(Color(R=0.6, G=0.6, B=0.6))

        door = Door.create_with_new_body_in_world(
            name="door",
            world=world,
            scale=Scale(0.11, 1, 2),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.0),
        )
        door.root.visual.dye_shapes(Color(R=0.55, G=0.27, B=0.07))

    with world.modify_world():
        wall.add(door.entry_way)

    with world.modify_world():
        handle = Handle.create_with_new_body_in_world(
            name="handle",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.6,
                y=0.25,
                x=0.06,
                yaw=np.pi,
            ),
            scale=Scale(0.05, 0.02, 0.2),
        )
        handle.root.visual.dye_shapes(Color(R=0.8, G=0.8, B=0.1))
        door.add(handle)

    world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
    with world.modify_world():
        hinge = Hinge.create_with_new_body_in_world(
            name="hinge",
            world=world,
            world_root_T_self=world_T_hinge,
            parent_connection_specification=Hinge.parent_connection_specification(
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=0.0),
                    upper=DerivativeMap(position=np.pi / 2, velocity=1.0),
                ),
            ),
        )
        door.add(hinge)

    return world, wall, door, handle


def test_door_opening(wall_door_handle_world, _hsr_world_setup, rclpy_node):
    world, wall, door, handle = wall_door_handle_world
    hsr_copy = deepcopy(_hsr_world_setup)
    world.merge_world(hsr_copy)
    odom_combined = world.get_body_by_name("odom_combined")
    odom_combined.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1, reference_frame=odom_combined.parent_kinematic_structure_entity
        )
    )

    context = Context.from_world(world, query_backend=ProbabilisticBackend())

    with simulated_robot:
        execute_single(Sage10kOpenDoor(door), context=context).perform()

    assert np.isclose(
        door.mechanical_joint.root.parent_connection.position, np.pi / 2, atol=2e-2
    )
