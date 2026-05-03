import numpy as np
import pytest
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Region, Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Color, Scale, Box
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    navigation_map_at_target,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@pytest.fixture
def simple_world():
    world = World()
    root = Body(name=PrefixedName("map"))
    with world.modify_world():
        world.add_body(root)

    # Add a target body
    target = Body(name=PrefixedName("target"))
    with world.modify_world():
        world.add_body(target)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=target,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0
                ),
            )
        )

    # Add an obstacle body
    obstacle = Body(
        name=PrefixedName("obstacle"),
        collision=ShapeCollection([Box(scale=Scale(0.5, 0.5, 0.5))]),
    )
    with world.modify_world():
        world.add_body(obstacle)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, y=0.5
                ),
            )
        )
    return world, target


def test_spawn_as_region(simple_world):
    world, target = simple_world

    # Create navigation map at target
    gcs = navigation_map_at_target(target=target)

    # Spawn GCS as region
    region = gcs.spawn_as_region()

    assert isinstance(region, Region)
    assert region in world.regions
    assert region.parent_connection.parent == gcs.search_space.reference_frame
    assert len(region.area.shapes) == len(gcs.graph.nodes())

    # Verify that shapes have correct origins
    # They should be relative to the region, and since the region is connected at identity
    # to the target, they should have the same coordinates as the boxes in GCS relative to the target.

    # Check first shape
    shape = region.area.shapes[0]
    box = list(gcs.graph.nodes())[0]

    expected_center_x = box.x_interval.center()
    expected_center_y = box.y_interval.center()
    expected_center_z = box.z_interval.center()

    # Shape origin relative to region
    assert np.allclose(shape.origin.x, expected_center_x)
    assert np.allclose(shape.origin.y, expected_center_y)
    assert np.allclose(shape.origin.z, expected_center_z)
