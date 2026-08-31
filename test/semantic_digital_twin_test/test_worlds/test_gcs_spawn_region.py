import numpy as np
import pytest
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Region, Body
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.geometry import Color, Scale, Box
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@pytest.fixture
def simple_world():
    world = World.create_with_root_body("map")

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
    gcs = PlanarGraphOfBoundingBoxes.navigation_map_at_target(target=target)

    # Spawn the GCS's free space as a region, extruded into a slab since the GCS's own
    # boxes are 2D. The floor's supporting surface stands in for the target's own
    # storage-space annotation here.
    slab_height = 0.1
    half_height = slab_height / 2
    floor = Floor(root=target, _world=world)
    boxes = BoundingBoxCollection(
        [box.extrude(half_height) for box in gcs.graph.nodes()],
        gcs.search_space.reference_frame,
    )
    region = floor.spawn_bounding_boxes_as_region(boxes=boxes)

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
    expected_center_z = box.origin.z

    # Shape origin relative to region
    assert np.allclose(shape.origin.x, expected_center_x)
    assert np.allclose(shape.origin.y, expected_center_y)
    assert np.allclose(shape.origin.z, expected_center_z)
    assert np.isclose(shape.scale.z, slab_height)
