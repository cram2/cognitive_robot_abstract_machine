import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% PlanarBoundingBox


def test_planar_bounding_box_transform_same_frame():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    bb = PlanarBoundingBox(
        -1,
        -1,
        1,
        1,
        HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=world.root),
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0, 1, 0, reference_frame=world.root
    )

    new_origin_bb = bb.transform_to_origin(new_origin)

    assert new_origin_bb.min_x == -1
    assert new_origin_bb.max_x == 1
    assert new_origin_bb.min_y == -2
    assert new_origin_bb.max_y == 0


def test_planar_bounding_box_transform_rotated():
    world = World()
    with world.modify_world():
        body1 = Body(name=PrefixedName("body1"))
        body2 = Body(name=PrefixedName("body2"))

        connection = FixedConnection(
            body1,
            body2,
            HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 0, yaw=np.pi / 2),
        )

        world.add_connection(connection)

    bb = PlanarBoundingBox(-0.5, -1, 0.5, 1, body2.global_pose)

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body1)

    new_bb = bb.transform_to_origin(new_origin)

    assert new_bb.min_x == 0.0
    assert new_bb.max_x == 2.0
    assert new_bb.min_y == pytest.approx(-0.5, abs=0.001)
    assert new_bb.max_y == pytest.approx(0.5, abs=0.001)
    assert sum(bb.dimensions) == sum(new_bb.dimensions)


def test_planar_bounding_box_event_round_trip():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    simple_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: closed(0, 2),
            SpatialVariables.y.value: closed(0, 2),
        }
    )
    event = Event.from_simple_sets(simple_event)

    bbc = BoundingBoxCollection.from_event(PlanarBoundingBox, world.root, event)
    bb = bbc.bounding_boxes[0]
    assert len(bbc.bounding_boxes) == 1
    assert bb.x_interval.lower == 0
    assert bb.x_interval.upper == 2
    assert bb.y_interval.lower == 0
    assert bb.y_interval.upper == 2


def test_planar_bounding_box_area():
    bb = PlanarBoundingBox(-0.5, -1, 0.5, 1, HomogeneousTransformationMatrix())

    assert bb.area == 2.0


def test_planar_bounding_box_area_of_a_flat_bounding_box_vanishes():
    bb = PlanarBoundingBox(-0.5, 1, 0.5, 1, HomogeneousTransformationMatrix())

    assert bb.area == 0.0


def test_planar_bounding_box_contains():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    bb = PlanarBoundingBox(
        -0.5, -1, 0.5, 1, HomogeneousTransformationMatrix(reference_frame=world.root)
    )

    point = Point2(0, 0, reference_frame=world.root)

    assert bb.contains(point)


def test_planar_bounding_box_does_not_contain_a_point_outside():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    bb = PlanarBoundingBox(
        -0.5, -1, 0.5, 1, HomogeneousTransformationMatrix(reference_frame=world.root)
    )

    point = Point2(10, 10, reference_frame=world.root)

    assert not bb.contains(point)


def test_planar_bounding_box_center():
    bb = PlanarBoundingBox(0, 0, 2, 4, HomogeneousTransformationMatrix())

    center = bb.center

    assert isinstance(center, Point2)
    assert float(center.x) == 1.0
    assert float(center.y) == 2.0


def test_planar_bounding_box_bloat():
    bb = PlanarBoundingBox(0, 0, 1, 1, HomogeneousTransformationMatrix())

    bloated = bb.bloat(0.5, 0.25)

    assert bloated.min_x == -0.5
    assert bloated.max_x == 1.5
    assert bloated.min_y == -0.25
    assert bloated.max_y == 1.25


def test_planar_bounding_box_intersection_with():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    a = PlanarBoundingBox(0, 0, 2, 2, origin)
    b = PlanarBoundingBox(1, 1, 3, 3, origin)

    intersection = a.intersection_with(b)

    assert intersection.min_x == 1
    assert intersection.min_y == 1
    assert intersection.max_x == 2
    assert intersection.max_y == 2


def test_planar_bounding_box_intersection_with_disjoint_box_is_none():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    a = PlanarBoundingBox(0, 0, 1, 1, origin)
    b = PlanarBoundingBox(5, 5, 6, 6, origin)

    assert a.intersection_with(b) is None


# %% BoundingBoxCollection[PlanarBoundingBox]


def test_planar_bounding_box_collection_merge():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    first = BoundingBoxCollection([PlanarBoundingBox(0, 0, 1, 1, origin)], world.root)
    second = BoundingBoxCollection([PlanarBoundingBox(1, 0, 2, 1, origin)], world.root)

    merged = first.merge(second)

    assert len(merged.bounding_boxes) == 2


def test_planar_bounding_box_collection_bounding_box():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    collection = BoundingBoxCollection(
        [
            PlanarBoundingBox(0, 0, 1, 1, origin),
            PlanarBoundingBox(2, -1, 3, 0, origin),
        ],
        world.root,
    )

    enclosing = collection.bounding_box()

    assert enclosing.min_x == 0
    assert enclosing.min_y == -1
    assert enclosing.max_x == 3
    assert enclosing.max_y == 1
