import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
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
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    bb = PlanarBoundingBox(
        min_x=-1,
        min_y=-1,
        max_x=1,
        max_y=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=world.root),
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0, 1, 0, reference_frame=world.root
    )

    new_origin_bb = bb.transform_to_origin(reference_T_new_origin=new_origin)

    assert new_origin_bb.min_x == -1
    assert new_origin_bb.max_x == 1
    assert new_origin_bb.min_y == -2
    assert new_origin_bb.max_y == 0


def test_planar_bounding_box_transform_rotated():
    world = World()
    with world.modify_world():
        body1 = Body(name=PrefixedName(name="body1"))
        body2 = Body(name=PrefixedName(name="body2"))

        connection = FixedConnection(
            parent=body1,
            child=body2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1, 0, 0, yaw=np.pi / 2
            ),
        )

        world.add_connection(connection=connection)

    bb = PlanarBoundingBox(
        min_x=-0.5, min_y=-1, max_x=0.5, max_y=1, origin=body2.global_pose
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body1)

    new_bb = bb.transform_to_origin(reference_T_new_origin=new_origin)

    assert new_bb.min_x == 0.0
    assert new_bb.max_x == 2.0
    assert new_bb.min_y == pytest.approx(-0.5, abs=0.001)
    assert new_bb.max_y == pytest.approx(0.5, abs=0.001)
    assert sum(bb.dimensions) == sum(new_bb.dimensions)


def test_planar_bounding_box_event_round_trip():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    simple_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: closed(0, 2),
            SpatialVariables.y.value: closed(0, 2),
        }
    )
    event = Event.from_simple_sets(simple_event)

    bbc = BoundingBoxCollection.from_event(
        box_type=PlanarBoundingBox, reference_frame=world.root, event=event
    )
    bb = bbc.bounding_boxes[0]
    assert len(bbc.bounding_boxes) == 1
    assert bb.x_interval.lower == 0
    assert bb.x_interval.upper == 2
    assert bb.y_interval.lower == 0
    assert bb.y_interval.upper == 2


def test_planar_bounding_box_area():
    bb = PlanarBoundingBox(
        min_x=-0.5,
        min_y=-1,
        max_x=0.5,
        max_y=1,
        origin=HomogeneousTransformationMatrix(),
    )

    assert bb.area == 2.0


def test_planar_bounding_box_area_of_a_flat_bounding_box_vanishes():
    bb = PlanarBoundingBox(
        min_x=-0.5,
        min_y=1,
        max_x=0.5,
        max_y=1,
        origin=HomogeneousTransformationMatrix(),
    )

    assert bb.area == 0.0


def test_planar_bounding_box_contains():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    bb = PlanarBoundingBox(
        min_x=-0.5,
        min_y=-1,
        max_x=0.5,
        max_y=1,
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
    )

    point = Point2(0, 0, reference_frame=world.root)

    assert bb.contains(point=point)


def test_planar_bounding_box_does_not_contain_a_point_outside():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    bb = PlanarBoundingBox(
        min_x=-0.5,
        min_y=-1,
        max_x=0.5,
        max_y=1,
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
    )

    point = Point2(10, 10, reference_frame=world.root)

    assert not bb.contains(point=point)


def test_planar_bounding_box_center():
    bb = PlanarBoundingBox(
        min_x=0, min_y=0, max_x=2, max_y=4, origin=HomogeneousTransformationMatrix()
    )

    center = bb.center

    assert isinstance(center, Point2)
    assert float(center.x) == 1.0
    assert float(center.y) == 2.0


def test_planar_bounding_box_bloat():
    bb = PlanarBoundingBox(
        min_x=0, min_y=0, max_x=1, max_y=1, origin=HomogeneousTransformationMatrix()
    )

    bloated = bb.bloat(x_amount=0.5, y_amount=0.25)

    assert bloated.min_x == -0.5
    assert bloated.max_x == 1.5
    assert bloated.min_y == -0.25
    assert bloated.max_y == 1.25


def test_planar_bounding_box_intersection_with():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    a = PlanarBoundingBox(min_x=0, min_y=0, max_x=2, max_y=2, origin=origin)
    b = PlanarBoundingBox(min_x=1, min_y=1, max_x=3, max_y=3, origin=origin)

    intersection = a.intersection_with(other=b)

    assert intersection.min_x == 1
    assert intersection.min_y == 1
    assert intersection.max_x == 2
    assert intersection.max_y == 2


def test_planar_bounding_box_intersection_with_disjoint_box_is_none():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    a = PlanarBoundingBox(min_x=0, min_y=0, max_x=1, max_y=1, origin=origin)
    b = PlanarBoundingBox(min_x=5, min_y=5, max_x=6, max_y=6, origin=origin)

    assert a.intersection_with(other=b) is None


# %% BoundingBoxCollection[PlanarBoundingBox, Point2]


def test_planar_bounding_box_collection_merge():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    first = BoundingBoxCollection(
        shapes=[PlanarBoundingBox(min_x=0, min_y=0, max_x=1, max_y=1, origin=origin)],
        reference_frame=world.root,
    )
    second = BoundingBoxCollection(
        shapes=[PlanarBoundingBox(min_x=1, min_y=0, max_x=2, max_y=1, origin=origin)],
        reference_frame=world.root,
    )

    merged = first.merge(other=second)

    assert len(merged.bounding_boxes) == 2


def test_planar_bounding_box_collection_bounding_box():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            kinematic_structure_entity=Body(name=PrefixedName(name="map"))
        )

    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    collection = BoundingBoxCollection(
        shapes=[
            PlanarBoundingBox(min_x=0, min_y=0, max_x=1, max_y=1, origin=origin),
            PlanarBoundingBox(min_x=2, min_y=-1, max_x=3, max_y=0, origin=origin),
        ],
        reference_frame=world.root,
    )

    enclosing = collection.bounding_box()

    assert enclosing.min_x == 0
    assert enclosing.min_y == -1
    assert enclosing.max_x == 3
    assert enclosing.max_y == 1
