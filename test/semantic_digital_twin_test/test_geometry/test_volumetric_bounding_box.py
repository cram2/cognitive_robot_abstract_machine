from collections import Counter

import numpy as np
import pytest
from random_events.interval import closed

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.numeric import NumericTransform
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body
from random_events.product_algebra import Event, SimpleEvent

from ...casadi_calls import CasadiCalls


def test_volumetric_bounding_box_transform_same_frame(pr2_apartment_state_reset):
    bb = VolumetricBoundingBox(
        -1,
        -1,
        -1,
        1,
        1,
        1,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=pr2_apartment_state_reset.root
        ),
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0, 0, 1, reference_frame=pr2_apartment_state_reset.root
    )

    assert bb.min_x == -1
    assert bb.max_x == 1
    assert bb.min_y == -1
    assert bb.max_y == 1
    assert bb.min_z == -1
    assert bb.max_z == 1
    assert bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]

    new_origin_bb = bb.transform_to_origin(new_origin)

    assert new_origin_bb.min_x == -1
    assert new_origin_bb.max_x == 1
    assert new_origin_bb.min_y == -1
    assert new_origin_bb.max_y == 1
    assert new_origin_bb.min_z == -2
    assert new_origin_bb.max_z == 0
    assert new_origin_bb.origin.to_position().to_np().tolist() == [0, 0, 1, 1]


def test_volumetric_bounding_box_transform_different_frame(pr2_apartment_state_reset):
    bb = VolumetricBoundingBox(
        0, 0, 0, 1, 1, 1, pr2_apartment_state_reset.root.global_pose
    )

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0,
        0,
        0,
        reference_frame=pr2_apartment_state_reset.get_body_by_name("base_footprint"),
    )

    assert bb.min_x == 0
    assert bb.max_x == 1
    assert bb.min_y == 0
    assert bb.max_y == 1
    assert bb.min_z == 0
    assert bb.max_z == 1
    assert bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]

    new_origin_bb = bb.transform_to_origin(new_origin)

    assert new_origin_bb.min_x == -1.3
    assert new_origin_bb.max_x == pytest.approx(-0.3, abs=0.001)
    assert new_origin_bb.min_y == -2
    assert new_origin_bb.max_y == -1
    assert new_origin_bb.min_z == 0
    assert new_origin_bb.max_z == 1
    assert new_origin_bb.origin.to_position().to_np().tolist() == [0, 0, 0, 1]


def test_volumetric_bounding_box_transform_rotated():
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

    bb = VolumetricBoundingBox(-0.5, -1, 0, 0.5, 1, 1, body2.global_pose)

    new_origin = HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body1)

    new_bb = bb.transform_to_origin(new_origin)

    assert new_bb.min_x == 0.0
    assert new_bb.max_x == 2.0
    assert new_bb.min_y == pytest.approx(-0.5, abs=0.001)
    assert new_bb.max_y == pytest.approx(0.5, abs=0.001)
    assert new_bb.min_z == 0
    assert new_bb.max_z == 1

    assert sum(bb.dimensions) == sum(new_bb.dimensions)


def test_event_casting(pr2_apartment_state_reset):
    simple_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: closed(0, 2),
            SpatialVariables.y.value: closed(0, 2),
            SpatialVariables.z.value: closed(0, 2),
        }
    )
    event = Event.from_simple_sets(simple_event)

    bbc = BoundingBoxCollection.from_event(
        VolumetricBoundingBox, pr2_apartment_state_reset.root, event
    )
    bb = bbc.bounding_boxes[0]
    assert len(bbc.bounding_boxes) == 1
    assert bb.x_interval.lower == 0
    assert bb.x_interval.upper == 2

    assert bb.y_interval.lower == 0
    assert bb.y_interval.upper == 2
    assert bb.z_interval.lower == 0
    assert bb.z_interval.upper == 2

    assert bb.min_x == -1
    assert bb.max_x == 1


def test_volume():
    bb = VolumetricBoundingBox(
        -0.5, -1, 0, 0.5, 1, 3, HomogeneousTransformationMatrix()
    )

    assert bb.volume == 6.0


def test_volume_of_a_flat_bounding_box_vanishes():
    bb = VolumetricBoundingBox(
        -0.5, -1, 1, 0.5, 1, 1, HomogeneousTransformationMatrix()
    )

    assert bb.volume == 0.0


def test_contains(pr2_apartment_state_reset):
    bb = VolumetricBoundingBox(
        -0.5, -1, 0, 0.5, 1, 1, pr2_apartment_state_reset.root.global_pose
    )

    point = Point3(0, 0, 0, reference_frame=pr2_apartment_state_reset.root)

    assert bb.contains(point)


# %% a box read as numbers


def _body_turned_a_quarter_next_to_another() -> tuple[Body, Body]:
    """
    Two bodies, the second one metre along the first's x-axis and turned a quarter
    around its z-axis.
    """
    world = World()
    with world.modify_world():
        body1 = Body(name=PrefixedName("body1"))
        body2 = Body(name=PrefixedName("body2"))
        world.add_connection(
            FixedConnection(
                body1,
                body2,
                HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 0, yaw=np.pi / 2),
            )
        )
    return body1, body2


def test_a_bounding_box_holds_its_origin_as_numbers():
    """
    A box's origin is read on every interval and every change of frame.
    """
    symbolic_origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, 3.0)

    bb = VolumetricBoundingBox(-0.5, -1, 0, 0.5, 1, 1, symbolic_origin)

    assert isinstance(bb.origin, NumericTransform)
    assert np.array_equal(bb.origin.to_np(), symbolic_origin.to_np())


def test_a_box_carried_to_a_poses_origin_holds_it_as_numbers():
    """
    A pose names a place as a transformation matrix does, so a box carried to one holds
    its origin as numbers all the same.
    """
    body1, body2 = _body_turned_a_quarter_next_to_another()
    bb = VolumetricBoundingBox(-0.5, -1, 0, 0.5, 1, 1, NumericTransform.identity(body2))
    pose = body1.global_pose

    transformed = bb.transform_to_origin(pose)

    assert isinstance(transformed.origin, NumericTransform)
    assert np.array_equal(transformed.origin.to_np(), pose.to_np())


def test_reading_a_boxs_intervals_calls_no_casadi():
    bb = VolumetricBoundingBox(
        -0.5, -1, 0, 0.5, 1, 1, HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)
    )

    with CasadiCalls() as casadi_calls:
        intervals = [bb.x_interval, bb.y_interval, bb.z_interval]

    assert [(interval.lower, interval.upper) for interval in intervals] == [
        (0.5, 1.5),
        (1.0, 3.0),
        (3.0, 4.0),
    ]
    assert casadi_calls.calls_by_caller == Counter()


def test_carrying_a_box_into_another_frame_calls_no_casadi():
    body1, body2 = _body_turned_a_quarter_next_to_another()
    bb = VolumetricBoundingBox(-0.5, -1, 0, 0.5, 1, 1, NumericTransform.identity(body2))

    with CasadiCalls() as casadi_calls:
        transformed = bb.transform_to_origin(NumericTransform.identity(body1))

    assert (
        transformed.min_x,
        transformed.min_y,
        transformed.min_z,
        transformed.max_x,
        transformed.max_y,
        transformed.max_z,
    ) == pytest.approx((0.0, -0.5, 0.0, 2.0, 0.5, 1.0))
    assert casadi_calls.calls_by_caller == Counter()
