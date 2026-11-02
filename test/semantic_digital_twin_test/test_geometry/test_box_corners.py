import numpy as np
import pytest
from random_events.interval import Bound, SimpleInterval

from krrood.bounds import BoundsInterface
from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    AxisAlignedBox,
    CoordinateBounds,
    PlanarBoundingBox,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% boxes in space


@pytest.fixture
def origin() -> HomogeneousTransformationMatrix:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 2, 3, reference_frame=world.root
    )


@pytest.fixture(params=[PlanarBoundingBox, VolumetricBoundingBox])
def box(request, origin):
    if request.param is PlanarBoundingBox:
        return PlanarBoundingBox(-1, -2, 4, 5, origin)
    return VolumetricBoundingBox(-1, -2, -3, 4, 5, 6, origin)


def test_a_box_in_space_implements_the_bounds_interface(box):
    assert isinstance(box, BoundsInterface)


def test_a_box_has_its_corners_as_points_of_its_own_type(box):
    [point_type] = get_generic_type_parameters(type(box), AxisAlignedBox)
    assert (type(box.lower), type(box.upper)) == (point_type, point_type)


def test_the_corners_of_a_box_are_where_its_intervals_start_and_end(box):
    coordinates = box.to_array_bounds()
    dimensionality = box.dimensionality()
    assert np.array_equal(box.lower.to_np()[:dimensionality], coordinates.lower)
    assert np.array_equal(box.upper.to_np()[:dimensionality], coordinates.upper)


def test_the_corners_of_a_box_are_in_the_frame_of_its_origin(box, origin):
    assert box.lower.reference_frame is origin.reference_frame
    assert box.upper.reference_frame is origin.reference_frame


# %% coordinate bounds


def test_a_box_gives_its_coordinates_as_bounds_that_clip_segments(box):
    assert isinstance(box.to_array_bounds(), CoordinateBounds)


# %% clipping a segment


def test_a_segment_through_the_bounds_is_clipped_to_the_part_inside():
    bounds = CoordinateBounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    start = np.array([-1.0, 0.5])
    direction = np.array([4.0, 0.0])
    assert bounds.clip_segment(start, direction) == SimpleInterval.from_data(
        0.25, 0.5, Bound.CLOSED, Bound.CLOSED
    )


def test_a_segment_that_misses_the_bounds_is_not_clipped():
    bounds = CoordinateBounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    start = np.array([-1.0, 2.0])
    direction = np.array([4.0, 0.0])
    assert bounds.clip_segment(start, direction) is None
