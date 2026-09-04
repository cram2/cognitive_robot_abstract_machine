from dataclasses import dataclass

import pytest
from random_events.interval import SimpleInterval
from typing_extensions import List, Tuple

from typing_extensions import get_args

from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import Point2, Point3
from semantic_digital_twin.world_description.geometry import (
    AxisAlignedBox,
    PointT,
    PlanarBoundingBox,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    BoxT,
)

# %% a box says which point type it is about


@pytest.mark.parametrize(
    "box_type, point_type",
    [(PlanarBoundingBox, Point2), (VolumetricBoundingBox, Point3)],
)
def test_a_box_binds_the_point_type_it_is_asked_about(box_type, point_type):
    """
    Each box type declares the point type it handles, so asking a box about a point of
    the other dimensionality is a type error rather than something that only surfaces
    once ``contains`` reads a coordinate the point does not have.
    """
    assert get_generic_type_parameters(box_type, AxisAlignedBox) == [point_type]


def test_a_collection_is_generic_over_its_box_and_point_types():
    """
    A collection forwards ``contains`` to its boxes, so it carries the same pair they do
    rather than widening back to any point.
    """
    assert BoundingBoxCollection.__parameters__ == (BoxT, PointT)


@pytest.mark.parametrize(
    "box_type, point_type",
    [(PlanarBoundingBox, Point2), (VolumetricBoundingBox, Point3)],
)
def test_a_collection_records_the_pair_it_was_given(box_type, point_type):
    """
    Both parameters reach the subscripted type, so a reader of the annotation sees which
    point the collection is about.
    """
    assert get_args(BoundingBoxCollection[box_type, point_type]) == (
        box_type,
        point_type,
    )


def test_a_box_must_implement_contains():
    """
    Every caller of a box reaches ``contains`` through the base, so the base is where
    the method is promised -- a box that does not answer it is not a box.
    """

    @dataclass(eq=False)
    class BoxWithoutContains(AxisAlignedBox[Point2]):
        @classmethod
        def axes(cls) -> Tuple[SpatialVariables, ...]:
            return SpatialVariables.x, SpatialVariables.y

        @property
        def _ordered_intervals(self) -> Tuple[SimpleInterval, ...]:
            return ()

        def get_points(self) -> List[Point2]:
            return []

        @classmethod
        def from_simple_event(cls, simple_event, origin):
            return []

    with pytest.raises(TypeError):
        BoxWithoutContains()
