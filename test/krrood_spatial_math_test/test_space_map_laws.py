"""
The laws every space map obeys, run against each implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood_spatial_math.point import Point
from krrood_spatial_math.line import Line
from krrood_spatial_math.space_map import (
    OrthogonalProjection,
    ProjectionLineT,
    SourcePointT,
    SpaceMap,
    TargetPointT,
)
from krrood_spatial_math.spatial_types import (
    CartesianPoint2Dimensional,
    EmbeddedPlaneProjection,
)

from .dataset.frames import NamedFrame, Room

# %% the space maps every law runs against


@dataclass
class SpaceMapCase:
    """
    A space map together with a target point it is tested at.
    """

    space_map: SpaceMap
    """
    The space map under test.
    """

    target_point: CartesianPoint2Dimensional
    """
    A point of the space map's target space.
    """


@pytest.fixture(
    params=[EmbeddedPlaneProjection],
    ids=lambda space_map_type: space_map_type.__name__,
)
def space_map_case(request: pytest.FixtureRequest) -> SpaceMapCase:
    """
    Each space map implementation, onto the plane of one frame.
    """
    frame = NamedFrame("plane", space=Room())
    return SpaceMapCase(
        space_map=request.param(plane_frame=frame),
        target_point=CartesianPoint2Dimensional(1.5, -2.0, reference_frame=frame),
    )


positions_along_line = [-3.0, 0.0, 0.5, 4.0]


# %% inverse laws


@pytest.mark.parametrize("position_along_line", positions_along_line)
def test_project_after_unproject_restores_the_target_point(
    space_map_case: SpaceMapCase, position_along_line: float
):
    """
    Every point unprojected from a target point projects back onto it.
    """
    space_map, target_point = space_map_case.space_map, space_map_case.target_point

    round_trip = space_map.project(
        space_map.unproject(target_point, position_along_line)
    )

    np.testing.assert_allclose(round_trip.to_np(), target_point.to_np())


@pytest.mark.parametrize("position_along_line", positions_along_line)
def test_unproject_lies_at_its_position_on_the_projection_line(
    space_map_case: SpaceMapCase, position_along_line: float
):
    """
    An unprojected point lies on the projection line, as many direction lengths from its
    point as its position along the line.
    """
    space_map, target_point = space_map_case.space_map, space_map_case.target_point
    projection_line = space_map.projection_line(target_point)

    unprojected = space_map.unproject(target_point, position_along_line)

    np.testing.assert_allclose(
        unprojected.to_np(),
        (
            projection_line.point + projection_line.direction * position_along_line
        ).to_np(),
    )


# %% the interface itself


def test_a_space_map_cannot_be_instantiated():
    """
    A space map is an interface, implemented but never instantiated.
    """
    with pytest.raises(InterfaceInstantiationError):
        SpaceMap()


@pytest.mark.parametrize("point_parameter", [SourcePointT, TargetPointT])
def test_a_space_map_sends_affine_points_to_affine_points(point_parameter: object):
    """
    Both the points a space map takes and those it gives are points given in a frame.
    """
    assert point_parameter.__bound__ is Point


def test_a_space_map_sends_points_along_straight_lines():
    assert ProjectionLineT.__bound__ is Line


# %% projections along the normal of a plane


def test_an_orthogonal_projection_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        OrthogonalProjection()


def test_an_orthogonal_projection_is_a_space_map():
    assert issubclass(OrthogonalProjection, SpaceMap)


def test_the_projection_onto_a_frames_plane_is_an_orthogonal_projection():
    assert issubclass(EmbeddedPlaneProjection, OrthogonalProjection)


def test_an_orthogonal_projection_declares_only_its_plane_and_where_points_lie_on_it():
    assert OrthogonalProjection.__abstractmethods__ == {"project", "embed_planar_point"}


def test_every_projection_line_of_an_orthogonal_projection_runs_along_the_plane_normal():
    projection = EmbeddedPlaneProjection(plane_frame=None)
    line = projection.projection_line(CartesianPoint2Dimensional(1.0, 2.0))

    np.testing.assert_array_equal(
        line.direction.to_np(), projection.plane_normal.to_np()
    )
    np.testing.assert_array_equal(
        line.point.to_np(),
        projection.embed_planar_point(
            point=CartesianPoint2Dimensional(1.0, 2.0)
        ).to_np(),
    )
