import numpy as np
import pytest

from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.spatial_types import (
    Line3Dimensional,
    EmbeddedPlaneProjection,
    CartesianPoint2Dimensional,
    CartesianPoint3Dimensional,
    Pose3Dimensional,
    Vector3Dimensional,
)
from .dataset.frames import NamedFrame

# %% fixtures


# %% project


class TestProject:
    def test_drops_height_in_the_plane_frame(self):
        frame = NamedFrame("world")
        point = CartesianPoint3Dimensional(1.5, -2.5, 3.0, reference_frame=frame)

        projected = EmbeddedPlaneProjection(plane_frame=frame).project(point)

        assert isinstance(projected, CartesianPoint2Dimensional)
        assert projected.reference_frame is frame
        np.testing.assert_allclose(projected.to_np(), [1.5, -2.5])

    def test_pose_position_projects_like_a_point(self):
        frame = NamedFrame("world")
        pose = Pose3Dimensional.from_xyz_rpy(
            x=1.5, y=-2.5, z=3.0, roll=0.1, pitch=0.2, yaw=0.3, reference_frame=frame
        )

        projected = EmbeddedPlaneProjection(plane_frame=frame).project(pose.position)

        np.testing.assert_allclose(projected.to_np(), [1.5, -2.5])


# %% unproject


class TestUnproject:
    def test_default_height_is_the_plane(self):
        frame = NamedFrame("world")
        point = CartesianPoint2Dimensional(1.0, 2.0, reference_frame=frame)

        unprojected = EmbeddedPlaneProjection(plane_frame=frame).unproject(point)

        assert isinstance(unprojected, CartesianPoint3Dimensional)
        assert unprojected.reference_frame is frame
        np.testing.assert_allclose(unprojected.to_np()[:3], [1.0, 2.0, 0.0])

    def test_explicit_height(self):
        point = CartesianPoint2Dimensional(1.0, 2.0)

        unprojected = EmbeddedPlaneProjection(plane_frame=None).unproject(
            point, position_along_line=5.0
        )

        np.testing.assert_allclose(unprojected.to_np()[:3], [1.0, 2.0, 5.0])


# %% inverse laws


class TestInverseLaws:
    """
    ``project(unproject(p, h)) == p`` holds for every point on the plane, and
    ``unproject(project(q), h) == q`` holds only for a point at height ``h``.
    """

    def test_project_after_unproject_restores_a_point_on_the_plane(self):
        frame = NamedFrame("world")
        point = CartesianPoint2Dimensional(1.0, 2.0, reference_frame=frame)
        projection = EmbeddedPlaneProjection(plane_frame=frame)

        round_trip = projection.project(
            projection.unproject(point, position_along_line=4.0)
        )

        np.testing.assert_allclose(round_trip.to_np(), point.to_np())

    def test_project_after_unproject_rejects_a_point_on_another_plane(self):
        frame = NamedFrame("world")
        other_frame = NamedFrame("other")
        point = CartesianPoint2Dimensional(1.0, 2.0, reference_frame=other_frame)

        with pytest.raises(ReferenceFrameMismatchError):
            EmbeddedPlaneProjection(plane_frame=frame).unproject(point)

    def test_unproject_after_project_restores_a_point_at_the_unprojection_height(self):
        frame = NamedFrame("world")
        projection = EmbeddedPlaneProjection(plane_frame=frame)
        point = CartesianPoint3Dimensional(1.0, 2.0, 4.0, reference_frame=frame)

        round_trip = projection.unproject(
            projection.project(point), position_along_line=4.0
        )

        np.testing.assert_allclose(round_trip.to_np(), point.to_np())

    def test_unproject_after_project_moves_a_point_off_the_unprojection_height(self):
        frame = NamedFrame("world")
        projection = EmbeddedPlaneProjection(plane_frame=frame)
        point = CartesianPoint3Dimensional(1.0, 2.0, 3.0, reference_frame=frame)
        height = 4.0

        round_trip = projection.unproject(
            projection.project(point), position_along_line=height
        )

        np.testing.assert_allclose(round_trip.to_np()[:3], [*point.to_np()[:2], height])


# %% plane without reference frame


class TestPlaneWithoutReferenceFrame:
    """
    A projection with no :attr:`EmbeddedPlaneProjection.plane_frame` relates only points
    that have no reference frame either.
    """

    def test_projects_a_point_without_reference_frame(self):
        point = CartesianPoint3Dimensional(1.0, 2.0, 3.0)

        projected = EmbeddedPlaneProjection(plane_frame=None).project(point)

        assert projected.reference_frame is None
        np.testing.assert_allclose(projected.to_np(), point.to_np()[:2])

    def test_rejects_projecting_a_point_with_reference_frame(self):
        point = CartesianPoint3Dimensional(
            1.0, 2.0, 3.0, reference_frame=NamedFrame("world")
        )

        with pytest.raises(ReferenceFrameMismatchError):
            EmbeddedPlaneProjection(plane_frame=None).project(point)

    def test_rejects_unprojecting_a_point_with_reference_frame(self):
        point = CartesianPoint2Dimensional(
            1.0, 2.0, reference_frame=NamedFrame("world")
        )

        with pytest.raises(ReferenceFrameMismatchError):
            EmbeddedPlaneProjection(plane_frame=None).unproject(point)

    def test_plane_with_reference_frame_rejects_projecting_a_point_without_one(self):
        frame = NamedFrame("world")

        with pytest.raises(ReferenceFrameMismatchError):
            EmbeddedPlaneProjection(plane_frame=frame).project(
                CartesianPoint3Dimensional(1.0, 2.0, 3.0)
            )

    def test_plane_with_reference_frame_rejects_unprojecting_a_point_without_one(self):
        frame = NamedFrame("world")

        with pytest.raises(ReferenceFrameMismatchError):
            EmbeddedPlaneProjection(plane_frame=frame).unproject(
                CartesianPoint2Dimensional(1.0, 2.0)
            )


# %% projection_line


class TestProjectionLine:
    def test_is_the_vertical_line_through_the_unprojected_point(self):
        frame = NamedFrame("world")
        point = CartesianPoint2Dimensional(1.0, 2.0, reference_frame=frame)
        projection = EmbeddedPlaneProjection(plane_frame=frame)

        line = projection.projection_line(point)

        assert isinstance(line, Line3Dimensional)
        np.testing.assert_allclose(
            line.point.to_np(), projection.unproject(point).to_np()
        )
        np.testing.assert_allclose(
            line.direction.to_np(), Vector3Dimensional.Z(reference_frame=frame).to_np()
        )
        assert line.direction.reference_frame is frame

    def test_every_point_of_the_line_projects_onto_the_point(self):
        frame = NamedFrame("world")
        point = CartesianPoint2Dimensional(1.0, 2.0, reference_frame=frame)
        projection = EmbeddedPlaneProjection(plane_frame=frame)
        line = projection.projection_line(point)

        point_on_line = line.point + line.direction * 7.0

        np.testing.assert_allclose(
            projection.project(point_on_line).to_np(), point.to_np()
        )
