import math
from copy import deepcopy

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    OrthogonalProjection,
    Point2,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class TestPoint2Construction:
    def test_defaults(self):
        point = Point2()
        assert point.x.to_np() == pytest.approx(0)
        assert point.y.to_np() == pytest.approx(0)

    def test_explicit_values(self):
        point = Point2(x=1.0, y=2.0)
        assert point.x.to_np() == pytest.approx(1.0)
        assert point.y.to_np() == pytest.approx(2.0)

    def test_shape(self):
        point = Point2(x=1, y=2)
        assert point.shape == (2, 1)

    def test_has_no_z(self):
        """
        A 2D point has no height of its own -- unlike ``Point3``, ``Point2`` does not
        carry a ``z`` attribute at all.
        """
        point = Point2(x=1, y=2)
        with pytest.raises(AttributeError):
            point.z

    def test_reference_frame(self):
        frame = Body(name=PrefixedName("world"))
        point = Point2(x=1, y=2, reference_frame=frame)
        assert point.reference_frame is frame

    def test_setters(self):
        point = Point2()
        point.x = 5.0
        point.y = -3.0
        assert point.x.to_np() == pytest.approx(5.0)
        assert point.y.to_np() == pytest.approx(-3.0)


@pytest.fixture
def rotated_mini_world(mini_world: World) -> World:
    """
    :func:`mini_world` with its tip turned a quarter turn about the root's z axis.
    """
    mini_world.connections[0].position = math.pi / 2
    return mini_world


class TestPoint2ToPoint3:
    def test_unprojects_at_the_given_height_in_its_own_frame(self):
        frame = Body(name=PrefixedName("world"))
        point = Point2(x=1.0, y=2.0, reference_frame=frame)

        unprojected = point.to_point3(height=5.0)

        assert unprojected.reference_frame is frame
        np.testing.assert_allclose(
            unprojected.to_np(),
            OrthogonalProjection(frame).unproject(point, height=5.0).to_np(),
        )

    def test_default_height_is_the_plane(self):
        point = Point2(x=1.0, y=2.0)

        np.testing.assert_allclose(
            point.to_point3().to_np(),
            OrthogonalProjection(None).unproject(point).to_np(),
        )


class TestPoint2FromPoint3:
    def test_defaults_to_the_plane_of_the_point_frame(self):
        frame = Body(name=PrefixedName("world"))
        point = Point3(1.0, 2.0, 3.0, reference_frame=frame)

        projected = Point2.from_point3(point)

        assert projected.reference_frame is frame
        np.testing.assert_allclose(
            projected.to_np(), OrthogonalProjection(frame).project(point).to_np()
        )

    def test_transforms_into_another_plane_frame(self, rotated_mini_world: World):
        root, tip = rotated_mini_world.root, rotated_mini_world.bodies[1]
        point = Point3(1.0, 0.0, 2.0, reference_frame=tip)

        projected = Point2.from_point3(point, plane_frame=root)

        assert projected.reference_frame is root
        np.testing.assert_allclose(
            projected.to_np(),
            OrthogonalProjection(root).project(point).to_np(),
            atol=1e-12,
        )


class TestPoint2FromPose:
    def test_projects_the_pose_position(self):
        frame = Body(name=PrefixedName("world"))
        pose = Pose.from_xyz_rpy(
            x=1.5, y=-2.5, z=3.0, roll=0.1, pitch=0.2, yaw=0.3, reference_frame=frame
        )

        projected = Point2.from_pose(pose)

        assert projected.reference_frame is frame
        np.testing.assert_allclose(
            projected.to_np(),
            OrthogonalProjection(frame).project(pose.position).to_np(),
        )

    def test_transforms_into_another_plane_frame(self, rotated_mini_world: World):
        root, tip = rotated_mini_world.root, rotated_mini_world.bodies[1]
        pose = Pose.from_xyz_rpy(x=1.0, z=2.0, yaw=0.3, reference_frame=tip)

        projected = Point2.from_pose(pose, plane_frame=root)

        assert projected.reference_frame is root
        np.testing.assert_allclose(
            projected.to_np(),
            OrthogonalProjection(root).project(pose.position).to_np(),
            atol=1e-12,
        )


class TestPoint2Symbolic:
    def test_symbolic_variable(self):
        x_variable = sm.FloatVariable(name="x")
        y_variable = sm.FloatVariable(name="y")
        point = Point2(x=x_variable, y=y_variable)
        assert not point.is_constant()

    def test_deepcopy(self):
        point = Point2(x=1.0, y=2.0)
        copied = deepcopy(point)
        assert copied.x.to_np() == pytest.approx(1.0)
        assert copied.y.to_np() == pytest.approx(2.0)


class TestPoint2JSON:
    def test_to_json_roundtrip(self):
        point = Point2(x=1.0, y=-2.0)
        data = to_json(point)
        restored = from_json(data)
        assert restored.x.to_np() == pytest.approx(1.0, abs=1e-6)
        assert restored.y.to_np() == pytest.approx(-2.0, abs=1e-6)

    def test_to_json_contains_data_key(self):
        point = Point2(x=1.0, y=2.0)
        data = point.to_json()
        assert "data" in data
        assert len(data["data"]) == 2


class TestPoint2Hash:
    def test_hash_equal_objects(self):
        first = Point2(x=1.0, y=2.0)
        second = Point2(x=1.0, y=2.0)
        assert hash(first) == hash(second)

    def test_hash_different_objects(self):
        first = Point2(x=1.0, y=2.0)
        second = Point2(x=1.0, y=2.1)
        assert hash(first) != hash(second)
