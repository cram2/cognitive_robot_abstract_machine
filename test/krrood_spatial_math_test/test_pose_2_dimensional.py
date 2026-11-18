"""
Planar poses: a position in the plane and a rotation about the z-axis.
"""

import math

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood_spatial_math.spatial_types import (
    CartesianPoint2Dimensional,
    Pose2Dimensional,
    Pose3Dimensional,
    CartesianPoint3Dimensional,
    Quaternion,
)
from krrood_spatial_math.spatial_types import RotationMatrix
from krrood_spatial_math.testing.orientation_comparison import assert_same_orientation

from .dataset.frames import NamedFrame
from .reference_implementations import quaternion_from_rpy


class TestPose2DimensionalConstruction:
    def test_defaults(self):
        p = Pose2Dimensional()
        assert p.x.to_np() == pytest.approx(0)
        assert p.y.to_np() == pytest.approx(0)
        assert p.yaw.to_np() == pytest.approx(0)

    def test_explicit_values(self):
        p = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=math.pi / 2)
        assert p.x.to_np() == pytest.approx(1.0)
        assert p.y.to_np() == pytest.approx(2.0)
        assert p.yaw.to_np() == pytest.approx(math.pi / 2)

    def test_has_no_height_or_tilt(self):
        """
        A 2D pose has no height or tilt of its own, so :class:`Pose2Dimensional` does not carry
        ``z``, ``roll`` or ``pitch`` at all.
        """
        p = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=0.5)
        with pytest.raises(AttributeError):
            p.z
        with pytest.raises(AttributeError):
            p.roll
        with pytest.raises(AttributeError):
            p.pitch

    def test_shape(self):
        p = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=0.5)
        assert p.shape == (3, 1)

    def test_reference_frame(self):
        frame = NamedFrame("world")
        p = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=0, reference_frame=frame)
        assert p.reference_frame is frame

    def test_setters(self):
        p = Pose2Dimensional()
        p.x = 5.0
        p.y = -3.0
        p.yaw = math.pi
        assert p.x.to_np() == pytest.approx(5.0)
        assert p.y.to_np() == pytest.approx(-3.0)
        assert p.yaw.to_np() == pytest.approx(math.pi)


class TestPose2DimensionalToPose:
    """
    Conversion of a planar pose into a full pose.
    """

    def test_to_pose_position(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=3.0, y=-1.5, yaw=0)
        pose = p2.pose
        assert isinstance(pose, Pose3Dimensional)
        assert pose.x.to_np() == pytest.approx(3.0)
        assert pose.y.to_np() == pytest.approx(-1.5)
        assert pose.z.to_np() == pytest.approx(0.0)

    def test_to_pose_yaw_only(self):
        yaw = math.pi / 4
        p2 = Pose2Dimensional.from_xy_yaw(x=0, y=0, yaw=yaw)
        pose = p2.pose
        _, _, actual_yaw = pose.rotation_matrix.rpy
        assert actual_yaw.to_np() == pytest.approx(yaw, abs=1e-6)

    def test_to_pose_roll_pitch_zero(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=1.0)
        pose = p2.pose
        roll, pitch, _ = pose.rotation_matrix.rpy
        assert roll.to_np() == pytest.approx(0.0, abs=1e-6)
        assert pitch.to_np() == pytest.approx(0.0, abs=1e-6)

    def test_to_pose_reference_frame_propagated(self):
        frame = NamedFrame("map")
        p2 = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=0, reference_frame=frame)
        assert p2.pose.reference_frame is frame

    def test_position_of_its_pose(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=2.0, y=-3.0, yaw=0)
        pt = p2.pose.position
        assert isinstance(pt, CartesianPoint3Dimensional)
        assert pt.x.to_np() == pytest.approx(2.0)
        assert pt.y.to_np() == pytest.approx(-3.0)
        assert pt.z.to_np() == pytest.approx(0.0)

    def test_quaternion(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=0, y=0, yaw=0)
        q = p2.quaternion
        assert isinstance(q, Quaternion)
        # identity quaternion: x=0, y=0, z=0, w=1
        expected = np.array([0, 0, 0, 1], dtype=float)
        assert np.allclose(q.to_np().flatten(), expected, atol=1e-6)

    def test_to_rotation_matrix(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=0, y=0, yaw=0)
        r = p2.rotation_matrix
        assert isinstance(r, RotationMatrix)
        assert np.allclose(r.to_np()[:3, :3], np.eye(3), atol=1e-6)

    def test_to_homogeneous_matrix(self):
        from krrood_spatial_math.spatial_types import (
            HomogeneousTransformationMatrix3Dimensional,
        )

        p2 = Pose2Dimensional.from_xy_yaw(x=1, y=2, yaw=0)
        m = p2.homogeneous_matrix
        assert isinstance(m, HomogeneousTransformationMatrix3Dimensional)
        assert m[0, 3].to_np() == pytest.approx(1.0)
        assert m[1, 3].to_np() == pytest.approx(2.0)

    def test_position_property(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=5, y=6, yaw=0.3)
        pt = p2.position
        assert isinstance(pt, CartesianPoint2Dimensional)
        assert pt.x.to_np() == pytest.approx(5.0)
        assert pt.y.to_np() == pytest.approx(6.0)

    def test_position_property_reference_frame_propagated(self):
        frame = NamedFrame("world")
        p2 = Pose2Dimensional.from_xy_yaw(x=5, y=6, yaw=0, reference_frame=frame)
        assert p2.position.reference_frame is frame

    def test_orientation_property(self):
        """
        The orientation of a planar pose is the quaternion of its yaw.
        """
        p2 = Pose2Dimensional.from_xy_yaw(x=0, y=0, yaw=0.3)
        q = p2.orientation
        assert isinstance(q, Quaternion)
        assert_same_orientation(q.to_np(), quaternion_from_rpy(0, 0, 0.3))


class TestPose2DimensionalFromPosition:
    def test_composes_position_and_yaw(self):
        position = CartesianPoint2Dimensional(x=1.5, y=-2.5)
        p2 = Pose2Dimensional(position, yaw=0.7)
        assert p2.x.to_np() == pytest.approx(1.5)
        assert p2.y.to_np() == pytest.approx(-2.5)
        assert p2.yaw.to_np() == pytest.approx(0.7)

    def test_inherits_position_reference_frame(self):
        frame = NamedFrame("world")
        position = CartesianPoint2Dimensional(x=1.0, y=2.0, reference_frame=frame)
        p2 = Pose2Dimensional(position, yaw=0.0)
        assert p2.reference_frame is frame

    def test_override_reference_frame(self):
        frame = NamedFrame("world")
        other_frame = NamedFrame("other")
        position = CartesianPoint2Dimensional(x=1.0, y=2.0, reference_frame=frame)
        p2 = Pose2Dimensional(position, yaw=0.0, reference_frame=other_frame)
        assert p2.reference_frame is other_frame


class TestPose2DimensionalFromPose:
    def test_roundtrip(self):
        original = Pose2Dimensional.from_xy_yaw(x=1.5, y=-2.5, yaw=0.7)
        pose3d = original.pose
        recovered = Pose2Dimensional.from_pose(pose3d)
        assert recovered.x.to_np() == pytest.approx(1.5, abs=1e-6)
        assert recovered.y.to_np() == pytest.approx(-2.5, abs=1e-6)
        assert recovered.yaw.to_np() == pytest.approx(0.7, abs=1e-6)

    def test_from_pose_inherits_reference_frame(self):
        frame = NamedFrame("base")
        pose = Pose3Dimensional.from_xyz_rpy(1, 2, 3, 0, 0, 0.5, reference_frame=frame)
        p2 = Pose2Dimensional.from_pose(pose)
        assert p2.reference_frame is frame


class TestPose2DimensionalSymbolic:
    def test_symbolic_variable(self):
        x_var = sm.FloatVariable(name="x")
        y_var = sm.FloatVariable(name="y")
        yaw_var = sm.FloatVariable(name="yaw")
        p2 = Pose2Dimensional.from_xy_yaw(x=x_var, y=y_var, yaw=yaw_var)
        assert not p2.is_constant()

    def test_deepcopy(self):
        from copy import deepcopy

        p2 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.5)
        p2_copy = deepcopy(p2)
        assert p2_copy.x.to_np() == pytest.approx(1.0)
        assert p2_copy.y.to_np() == pytest.approx(2.0)
        assert p2_copy.yaw.to_np() == pytest.approx(0.5)


class TestPose2DimensionalJSON:
    def test_to_json_roundtrip(self):
        from krrood.adapters.json_serializer import to_json, from_json

        p2 = Pose2Dimensional.from_xy_yaw(x=1.0, y=-2.0, yaw=0.3)
        data = to_json(p2)
        p2_restored = from_json(data)
        assert p2_restored.x.to_np() == pytest.approx(1.0, abs=1e-6)
        assert p2_restored.y.to_np() == pytest.approx(-2.0, abs=1e-6)
        assert p2_restored.yaw.to_np() == pytest.approx(0.3, abs=1e-6)

    def test_to_json_contains_data_key(self):
        p2 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.5)
        j = p2.to_json()
        assert "data" in j
        assert len(j["data"]) == 3


class TestPose2DimensionalHash:
    def test_hash_equal_objects(self):
        p1 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.5)
        p2 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.5)
        assert hash(p1) == hash(p2)

    def test_hash_different_objects(self):
        p1 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.5)
        p2 = Pose2Dimensional.from_xy_yaw(x=1.0, y=2.0, yaw=0.6)
        assert hash(p1) != hash(p2)
