import math

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import SpatialTypeNotJsonSerializable
from semantic_digital_twin.spatial_types import Pose2D, Pose, Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import RotationMatrix
from semantic_digital_twin.world_description.world_entity import Body


class TestPose2DConstruction:
    def test_defaults(self):
        p = Pose2D()
        assert p.x.to_np() == pytest.approx(0)
        assert p.y.to_np() == pytest.approx(0)
        assert p.yaw.to_np() == pytest.approx(0)

    def test_explicit_values(self):
        p = Pose2D(x=1.0, y=2.0, yaw=math.pi / 2)
        assert p.x.to_np() == pytest.approx(1.0)
        assert p.y.to_np() == pytest.approx(2.0)
        assert p.yaw.to_np() == pytest.approx(math.pi / 2)

    def test_fixed_z_roll_pitch(self):
        p = Pose2D(x=1, y=2, yaw=0.5)
        assert p.z == 0
        assert p.roll == 0
        assert p.pitch == 0

    def test_shape(self):
        p = Pose2D(x=1, y=2, yaw=0.5)
        assert p.shape == (3, 1)

    def test_reference_frame(self):
        frame = Body(name=PrefixedName("world"))
        p = Pose2D(x=1, y=2, yaw=0, reference_frame=frame)
        assert p.reference_frame is frame

    def test_setters(self):
        p = Pose2D()
        p.x = 5.0
        p.y = -3.0
        p.yaw = math.pi
        assert p.x.to_np() == pytest.approx(5.0)
        assert p.y.to_np() == pytest.approx(-3.0)
        assert p.yaw.to_np() == pytest.approx(math.pi)


class TestPose2DToPose:
    def test_to_pose_position(self):
        p2 = Pose2D(x=3.0, y=-1.5, yaw=0)
        pose = p2.to_pose()
        assert isinstance(pose, Pose)
        assert pose.x.to_np() == pytest.approx(3.0)
        assert pose.y.to_np() == pytest.approx(-1.5)
        assert pose.z.to_np() == pytest.approx(0.0)

    def test_to_pose_yaw_only(self):
        yaw = math.pi / 4
        p2 = Pose2D(x=0, y=0, yaw=yaw)
        pose = p2.to_pose()
        _, _, actual_yaw = pose.to_rotation_matrix().to_rpy()
        assert actual_yaw.to_np() == pytest.approx(yaw, abs=1e-6)

    def test_to_pose_roll_pitch_zero(self):
        p2 = Pose2D(x=1, y=2, yaw=1.0)
        pose = p2.to_pose()
        roll, pitch, _ = pose.to_rotation_matrix().to_rpy()
        assert roll.to_np() == pytest.approx(0.0, abs=1e-6)
        assert pitch.to_np() == pytest.approx(0.0, abs=1e-6)

    def test_to_pose_reference_frame_propagated(self):
        frame = Body(name=PrefixedName("map"))
        p2 = Pose2D(x=1, y=2, yaw=0, reference_frame=frame)
        assert p2.to_pose().reference_frame is frame

    def test_to_position(self):
        p2 = Pose2D(x=2.0, y=-3.0, yaw=0)
        pt = p2.to_position()
        assert isinstance(pt, Point3)
        assert pt.x.to_np() == pytest.approx(2.0)
        assert pt.y.to_np() == pytest.approx(-3.0)
        assert pt.z.to_np() == pytest.approx(0.0)

    def test_to_quaternion(self):
        p2 = Pose2D(x=0, y=0, yaw=0)
        q = p2.to_quaternion()
        assert isinstance(q, Quaternion)
        # identity quaternion: x=0, y=0, z=0, w=1
        expected = np.array([0, 0, 0, 1], dtype=float)
        assert np.allclose(q.to_np().flatten(), expected, atol=1e-6)

    def test_to_rotation_matrix(self):
        p2 = Pose2D(x=0, y=0, yaw=0)
        r = p2.to_rotation_matrix()
        assert isinstance(r, RotationMatrix)
        assert np.allclose(r.to_np()[:3, :3], np.eye(3), atol=1e-6)

    def test_to_homogeneous_matrix(self):
        from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

        p2 = Pose2D(x=1, y=2, yaw=0)
        m = p2.to_homogeneous_matrix()
        assert isinstance(m, HomogeneousTransformationMatrix)
        assert m[0, 3].to_np() == pytest.approx(1.0)
        assert m[1, 3].to_np() == pytest.approx(2.0)

    def test_position_property(self):
        p2 = Pose2D(x=5, y=6, yaw=0)
        pt = p2.position
        assert pt.x.to_np() == pytest.approx(5.0)
        assert pt.y.to_np() == pytest.approx(6.0)

    def test_orientation_property(self):
        p2 = Pose2D(x=0, y=0, yaw=0)
        q = p2.orientation
        assert isinstance(q, Quaternion)


class TestPose2DFromPose:
    def test_roundtrip(self):
        original = Pose2D(x=1.5, y=-2.5, yaw=0.7)
        pose3d = original.to_pose()
        recovered = Pose2D.from_pose(pose3d)
        assert recovered.x.to_np() == pytest.approx(1.5, abs=1e-6)
        assert recovered.y.to_np() == pytest.approx(-2.5, abs=1e-6)
        assert recovered.yaw.to_np() == pytest.approx(0.7, abs=1e-6)

    def test_from_pose_inherits_reference_frame(self):
        frame = Body(name=PrefixedName("base"))
        pose = Pose.from_xyz_rpy(1, 2, 3, 0, 0, 0.5, reference_frame=frame)
        p2 = Pose2D.from_pose(pose)
        assert p2.reference_frame is frame

    def test_from_pose_override_reference_frame(self):
        frame1 = Body(name=PrefixedName("a"))
        frame2 = Body(name=PrefixedName("b"))
        pose = Pose.from_xyz_rpy(0, 0, 0, 0, 0, 0, reference_frame=frame1)
        p2 = Pose2D.from_pose(pose, reference_frame=frame2)
        assert p2.reference_frame is frame2


class TestPose2DSymbolic:
    def test_symbolic_variable(self):
        x_var = sm.FloatVariable(name="x")
        y_var = sm.FloatVariable(name="y")
        yaw_var = sm.FloatVariable(name="yaw")
        p2 = Pose2D(x=x_var, y=y_var, yaw=yaw_var)
        assert not p2.is_constant()

    def test_deepcopy(self):
        from copy import deepcopy

        p2 = Pose2D(x=1.0, y=2.0, yaw=0.5)
        p2_copy = deepcopy(p2)
        assert p2_copy.x.to_np() == pytest.approx(1.0)
        assert p2_copy.y.to_np() == pytest.approx(2.0)
        assert p2_copy.yaw.to_np() == pytest.approx(0.5)


class TestPose2DJSON:
    def test_to_json_roundtrip(self):
        from krrood.adapters.json_serializer import to_json, from_json

        p2 = Pose2D(x=1.0, y=-2.0, yaw=0.3)
        data = to_json(p2)
        p2_restored = from_json(data)
        assert p2_restored.x.to_np() == pytest.approx(1.0, abs=1e-6)
        assert p2_restored.y.to_np() == pytest.approx(-2.0, abs=1e-6)
        assert p2_restored.yaw.to_np() == pytest.approx(0.3, abs=1e-6)

    def test_to_json_symbolic_raises(self):
        p2 = Pose2D(x=sm.FloatVariable(name="x"), y=0, yaw=0)
        with pytest.raises(SpatialTypeNotJsonSerializable):
            p2.to_json()

    def test_to_json_contains_data_key(self):
        p2 = Pose2D(x=1.0, y=2.0, yaw=0.5)
        j = p2.to_json()
        assert "data" in j
        assert len(j["data"]) == 3


class TestPose2DHash:
    def test_hash_equal_objects(self):
        p1 = Pose2D(x=1.0, y=2.0, yaw=0.5)
        p2 = Pose2D(x=1.0, y=2.0, yaw=0.5)
        assert hash(p1) == hash(p2)

    def test_hash_different_objects(self):
        p1 = Pose2D(x=1.0, y=2.0, yaw=0.5)
        p2 = Pose2D(x=1.0, y=2.0, yaw=0.6)
        assert hash(p1) != hash(p2)
