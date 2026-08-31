import math

import pytest

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import SpatialTypeNotJsonSerializable
from semantic_digital_twin.spatial_types import Point2, Point3, Pose
from semantic_digital_twin.world_description.world_entity import Body


class TestPoint2Construction:
    def test_defaults(self):
        p = Point2()
        assert p.x.to_np() == pytest.approx(0)
        assert p.y.to_np() == pytest.approx(0)

    def test_explicit_values(self):
        p = Point2(x=1.0, y=2.0)
        assert p.x.to_np() == pytest.approx(1.0)
        assert p.y.to_np() == pytest.approx(2.0)

    def test_shape(self):
        p = Point2(x=1, y=2)
        assert p.shape == (2, 1)

    def test_has_no_z(self):
        """
        A 2D point has no height of its own -- unlike ``Point3``, ``Point2`` does not
        carry a ``z`` attribute at all.
        """
        p = Point2(x=1, y=2)
        with pytest.raises(AttributeError):
            p.z

    def test_reference_frame(self):
        frame = Body(name=PrefixedName("world"))
        p = Point2(x=1, y=2, reference_frame=frame)
        assert p.reference_frame is frame

    def test_setters(self):
        p = Point2()
        p.x = 5.0
        p.y = -3.0
        assert p.x.to_np() == pytest.approx(5.0)
        assert p.y.to_np() == pytest.approx(-3.0)


class TestPoint2ToPoint3:
    def test_to_point3_default_z(self):
        p2 = Point2(x=1.0, y=2.0)
        p3 = p2.to_point3()
        assert isinstance(p3, Point3)
        assert p3.x.to_np() == pytest.approx(1.0)
        assert p3.y.to_np() == pytest.approx(2.0)
        assert p3.z.to_np() == pytest.approx(0.0)

    def test_to_point3_explicit_z(self):
        p2 = Point2(x=1.0, y=2.0)
        p3 = p2.to_point3(z=5.0)
        assert p3.z.to_np() == pytest.approx(5.0)

    def test_to_point3_reference_frame_propagated(self):
        frame = Body(name=PrefixedName("world"))
        p2 = Point2(x=1.0, y=2.0, reference_frame=frame)
        p3 = p2.to_point3()
        assert p3.reference_frame is frame


class TestPoint2FromPose:
    def test_drops_z_roll_pitch_yaw(self):
        pose = Pose.from_xyz_rpy(x=1.5, y=-2.5, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        p2 = Point2.from_pose(pose)
        assert p2.x.to_np() == pytest.approx(1.5)
        assert p2.y.to_np() == pytest.approx(-2.5)

    def test_from_pose_inherits_reference_frame(self):
        frame = Body(name=PrefixedName("world"))
        pose = Pose.from_xyz_rpy(x=1.0, y=2.0, reference_frame=frame)
        p2 = Point2.from_pose(pose)
        assert p2.reference_frame is frame

    def test_from_pose_override_reference_frame(self):
        frame = Body(name=PrefixedName("world"))
        other_frame = Body(name=PrefixedName("other"))
        pose = Pose.from_xyz_rpy(x=1.0, y=2.0, reference_frame=frame)
        p2 = Point2.from_pose(pose, reference_frame=other_frame)
        assert p2.reference_frame is other_frame


class TestPoint2Symbolic:
    def test_symbolic_variable(self):
        x_var = sm.FloatVariable(name="x")
        y_var = sm.FloatVariable(name="y")
        p2 = Point2(x=x_var, y=y_var)
        assert not p2.is_constant()

    def test_deepcopy(self):
        from copy import deepcopy

        p2 = Point2(x=1.0, y=2.0)
        p2_copy = deepcopy(p2)
        assert p2_copy.x.to_np() == pytest.approx(1.0)
        assert p2_copy.y.to_np() == pytest.approx(2.0)


class TestPoint2JSON:
    def test_to_json_roundtrip(self):
        from krrood.adapters.json_serializer import to_json, from_json

        p2 = Point2(x=1.0, y=-2.0)
        data = to_json(p2)
        p2_restored = from_json(data)
        assert p2_restored.x.to_np() == pytest.approx(1.0, abs=1e-6)
        assert p2_restored.y.to_np() == pytest.approx(-2.0, abs=1e-6)

    def test_to_json_symbolic_raises(self):
        p2 = Point2(x=sm.FloatVariable(name="x"), y=0)
        with pytest.raises(SpatialTypeNotJsonSerializable):
            p2.to_json()

    def test_to_json_contains_data_key(self):
        p2 = Point2(x=1.0, y=2.0)
        j = p2.to_json()
        assert "data" in j
        assert len(j["data"]) == 2


class TestPoint2Hash:
    def test_hash_equal_objects(self):
        p1 = Point2(x=1.0, y=2.0)
        p2 = Point2(x=1.0, y=2.0)
        assert hash(p1) == hash(p2)

    def test_hash_different_objects(self):
        p1 = Point2(x=1.0, y=2.0)
        p2 = Point2(x=1.0, y=2.1)
        assert hash(p1) != hash(p2)
