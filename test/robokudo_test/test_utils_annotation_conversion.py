# %% Imports

import sys
from dataclasses import replace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from robokudo.cas import CAS
from robokudo.types.annotation import (
    PoseAnnotation,
    StampedPoseAnnotation,
    PositionAnnotation,
    SemanticColor,
    Classification,
    StampedPositionAnnotation,
    Shape,
    Cuboid,
    Cylinder,
    Sphere,
    LocationAnnotation,
)
from robokudo.types.core import Annotation
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import BoundingBox3D
from robokudo.utils.annotation_conversion import (
    PoseAnnotationToStampedPoseAnnotationConverter,
    PositionAnnotationToStampedPoseAnnotationConverter,
    SemanticColor2ODConverter,
    Classification2ODConverter,
    StampedPose2ODConverter,
    Pose2ODConverter,
    Position2ODConverter,
    StampedPosition2ODConverter,
    BoundingBox3DForShapeSizeConverter,
    Shape2ODConverter,
    Cuboid2ODConverter,
    Cylinder2ODConverter,
    Sphere2ODConverter,
    Location2ODConverter,
)
from robokudo_msgs.msg import ObjectDesignator, ShapeSize

from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    Box as SemDTBox,
    Cylinder as SemDTCylinder,
    Sphere as SemDTSphere,
)
from semantic_digital_twin.world_description.world_entity import Body, Region
from . import _assertions

# %% Camera behavior


class TestUtilsAnnotationConversion(object):
    @pytest.fixture
    def cas_with_tf(self):
        cas = CAS()

        camera_frame = "some_weird_non_default_frame_id"
        timestamp_nanoseconds = 123_000_000_456
        world_body = Body(name=PrefixedName(name="map"))
        camera_body = Body(name=PrefixedName(name=camera_frame))
        camera_model = PinholeCameraModel(
            image_resolution=CameraResolution(width=1024, height=1280),
            focal_length_x=1050.0,
            focal_length_y=639.5,
            principal_point=Point2(x=512.0, y=479.5),
        )
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.5,
            pos_y=0.5,
            pos_z=0.5,
            quat_x=-0.5,
            quat_y=0.5,
            quat_z=-0.5,
            quat_w=0.5,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        cas.camera_observation = CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"),
                root=camera_body,
                camera_model=camera_model,
            ),
            effective_camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=timestamp_nanoseconds,
        )
        return cas

    def test_pose_annotation_to_stamped_pose_annotation_can_convert(self):
        converter = PoseAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        position_annotation = PositionAnnotation()

        assert converter.can_convert(pose_annotation, StampedPoseAnnotation)

        assert not converter.can_convert(pose_annotation, PoseAnnotation)

        assert not converter.can_convert(position_annotation, StampedPoseAnnotation)

    def test_pose_annotation_to_stamped_pose_annotation_convert(self):
        converter = PoseAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_no_default_source"
        pose_annotation.rotation = np.random.rand(4)
        pose_annotation.translation = np.random.rand(3)

        stamped_pose_annotation = converter.convert(pose_annotation)

        assert stamped_pose_annotation.source == pose_annotation.source
        assert np.all(stamped_pose_annotation.rotation == pose_annotation.rotation)
        assert np.all(
            stamped_pose_annotation.translation == pose_annotation.translation
        )

    def test_position_annotation_to_stamped_pose_annotation_can_convert(self):
        converter = PositionAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        position_annotation = PositionAnnotation()

        assert converter.can_convert(position_annotation, StampedPoseAnnotation)

        assert not converter.can_convert(pose_annotation, StampedPoseAnnotation)

        assert not converter.can_convert(position_annotation, PoseAnnotation)

    def test_position_annotation_to_stamped_pose_annotation_convert(self):
        converter = PositionAnnotationToStampedPoseAnnotationConverter()

        pos_annotation = PositionAnnotation()
        pos_annotation.source = "some_weird_no_default_source"
        pos_annotation.translation = np.random.rand(3)

        stamped_pose_annotation = converter.convert(pos_annotation)

        assert stamped_pose_annotation.source == pos_annotation.source
        assert np.all(stamped_pose_annotation.translation == pos_annotation.translation)

    def test_semantic_color_2_od_converter_can_convert(self):
        converter = SemanticColor2ODConverter()
        assert converter.can_convert(SemanticColor())
        assert not converter.can_convert(Annotation())

    def test_semantic_color_2_od_converter_convert(self):
        cas = CAS()
        od = ObjectDesignator()

        converter = SemanticColor2ODConverter()

        semantic_color = SemanticColor()
        semantic_color.color = "some_weird_non_default_color"

        converter.convert(semantic_color, cas, od)

        assert len(od.color) == 1
        assert od.color[0] == semantic_color.color

    def test_classification_2_od_converter_can_convert(self):
        converter = Classification2ODConverter()
        assert converter.can_convert(Classification())
        assert not converter.can_convert(Annotation())

    def test_classification_2_od_converter_convert(self):
        cas = CAS()
        od = ObjectDesignator()

        converter = Classification2ODConverter()

        classification = Classification()
        classification.classname = "some_weird_non_default_classname"

        converter.convert(classification, cas, od)

        assert od.type == classification.classname

    def test_stamped_pose_2_od_converter_can_convert(self):
        converter = StampedPose2ODConverter()
        assert converter.can_convert(StampedPoseAnnotation())
        assert not converter.can_convert(Annotation())

    def test_stamped_pose_2_od_converter_convert(self):
        cas = CAS()
        od = ObjectDesignator()

        converter = StampedPose2ODConverter()

        stamped_pose_annotation = StampedPoseAnnotation()
        stamped_pose_annotation.source = "some_weird_non_default_source"
        stamped_pose_annotation.translation = np.random.rand(3)
        stamped_pose_annotation.rotation = np.random.rand(4)
        stamped_pose_annotation.frame = "some_weird_non_default_frame_id"
        stamped_pose_annotation.timestamp = Time(sec=123, nanosec=456)

        converter.convert(stamped_pose_annotation, cas, od)

        assert len(od.pose) == 1
        pose: PoseStamped = od.pose[0]
        assert pose.header.stamp == stamped_pose_annotation.timestamp
        assert pose.header.frame_id == stamped_pose_annotation.frame
        assert pose.pose.position.x == stamped_pose_annotation.translation[0]
        assert pose.pose.position.y == stamped_pose_annotation.translation[1]
        assert pose.pose.position.z == stamped_pose_annotation.translation[2]
        assert pose.pose.orientation.x == stamped_pose_annotation.rotation[0]
        assert pose.pose.orientation.y == stamped_pose_annotation.rotation[1]
        assert pose.pose.orientation.z == stamped_pose_annotation.rotation[2]
        assert pose.pose.orientation.w == stamped_pose_annotation.rotation[3]

    def test_pose_2_od_converter_can_convert(self):
        converter = Pose2ODConverter()
        assert converter.can_convert(PoseAnnotation())
        assert not converter.can_convert(Annotation())

    def test_pose_2_od_converter_convert_in_camera(self, cas_with_tf: CAS):
        cas_with_tf.camera_observation = replace(
            cas_with_tf.require_camera_observation(), world_T_camera=None
        )
        od = ObjectDesignator()

        converter = Pose2ODConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_non_default_source"
        pose_annotation.translation = np.random.rand(3)
        pose_annotation.rotation = np.random.rand(4)

        converter.convert(pose_annotation, cas_with_tf, od)

        assert len(od.pose) == 1
        pose: PoseStamped = od.pose[0]
        observation = cas_with_tf.require_camera_observation()
        assert pose.header.frame_id == observation.camera.root.name.name
        assert (
            pose.header.stamp.sec == observation.timestamp_nanoseconds // 1_000_000_000
        )
        assert (
            pose.header.stamp.nanosec
            == observation.timestamp_nanoseconds % 1_000_000_000
        )
        assert pose.pose.position.x == pose_annotation.translation[0]
        assert pose.pose.position.y == pose_annotation.translation[1]
        assert pose.pose.position.z == pose_annotation.translation[2]
        assert pose.pose.orientation.x == pose_annotation.rotation[0]
        assert pose.pose.orientation.y == pose_annotation.rotation[1]
        assert pose.pose.orientation.z == pose_annotation.rotation[2]
        assert pose.pose.orientation.w == pose_annotation.rotation[3]

    def test_pose_2_od_converter_convert_in_world(self, cas_with_tf: CAS):
        camera_to_world_quat = (
            cas_with_tf.require_camera_observation()
            .world_T_camera_or_raise()
            .quaternion.to_list()
        )
        od = ObjectDesignator()

        converter = Pose2ODConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_non_default_source"
        pose_annotation.translation = np.random.rand(3)
        pose_annotation.rotation = np.random.rand(4)

        converter.convert(pose_annotation, cas_with_tf, od)

        new_rotation = (
            R.from_quat(camera_to_world_quat) * R.from_quat(pose_annotation.rotation)
        ).as_quat(True)

        assert len(od.pose) == 1
        pose: PoseStamped = od.pose[0]
        assert pose.header.frame_id == "map"
        observation = cas_with_tf.require_camera_observation()
        assert (
            pose.header.stamp.sec == observation.timestamp_nanoseconds // 1_000_000_000
        )
        assert (
            pose.header.stamp.nanosec
            == observation.timestamp_nanoseconds % 1_000_000_000
        )
        assert pose.pose.position.x == pose_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -pose_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -pose_annotation.translation[1] + 0.5
        assert np.isclose(pose.pose.orientation.x, new_rotation[0])
        assert np.isclose(pose.pose.orientation.y, new_rotation[1])
        assert np.isclose(pose.pose.orientation.z, new_rotation[2])
        assert np.isclose(pose.pose.orientation.w, new_rotation[3])

    def test_position_2_od_converter_can_convert(self):
        converter = Position2ODConverter()

        position_annotation = PositionAnnotation()
        other_ann = Annotation()

        assert converter.can_convert(position_annotation)
        assert not converter.can_convert(other_ann)

    def test_position_2_od_converter_convert(self, cas_with_tf: CAS):
        camera_to_world_quat = (
            cas_with_tf.require_camera_observation()
            .world_T_camera_or_raise()
            .quaternion.to_list()
        )

        od = ObjectDesignator()

        converter = Position2ODConverter()

        position_annotation = PositionAnnotation()
        position_annotation.source = "some_weird_non_default_source"
        position_annotation.translation = np.random.rand(3)

        converter.convert(position_annotation, cas_with_tf, od)

        assert len(od.pose) == 1
        pose: PoseStamped = od.pose[0]
        assert pose.pose.position.x == position_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -position_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -position_annotation.translation[1] + 0.5
        _assertions.assert_quat_equal_up_to_sign(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            camera_to_world_quat,
        )

    def test_stamped_position_2_od_converter_can_convert(self):
        converter = StampedPosition2ODConverter()
        assert converter.can_convert(StampedPositionAnnotation())
        assert not converter.can_convert(Annotation())

    def test_stamped_position_2_od_converter_convert(self, cas_with_tf: CAS):
        camera_to_world_quat = (
            cas_with_tf.require_camera_observation()
            .world_T_camera_or_raise()
            .quaternion.to_list()
        )
        od = ObjectDesignator()

        converter = StampedPosition2ODConverter()

        stamped_pose_annotation = StampedPoseAnnotation()
        stamped_pose_annotation.source = "some_weird_non_default_source"
        stamped_pose_annotation.translation = np.random.rand(3)
        stamped_pose_annotation.rotation = np.random.rand(4)
        stamped_pose_annotation.frame = "some_weird_non_default_frame_id"
        stamped_pose_annotation.timestamp = Time(sec=123, nanosec=456)

        converter.convert(stamped_pose_annotation, cas_with_tf, od)

        assert len(od.pose) == 1
        pose: PoseStamped = od.pose[0]
        assert pose.header.stamp == stamped_pose_annotation.timestamp
        assert pose.header.frame_id == stamped_pose_annotation.frame
        assert pose.pose.position.x == stamped_pose_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -stamped_pose_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -stamped_pose_annotation.translation[1] + 0.5
        _assertions.assert_quat_equal_up_to_sign(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            camera_to_world_quat,
        )

    def test_bounding_box_3d_for_shape_size_converter_can_convert(self):
        converter = BoundingBox3DForShapeSizeConverter()
        assert converter.can_convert(BoundingBox3D())
        assert not converter.can_convert(Annotation())

    def test_bounding_box_3d_for_shape_size_converter_convert(self):
        od = ObjectDesignator()
        cas = CAS()

        converter = BoundingBox3DForShapeSizeConverter()

        bbox3d = BoundingBox3D()
        bbox3d.x_length = np.random.randint(sys.maxsize)
        bbox3d.y_length = np.random.randint(sys.maxsize)
        bbox3d.z_length = np.random.randint(sys.maxsize)

        converter.convert(bbox3d, cas, od)

        assert (
            od.size == ""
        )  # TODO: Correct assertion when implementation of converter is complete
        assert len(od.shape_size) == 1
        shape_size: ShapeSize = od.shape_size[0]
        assert shape_size.dimensions.x == float(bbox3d.x_length)
        assert shape_size.dimensions.y == float(bbox3d.y_length)
        assert shape_size.dimensions.z == float(bbox3d.z_length)
        assert shape_size.radius == 0.0

    def test_shape_2_od_converter_can_convert(self):
        converter = Shape2ODConverter()
        assert converter.can_convert(Shape())
        assert not converter.can_convert(Annotation())

    def test_shape_2_od_converter_convert(self):
        od = ObjectDesignator()

        converter = Shape2ODConverter()

        shape = Shape()
        shape.geometry = SemDTSphere(radius=0.17)

        converter.convert(shape, CAS(), od)

        assert len(od.shape) == 1
        assert od.shape[0] == shape.shape_name

    def test_cuboid_2_od_converter_can_convert(self):
        converter = Cuboid2ODConverter()
        assert converter.can_convert(Cuboid())
        assert not converter.can_convert(Annotation())

    def test_cuboid_2_od_converter_convert(self):
        od = ObjectDesignator()

        converter = Cuboid2ODConverter()

        shape = Cuboid()
        shape.geometry = SemDTBox()

        converter.convert(shape, CAS(), od)

        assert len(od.shape) == 1
        assert od.shape[0] == shape.shape_name

    def test_sphere_2_od_converter_can_convert(self):
        converter = Sphere2ODConverter()
        assert converter.can_convert(Sphere())
        assert not converter.can_convert(Annotation())

    def test_cylinder_2_od_converter_can_convert(self):
        converter = Cylinder2ODConverter()
        assert converter.can_convert(Cylinder())
        assert not converter.can_convert(Annotation())

    def test_cylinder_2_od_converter_convert(self):
        od = ObjectDesignator()

        converter = Cylinder2ODConverter()

        shape = Cylinder()
        shape.geometry = SemDTCylinder(width=0.22, height=0.41)

        converter.convert(shape, CAS(), od)

        assert len(od.shape) == 1
        assert od.shape[0] == shape.shape_name

    def test_sphere_2_od_converter_convert(self):
        od = ObjectDesignator()

        converter = Sphere2ODConverter()

        shape = Sphere()
        shape.geometry = SemDTSphere(radius=np.random.random())

        converter.convert(shape, CAS(), od)

        assert len(od.shape) == 1
        assert od.shape[0] == shape.shape_name

        assert len(od.shape_size) == 1
        assert od.shape_size[0].radius == shape.geometry.radius

    def test_location_2_od_converter_can_convert(self):
        converter = Location2ODConverter()
        assert converter.can_convert(LocationAnnotation())
        assert not converter.can_convert(Annotation())

    def test_location_2_od_converter_convert(self):
        od = ObjectDesignator()

        converter = Location2ODConverter()

        loc = LocationAnnotation()
        loc.region = Region(name=PrefixedName(name="some_weird_non_default_name"))

        converter.convert(loc, CAS(), od)

        assert od.location == str(loc.region.name)

    def test_location_2_od_converter_convert_with_legacy_name_fallback(self):
        od = ObjectDesignator()

        converter = Location2ODConverter()

        loc = LocationAnnotation()
        loc.name = "legacy_location_name"

        converter.convert(loc, CAS(), od)

        assert od.location == loc.name
