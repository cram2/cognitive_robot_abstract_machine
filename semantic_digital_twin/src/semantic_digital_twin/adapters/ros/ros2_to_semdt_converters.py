from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geometry_msgs.msg as geometry_msgs
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from .msg_converter import Ros2ToSemDTConverter, InputType, OutputType
from ...spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    Quaternion,
)
from ...spatial_types.spatial_types import Pose
from ...world import World
from ...world_description.geometry import Color, Box, Scale, Cylinder, Sphere, FileMesh


@dataclass
class TransformStampedToSemDTConverter(
    Ros2ToSemDTConverter[
        geometry_msgs.TransformStamped, HomogeneousTransformationMatrix
    ]
):

    @classmethod
    def convert(cls, data: InputType, world: World) -> OutputType:
        result = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=data.transform.translation.x,
            pos_y=data.transform.translation.y,
            pos_z=data.transform.translation.z,
            quat_x=data.transform.rotation.x,
            quat_y=data.transform.rotation.y,
            quat_z=data.transform.rotation.z,
            quat_w=data.transform.rotation.w,
        )
        if data.child_frame_id != "":
            result.reference_frame = world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            )
        if data.child_frame_id != "":
            result.child_frame = world.get_kinematic_structure_entity_by_name(
                data.child_frame_id
            )
        return result


@dataclass
class PoseStampedToSemDTConverter(
    Ros2ToSemDTConverter[geometry_msgs.PoseStamped, Pose]
):

    @classmethod
    def convert(cls, data: geometry_msgs.PoseStamped, world: World) -> Pose:
        result = Pose.from_xyz_quaternion(
            pos_x=data.pose.position.x,
            pos_y=data.pose.position.y,
            pos_z=data.pose.position.z,
            quat_x=data.pose.orientation.x,
            quat_y=data.pose.orientation.y,
            quat_z=data.pose.orientation.z,
            quat_w=data.pose.orientation.w,
        )
        if data.header.frame_id != "":
            result.reference_frame = world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            )
        return result


@dataclass
class PoseToSemDTConverter(Ros2ToSemDTConverter[geometry_msgs.Pose, Pose]):

    @classmethod
    def convert(cls, data: geometry_msgs.Pose, world: World) -> Pose:
        result = Pose.from_xyz_quaternion(
            pos_x=data.position.x,
            pos_y=data.position.y,
            pos_z=data.position.z,
            quat_x=data.orientation.x,
            quat_y=data.orientation.y,
            quat_z=data.orientation.z,
            quat_w=data.orientation.w,
        )
        return result


@dataclass
class PointStampedToSemDTConverter(
    Ros2ToSemDTConverter[geometry_msgs.PointStamped, Point3]
):

    @classmethod
    def convert(cls, data: OutputType, world: World) -> InputType:
        result = Point3(
            data.point.x,
            data.point.y,
            data.point.z,
        )
        if data.header.frame_id != "":
            result.reference_frame = world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            )
        return result


@dataclass
class Vector3StampedToSemDTConverter(
    Ros2ToSemDTConverter[geometry_msgs.Vector3Stamped, Vector3]
):

    @classmethod
    def convert(cls, data: geometry_msgs.Vector3Stamped, world: World) -> Vector3:
        result = Vector3(
            data.vector.x,
            data.vector.y,
            data.vector.z,
        )
        if data.header.frame_id != "":
            result.reference_frame = world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            )
        return result


@dataclass
class QuaternionToSemDTConverter(
    Ros2ToSemDTConverter[geometry_msgs.QuaternionStamped, Quaternion]
):

    @classmethod
    def convert(cls, data: geometry_msgs.QuaternionStamped, world: World) -> Quaternion:
        result = Quaternion(
            data.quaternion.x,
            data.quaternion.y,
            data.quaternion.z,
            data.quaternion.w,
        )
        if data.header.frame_id != "":
            result.reference_frame = world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            )
        return result


@dataclass
class ColorToSemDTConverter(Ros2ToSemDTConverter[ColorRGBA, Color]):

    @classmethod
    def convert(cls, data: OutputType, world: World) -> InputType:
        return Color(data.r, data.g, data.b, data.a)


@dataclass
class CubeMarkerToSemDTConverter(Ros2ToSemDTConverter[Marker, Box]):

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return super().can_convert(data) and data.type == Marker.CUBE

    @classmethod
    def convert(cls, data: Marker, world: World) -> Box:
        result = Box(
            origin=PoseToSemDTConverter.convert(
                data.pose, world
            ).to_homogeneous_matrix(),
            color=ColorToSemDTConverter.convert(data.color, world),
            scale=Scale(data.scale.x, data.scale.y, data.scale.z),
        )
        result.origin.reference_frame = world.get_kinematic_structure_entity_by_name(
            data.header.frame_id
        )
        return result


@dataclass
class CylinderMarkerToSemDTConverter(Ros2ToSemDTConverter[Marker, Cylinder]):

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return super().can_convert(data) and data.type == Marker.CYLINDER

    @classmethod
    def convert(cls, data: Marker, world: World) -> Cylinder:
        result = Cylinder(
            origin=PoseToSemDTConverter.convert(
                data.pose, world
            ).to_homogeneous_matrix(),
            color=ColorToSemDTConverter.convert(data.color, world),
            width=data.scale.x,
            height=data.scale.z,
        )
        result.origin.reference_frame = world.get_kinematic_structure_entity_by_name(
            data.header.frame_id
        )
        return result


@dataclass
class SphereMarkerToSemDTConverter(Ros2ToSemDTConverter[Marker, Sphere]):

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return super().can_convert(data) and data.type == Marker.SPHERE

    @classmethod
    def convert(cls, data: Marker, world: World) -> Sphere:
        result = Sphere(
            origin=PoseToSemDTConverter.convert(
                data.pose, world
            ).to_homogeneous_matrix(),
            color=ColorToSemDTConverter.convert(data.color, world),
            radius=data.scale.x / 2,
        )
        result.origin.reference_frame = world.get_kinematic_structure_entity_by_name(
            data.header.frame_id
        )
        return result


@dataclass
class MeshMarkerToSemDTConverter(Ros2ToSemDTConverter[Marker, FileMesh]):

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return super().can_convert(data) and data.type == Marker.MESH_RESOURCE

    @classmethod
    def convert(cls, data: Marker, world: World) -> FileMesh:
        result = FileMesh(
            origin=PoseToSemDTConverter.convert(
                data.pose, world
            ).to_homogeneous_matrix(),
            color=ColorToSemDTConverter.convert(data.color, world),
            scale=Scale(data.scale.x, data.scale.y, data.scale.z),
            filename=data.mesh_resource.split("//")[-1],
        )
        result.origin.reference_frame = world.get_kinematic_structure_entity_by_name(
            data.header.frame_id
        )
        return result
