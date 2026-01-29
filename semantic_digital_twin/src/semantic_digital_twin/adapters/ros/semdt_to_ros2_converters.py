from __future__ import annotations

from dataclasses import dataclass

import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
from std_msgs.msg import ColorRGBA
from trimesh.visual import TextureVisuals
from visualization_msgs.msg import Marker

from .msg_converter import SemDTToRos2Converter, InputType
from ...spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    Quaternion,
)
from ...spatial_types.spatial_types import Pose
from ...world_description.geometry import (
    Box,
    Cylinder,
    Sphere,
    Color,
    FileMesh,
    TriangleMesh,
)


@dataclass
class HomogeneousTransformationMatrixToRos2Converter(
    SemDTToRos2Converter[
        HomogeneousTransformationMatrix, geometry_msgs.TransformStamped
    ]
):

    @classmethod
    def convert(
        cls, data: HomogeneousTransformationMatrix
    ) -> geometry_msgs.TransformStamped:
        result = geometry_msgs.TransformStamped()
        if data.reference_frame is not None:
            result.header.frame_id = str(data.reference_frame.name)
        if data.child_frame is not None:
            result.child_frame_id = str(data.child_frame.name)
        position = data.to_position().to_np()
        orientation = data.to_rotation_matrix().to_quaternion().to_np()
        result.transform.translation = geometry_msgs.Vector3(
            x=position[0], y=position[1], z=position[2]
        )
        result.transform.rotation = geometry_msgs.Quaternion(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        )
        return result


@dataclass
class PoseToRos2StampedConverter(SemDTToRos2Converter[Pose, geometry_msgs.PoseStamped]):

    @classmethod
    def convert(cls, data: Pose) -> geometry_msgs.PoseStamped:
        result = geometry_msgs.PoseStamped()
        if data.reference_frame is not None:
            result.header.frame_id = str(data.reference_frame.name)
        position = data.to_position().to_np()
        orientation = data.to_rotation_matrix().to_quaternion().to_np()
        result.pose.position = geometry_msgs.Point(
            x=position[0], y=position[1], z=position[2]
        )
        result.pose.orientation = geometry_msgs.Quaternion(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        )
        return result


@dataclass
class PoseToRos2Converter(SemDTToRos2Converter[Pose, geometry_msgs.Pose]):

    @classmethod
    def convert(cls, data: Pose) -> geometry_msgs.Pose:
        result = geometry_msgs.Pose()
        position = data.to_position().to_np()
        orientation = data.to_rotation_matrix().to_quaternion().to_np()
        result.position = geometry_msgs.Point(
            x=position[0], y=position[1], z=position[2]
        )
        result.orientation = geometry_msgs.Quaternion(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        )
        return result


@dataclass
class Point3ToRos2Converter(SemDTToRos2Converter[Point3, geometry_msgs.PointStamped]):

    @classmethod
    def convert(cls, data: Point3) -> geometry_msgs.PointStamped:
        point_stamped = geometry_msgs.PointStamped()
        if data.reference_frame is not None:
            point_stamped.header.frame_id = str(data.reference_frame.name)
        position = data.evaluate()
        point_stamped.point = geometry_msgs.Point(
            x=position[0], y=position[1], z=position[2]
        )
        return point_stamped


@dataclass
class Vector3ToRos2Converter(
    SemDTToRos2Converter[Vector3, geometry_msgs.Vector3Stamped]
):

    @classmethod
    def convert(cls, data: Vector3) -> geometry_msgs.Vector3Stamped:
        vector_stamped = geometry_msgs.Vector3Stamped()
        if data.reference_frame is not None:
            vector_stamped.header.frame_id = str(data.reference_frame.name)
        vector = data.evaluate()
        vector_stamped.vector = geometry_msgs.Vector3(
            x=vector[0], y=vector[1], z=vector[2]
        )
        return vector_stamped


@dataclass
class QuaternionToRos2Converter(
    SemDTToRos2Converter[Quaternion, geometry_msgs.QuaternionStamped]
):

    @classmethod
    def convert(cls, data: Quaternion) -> geometry_msgs.QuaternionStamped:
        vector_stamped = geometry_msgs.QuaternionStamped()
        if data.reference_frame is not None:
            vector_stamped.header.frame_id = str(data.reference_frame.name)
        vector = data.evaluate()
        vector_stamped.quaternion = geometry_msgs.Quaternion(
            x=vector[0], y=vector[1], z=vector[2], w=vector[3]
        )
        return vector_stamped


@dataclass
class ColorToRos2Converter(SemDTToRos2Converter[Color, ColorRGBA]):

    @classmethod
    def convert(cls, data: Color) -> ColorRGBA:
        return std_msgs.ColorRGBA(r=data.R, g=data.G, b=data.B, a=data.A)


@dataclass
class ShapeToRos2Converter(SemDTToRos2Converter[InputType, Marker]):

    @classmethod
    def convert(cls, data: InputType) -> Marker:
        marker = visualization_msgs.Marker()
        marker.header.frame_id = str(data.origin.reference_frame.name)
        marker.color = ColorToRos2Converter.convert(data.color)
        marker.pose = PoseToRos2Converter.convert(data.origin.to_pose())
        return marker


@dataclass
class BoxToRos2Converter(ShapeToRos2Converter[Box]):

    @classmethod
    def convert(cls, data: Box) -> Marker:
        marker = super().convert(data)
        marker.type = visualization_msgs.Marker.CUBE
        marker.scale.x = data.scale.x
        marker.scale.y = data.scale.y
        marker.scale.z = data.scale.z
        return marker


@dataclass
class CylinderToRos2Converter(ShapeToRos2Converter[Cylinder]):

    @classmethod
    def convert(cls, data: Cylinder) -> Marker:
        marker = super().convert(data)
        marker.type = visualization_msgs.Marker.CYLINDER
        marker.scale.x = data.width
        marker.scale.y = data.width
        marker.scale.z = data.height
        return marker


@dataclass
class SphereToRos2Converter(ShapeToRos2Converter[Sphere]):

    @classmethod
    def convert(cls, data: Sphere) -> Marker:
        marker = super().convert(data)
        marker.type = visualization_msgs.Marker.SPHERE
        marker.scale.x = data.radius * 2
        marker.scale.y = data.radius * 2
        marker.scale.z = data.radius * 2
        return marker


@dataclass
class FileMeshToRos2Converter(ShapeToRos2Converter[FileMesh]):

    @classmethod
    def convert(cls, data: FileMesh) -> Marker:
        marker = super().convert(data)
        marker.type = visualization_msgs.Marker.MESH_RESOURCE
        marker.mesh_resource = "file://" + data.filename
        marker.scale.x = data.scale.x
        marker.scale.y = data.scale.y
        marker.scale.z = data.scale.z
        marker.mesh_use_embedded_materials = True
        if data.mesh.visual.kind == TextureVisuals().kind:
            marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)
        return marker


@dataclass
class TriangleMeshToRos2Converter(ShapeToRos2Converter[TriangleMesh]):

    @classmethod
    def convert(cls, data: TriangleMesh) -> Marker:
        marker = super().convert(data)
        marker.type = visualization_msgs.Marker.MESH_RESOURCE
        marker.mesh_resource = "file://" + data.file.name
        marker.scale.x = data.scale.x
        marker.scale.y = data.scale.y
        marker.scale.z = data.scale.z
        marker.mesh_use_embedded_materials = True
        if data.mesh.visual.kind == TextureVisuals().kind:
            marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)
        return marker
