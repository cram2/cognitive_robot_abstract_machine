from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Type

import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
from geometry_msgs.msg import (
    TransformStamped,
    PointStamped,
    Vector3Stamped,
    PoseStamped,
    QuaternionStamped,
)
from std_msgs.msg import ColorRGBA
from typing_extensions import Generic, TypeVar, ClassVar
from visualization_msgs.msg import Marker

from krrood.utils import recursive_subclasses, DataclassException
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Shape,
    Box,
    Cylinder,
    Sphere,
    Color,
    FileMesh,
)

OurType = TypeVar("OurType")
Ros2Type = TypeVar("Ros2Type")


@dataclass
class ROS2ConversionError(DataclassException): ...


@dataclass
class CannotConvertToRos2Error(ROS2ConversionError): ...


@dataclass
class CannotConvertFromRos2Error(ROS2ConversionError): ...


@dataclass
class ROS2MessageConverter(ABC, Generic[OurType, Ros2Type]):
    our_registry: ClassVar[Dict[Type, Type[ROS2MessageConverter]]] = field(default={})
    ros2_registry: ClassVar[Dict[Type, Type[ROS2MessageConverter]]] = field(default={})

    our_type: Type
    ros2_type: Type

    def __post_init__(self):
        self.our_registry[self.__class__] = self.__class__

    @classmethod
    def get_to_ros2_converter(cls, our_type: Type) -> Type[ROS2MessageConverter]:
        for sub_class in recursive_subclasses(cls):
            if sub_class.our_type == our_type:
                return sub_class

        raise CannotConvertToRos2Error()

    @classmethod
    def get_to_our_converter(cls, ros2_type: Type) -> Type[ROS2MessageConverter]:
        for sub_class in recursive_subclasses(cls):
            if sub_class.ros2_type == ros2_type:
                return sub_class
        raise CannotConvertFromRos2Error()

    @classmethod
    def to_ros2_message(cls, data: OurType) -> Ros2Type:
        return ROS2MessageConverter.get_to_ros2_converter(type(data))._to_ros2_message(
            data
        )

    @classmethod
    @abstractmethod
    def _to_ros2_message(cls, data: OurType) -> Ros2Type:
        pass

    @classmethod
    def from_ros2_message(cls, data: Ros2Type) -> OurType:
        return ROS2MessageConverter.get_to_our_converter(type(data))._from_ros2_message(
            data
        )

    @classmethod
    @abstractmethod
    def _from_ros2_message(cls, data: Ros2Type, world: World) -> OurType:
        pass


@dataclass
class HomogeneousTransformationMatrixROS2Converter(
    ROS2MessageConverter[HomogeneousTransformationMatrix, TransformStamped]
):
    our_type = HomogeneousTransformationMatrix
    ros2_type = TransformStamped

    @classmethod
    def _to_ros2_message(
        cls, data: HomogeneousTransformationMatrix
    ) -> TransformStamped:
        result = TransformStamped()
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

    @classmethod
    def _from_ros2_message(
        cls, data: TransformStamped, world: World
    ) -> HomogeneousTransformationMatrix:
        result = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=data.transform.translation.x,
            pos_y=data.transform.translation.y,
            pos_z=data.transform.translation.z,
            quat_x=data.transform.rotation.x,
            quat_y=data.transform.rotation.y,
            quat_z=data.transform.rotation.z,
            quat_w=data.transform.rotation.w,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            ),
            child_frame=world.get_kinematic_structure_entity_by_name(
                data.child_frame_id
            ),
        )
        return result


@dataclass
class PoseROS2Converter(ROS2MessageConverter[Pose, PoseStamped]):
    our_type = Pose
    ros2_type = PoseStamped

    @classmethod
    def _to_ros2_message(cls, data: Pose) -> PoseStamped:
        result = PoseStamped()
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

    @classmethod
    def _from_ros2_message(cls, data: PoseStamped, world: World) -> Pose:
        result = Pose.from_xyz_quaternion(
            pos_x=data.pose.position.x,
            pos_y=data.pose.position.y,
            pos_z=data.pose.position.z,
            quat_x=data.pose.orientation.x,
            quat_y=data.pose.orientation.y,
            quat_z=data.pose.orientation.z,
            quat_w=data.pose.orientation.w,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            ),
        )
        return result


@dataclass
class Point3ROS2Converter(ROS2MessageConverter[Point3, PointStamped]):
    our_type = Point3
    ros2_type = PointStamped

    @classmethod
    def _to_ros2_message(cls, data: Point3) -> PointStamped:
        point_stamped = PointStamped()
        if data.reference_frame is not None:
            point_stamped.header.frame_id = str(data.reference_frame.name)
        position = data.evaluate()
        point_stamped.point = geometry_msgs.Point(
            x=position[0], y=position[1], z=position[2]
        )
        return point_stamped

    @classmethod
    def _from_ros2_message(cls, data: Ros2Type, world: World) -> OurType:
        return Point3(
            data.point.x,
            data.point.y,
            data.point.z,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            ),
        )


@dataclass
class Vector3ROS2Converter(ROS2MessageConverter[Vector3, Vector3Stamped]):
    our_type = Vector3
    ros2_type = Vector3Stamped

    @classmethod
    def _to_ros2_message(cls, data: Vector3) -> Vector3Stamped:
        vector_stamped = Vector3Stamped()
        if data.reference_frame is not None:
            vector_stamped.header.frame_id = str(data.reference_frame.name)
        vector = data.evaluate()
        vector_stamped.vector = geometry_msgs.Point(
            x=vector[0], y=vector[1], z=vector[2]
        )
        return vector_stamped

    @classmethod
    def _from_ros2_message(cls, data: Ros2Type, world: World) -> Vector3:
        return Vector3(
            data.point.x,
            data.point.y,
            data.point.z,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            ),
        )


@dataclass
class QuaternionROS2Converter(ROS2MessageConverter[Quaternion, QuaternionStamped]):
    our_type = Quaternion
    ros2_type = QuaternionStamped

    @classmethod
    def _to_ros2_message(cls, data: Quaternion) -> QuaternionStamped:
        vector_stamped = QuaternionStamped()
        if data.reference_frame is not None:
            vector_stamped.header.frame_id = str(data.reference_frame.name)
        vector = data.evaluate()
        vector_stamped.quaternion = geometry_msgs.Quaternion(
            x=vector[0], y=vector[1], z=vector[2]
        )
        return vector_stamped

    @classmethod
    def _from_ros2_message(cls, data: QuaternionStamped, world: World) -> Quaternion:
        return Quaternion(
            data.quaternion.x,
            data.quaternion.y,
            data.quaternion.z,
            data.quaternion.w,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                data.header.frame_id
            ),
        )


@dataclass
class ColorROS2Converter(ROS2MessageConverter[Color, ColorRGBA]):
    our_type = Color
    ros2_type = ColorRGBA

    @classmethod
    def _to_ros2_message(cls, data: Color) -> ColorRGBA:
        return std_msgs.ColorRGBA(r=data.R, g=data.G, b=data.B, a=data.A)

    @classmethod
    def _from_ros2_message(cls, data: Ros2Type, world: World) -> OurType:
        return Color(data.r, data.g, data.b, data.a)


@dataclass
class ShapeROS2Converter(ROS2MessageConverter[OurType, Marker]):
    our_type = OurType
    ros2_type = Marker

    @classmethod
    def to_ros2_message(cls, data: Shape) -> Marker:
        return super().to_ros2_message(data)

    @classmethod
    def _to_ros2_message(cls, data: OurType) -> Ros2Type:
        marker = visualization_msgs.Marker()
        marker.header.frame_id = str(data.origin.reference_frame.name)
        marker.color = ColorROS2Converter.to_ros2_message(data.color)
        marker.pose = PoseROS2Converter.to_ros2_message(data.origin.to_pose()).pose
        return marker

    @classmethod
    def _from_ros2_message(cls, data: Ros2Type, world: World) -> OurType:
        raise ROS2ConversionError("cannot convert marker to shape")


@dataclass
class BoxShapeROS2Converter(ShapeROS2Converter[Box]):
    our_type = Box
    ros2_type = Marker

    @classmethod
    def _to_ros2_message(cls, data: Box) -> Marker:
        marker = super()._to_ros2_message(data)
        marker.type = visualization_msgs.Marker.CUBE
        marker.scale.x = data.scale.x
        marker.scale.y = data.scale.y
        marker.scale.z = data.scale.z
        return marker


@dataclass
class CylinderShapeROS2Converter(ShapeROS2Converter[Cylinder]):
    our_type = Cylinder
    ros2_type = Marker

    @classmethod
    def _to_ros2_message(cls, data: Cylinder) -> Marker:
        marker = super()._to_ros2_message(data)
        marker.type = visualization_msgs.Marker.CYLINDER
        marker.scale.x = data.width
        marker.scale.y = data.width
        marker.scale.z = data.height
        return marker


@dataclass
class SphereShapeROS2Converter(ShapeROS2Converter[Sphere]):
    our_type = Sphere
    ros2_type = Marker

    @classmethod
    def _to_ros2_message(cls, data: Sphere) -> Marker:
        marker = super()._to_ros2_message(data)
        marker.type = visualization_msgs.Marker.SPHERE
        marker.scale.x = data.radius * 2
        marker.scale.y = data.radius * 2
        marker.scale.z = data.radius * 2
        return marker


@dataclass
class FileMeshShapeROS2Converter(ShapeROS2Converter[FileMesh]):
    our_type = FileMesh
    ros2_type = Marker

    @classmethod
    def _to_ros2_message(cls, data: FileMesh) -> Marker:
        marker = super()._to_ros2_message(data)
        marker.type = visualization_msgs.Marker.MESH_RESOURCE
        marker.mesh_resource = "file://" + data.filename
        marker.scale.x = data.scale.x
        marker.scale.y = data.scale.y
        marker.scale.z = data.scale.z
        marker.mesh_use_embedded_materials = True
        marker.color = ColorRGBA()
        return marker
