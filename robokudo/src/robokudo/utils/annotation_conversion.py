"""
This module provides conversion methods from Annotations to other types, such as ROS Message types.
It's mainly used to in the result generation to fill the query result from the available annotations,
hence it requires a target ObjectDesignator to act on.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

from geometry_msgs.msg import Vector3, PoseStamped
from typing_extensions import TYPE_CHECKING, Type

from robokudo_msgs.msg import ShapeSize
from . import annotator_helper
from .. import defs
from ..cas import CASViews
from ..types.annotation import (
    PoseAnnotation,
    StampedPoseAnnotation,
    Classification,
    PositionAnnotation,
    SemanticColor,
    LocationAnnotation,
    Sphere,
    Cuboid,
    Shape,
    StampedPositionAnnotation,
)
from ..types.cv import BoundingBox3D

if TYPE_CHECKING:
    from robokudo_msgs.msg import ObjectDesignator
    from ..types.core import Annotation
    from ..cas import CAS


class Annotation2AnnotationConverter(ABC):
    """An abstract converter class for converting annotations to other annotations."""

    @abstractmethod
    def can_convert(self, annotation: Annotation, target_annotation_type: type) -> bool:
        """Checks whether the converter can convert the given annotation to the given target annotation type.

        :param annotation: The annotation to check for conversion.
        :param target_annotation_type: The target annotation type to convert `annotation` to.
        :returns: Whether the converter can convert `annotation` to `target_annotation_type`.
        """
        pass

    @abstractmethod
    def convert(self, annotation: Annotation, cas: CAS = None) -> Annotation:
        """Converts the given annotation to the given target annotation type.

        Use `self.can_convert` to check whether the converter is able to convert the given annotation to the desired type.

        :param annotation: The annotation to convert.
        :param cas: The CAS to use for conversion.
        :returns: The converted annotation.
        """
        pass


class Annotation2ODConverter(ABC):
    """An abstract converter class for converting annotations to object designators."""

    @abstractmethod
    def can_convert(self, annotation: Annotation) -> bool:
        """Checks whether the converter can convert the given annotation to an object designator.

        :param annotation: The annotation to check for conversion.
        :returns: Whether the converter can convert `annotation` to `ObjectDesignator`.
        """
        pass

    @abstractmethod
    def convert(
            self, annotation: Annotation, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        """Converts the data of the given annotation to data in the given object designator. Modifies the object designator in-place.

        Use `self.can_convert` to check whether the converter is able to convert the given annotation to an object designator.

        :param annotation: The annotation to convert.
        :param cas: The CAS to use for conversion.
        :param object_designator: The object designator to fill with the data of `annotation`
        """
        pass


########################################
# Annotation 2 Annotation Converters
########################################


class PoseAnnotationToStampedPoseAnnotationConverter(Annotation2AnnotationConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `PoseAnnotation` to a `StampedPoseAnnotation`."""

    def can_convert(self, annotation: Annotation, target_annotation_type: Type) -> bool:
        return (
                isinstance(annotation, PoseAnnotation)
                and target_annotation_type == StampedPoseAnnotation
        )

    def convert(
            self, annotation: PoseAnnotation, cas: CAS = None
    ) -> StampedPoseAnnotation:
        spa = StampedPoseAnnotation()
        spa.source = annotation.source
        spa.translation = annotation.translation
        spa.rotation = annotation.rotation
        spa.source = annotation.source
        return spa


class PositionAnnotationToStampedPoseAnnotationConverter(
    Annotation2AnnotationConverter
):
    """Extended `Annotation2AnnotationConverter` that converts a `Positionnnotation` to a `StampedPoseAnnotation`."""

    def can_convert(self, annotation: Annotation, target_annotation_type: Type) -> bool:
        return (
                isinstance(annotation, PositionAnnotation)
                and target_annotation_type == StampedPoseAnnotation
        )

    def convert(
            self, annotation: PositionAnnotation, cas: CAS = None
    ) -> StampedPoseAnnotation:
        spa = StampedPoseAnnotation()
        spa.source = annotation.source
        spa.translation = annotation.translation
        spa.rotation = [0, 0, 0, 1]
        return spa


########################################
# Annotation 2 ObjectDesignator Converters
########################################


class SemanticColor2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `SemanticColor` annotation to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, SemanticColor)

    def convert(
            self, annotation: SemanticColor, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        object_designator.color.append(annotation.color)


class Classification2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `Classification` annotation to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Classification)

    def convert(
            self, annotation: Classification, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        object_designator.type = annotation.classname


# class PoseBase2ODConverter(Annotation2ODConverter):
#    def transform_cam_pose_to_world_pose(self, annotation: Annotation, cas: CAS) -> PoseAnnotation:
#        return transform_pose_from_cam_to_world(cas, annotation)
#
#    def pose_stamped_from_pose_annotation(self, pose_annotation: PoseAnnotation) -> PoseStamped:
#        """
#        Convert a RoboKudo Pose Annotation to a ROS-style PoseStamped.
#        Be warned though, that we don't know anything about the header/frame, because the normal annotation
#        doesn't carry this information.
#
#        :param pose_annotation:
#        :return: geometry_msgs.msg.PoseStamped with only the position and rotation information from pose_annotation
#        """
#
#        ps = PoseStamped()
#        ps.pose.position.x = pose_annotation.translation[0]
#        ps.pose.position.y = pose_annotation.translation[1]
#        ps.pose.position.z = pose_annotation.translation[2]
#
#        ps.pose.orientation.x = pose_annotation.rotation[0]
#        ps.pose.orientation.y = pose_annotation.rotation[1]
#        ps.pose.orientation.z = pose_annotation.rotation[2]
#        ps.pose.orientation.w = pose_annotation.rotation[3]
#        return ps


class StampedPose2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `StampedPoseAnnotation` to `ObjectDesignator` data."""

    def can_convert(self, annotation) -> bool:
        return isinstance(annotation, StampedPoseAnnotation)

    def convert(
            self,
            annotation: StampedPoseAnnotation,
            cas: CAS,
            object_designator: ObjectDesignator,
    ) -> None:
        ps = PoseStamped()
        ps.pose.position.x = annotation.translation[0]
        ps.pose.position.y = annotation.translation[1]
        ps.pose.position.z = annotation.translation[2]

        ps.pose.orientation.x = annotation.rotation[0]
        ps.pose.orientation.y = annotation.rotation[1]
        ps.pose.orientation.z = annotation.rotation[2]
        ps.pose.orientation.w = annotation.rotation[3]

        ps.header.frame_id = annotation.frame
        ps.header.stamp.sec = annotation.timestamp

        object_designator.pose.append(ps)
        object_designator.pose_source.append(annotation.source)


class Pose2ODConverter(StampedPose2ODConverter):
    """Extended `StampedPose2ODConverter` that converts a `PoseAnnotation` to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, PoseAnnotation)

    def convert(
            self, annotation: PoseAnnotation, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        """Converts the data of the given annotation to data in the given object designator. Modifies the object designator in-place.

        Use `self.can_convert` to check whether the converter is able to convert the given annotation to an object designator.
        Conversion requires a valid cam to world transform in the given CAS.

        :param annotation: The annotation to convert.
        :param cas: The CAS to use for conversion.
        :param object_designator: The object designator to fill with the data of `annotation`
        """
        use_cam_coords = cas.get(CASViews.VIEWPOINT_CAM_TO_WORLD) is None
        if use_cam_coords:
            pose_annotation = annotation
        else:
            pose_annotation = annotator_helper.transform_pose_from_cam_to_world(
                cas, annotation
            )

        # First, convert the PoseAnnotation to a StampedPose
        converter = PoseAnnotationToStampedPoseAnnotationConverter()
        spa = converter.convert(pose_annotation)
        # Fill the missing header information
        spa.timestamp = cas.get(CASViews.CAM_INFO).header.stamp.sec

        # For the frame, check if we should return cam or world coordinates based on the previous check
        if use_cam_coords:
            spa.frame = cas.get(CASViews.CAM_INFO).header.frame_id
        else:
            spa.frame = "map"

        # Second, do the actual conversion and place info in ObjectDesignator
        # self.stamped_pose_converter.convert(spa, cas, object_designator)
        super().convert(spa, cas, object_designator)


class Position2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `PositionAnnotation` to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, PositionAnnotation)

    def convert(
            self,
            annotation: PositionAnnotation,
            cas: CAS,
            object_designator: ObjectDesignator,
    ) -> None:
        """Converts the data of the given annotation to data in the given object designator. Modifies the object designator in-place.

        Use `self.can_convert` to check whether the converter is able to convert the given annotation to an object designator.
        Conversion requires a valid cam to world transform in the given CAS.

        :param annotation: The annotation to convert.
        :param cas: The CAS to use for conversion.
        :param object_designator: The object designator to fill with the data of `annotation`
        """
        pos = PoseAnnotation()
        pos.source = annotation.source
        pos.translation.insert(0, annotation.translation[0])
        pos.translation.insert(1, annotation.translation[1])
        pos.translation.insert(2, annotation.translation[2])

        pos.rotation.insert(0, 0)
        pos.rotation.insert(1, 0)
        pos.rotation.insert(2, 0)
        pos.rotation.insert(3, 1)

        pose_map = annotator_helper.transform_pose_from_cam_to_world(cas, pos)

        # Must be a PoseStamped due to type specification in ros message
        ps = PoseStamped()
        ps.pose.position.x = pose_map.translation[0]
        ps.pose.position.y = pose_map.translation[1]
        ps.pose.position.z = pose_map.translation[2]

        ps.pose.orientation.x = pose_map.rotation[0]
        ps.pose.orientation.y = pose_map.rotation[1]
        ps.pose.orientation.z = pose_map.rotation[2]
        ps.pose.orientation.w = pose_map.rotation[3]

        # We assume that the pose annotation is in CAMERA coordinates
        ps.header = copy.deepcopy(cas.get(CASViews.CAM_INFO).header)
        ps.header.frame_id = "map"
        object_designator.pose.append(ps)

        object_designator.pose_source.append(annotation.source)


class StampedPosition2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `StampedPositionAnnotation` to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, StampedPositionAnnotation)

    def convert(
            self,
            annotation: StampedPositionAnnotation,
            cas: CAS,
            object_designator: ObjectDesignator,
    ) -> None:
        """Converts the data of the given annotation to data in the given object designator. Modifies the object designator in-place.

        Use `self.can_convert` to check whether the converter is able to convert the given annotation to an object designator.
        Conversion requires a valid cam to world transform in the given CAS.

        :param annotation: The annotation to convert.
        :param cas: The CAS to use for conversion.
        :param object_designator: The object designator to fill with the data of `annotation`
        """
        # Create pose from position
        pos = PoseAnnotation()
        pos.source = annotation.source
        pos.translation.insert(0, annotation.translation[0])
        pos.translation.insert(1, annotation.translation[1])
        pos.translation.insert(2, annotation.translation[2])

        pos.rotation.insert(0, 0)
        pos.rotation.insert(1, 0)
        pos.rotation.insert(2, 0)
        pos.rotation.insert(3, 1)

        pose_map = annotator_helper.transform_pose_from_cam_to_world(cas, pos)

        # Must be a PoseStamped due to type specification in ros message
        ps = PoseStamped()
        ps.pose.position.x = pose_map.translation[0]
        ps.pose.position.y = pose_map.translation[1]
        ps.pose.position.z = pose_map.translation[2]

        ps.pose.orientation.x = pose_map.rotation[0]
        ps.pose.orientation.y = pose_map.rotation[1]
        ps.pose.orientation.z = pose_map.rotation[2]
        ps.pose.orientation.w = pose_map.rotation[3]

        # Keep stamp data
        ps.header.frame_id = annotation.frame
        ps.header.stamp.sec = annotation.timestamp

        object_designator.pose.append(ps)
        object_designator.pose_source.append(annotation.source)


class BoundingBox3DForShapeSizeConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `robokudo.types.cv.BoundingBox3D` to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, BoundingBox3D)

    def convert(
            self, annotation: BoundingBox3D, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        # TODO A (new!) Annotator should infer semantic size.
        #  Find a good decision criteria for size. This should be a semantic label! (small, large etc.)
        object_designator.size = f""
        vector = Vector3()
        vector.x = float(annotation.x_length)
        vector.y = float(annotation.y_length)
        vector.z = float(annotation.z_length)
        size = ShapeSize()
        size.dimensions = vector
        size.radius = float(0.0)
        object_designator.shape_size.append(size)


class Shape2ODConverter(Annotation2ODConverter):
    """Extended `Annotation2AnnotationConverter` that converts a `Shape` annotation to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Shape)

    def convert(
            self, annotation: Shape, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        object_designator.shape.append(annotation.type)


class Cuboid2ODConverter(Shape2ODConverter):
    """Extended `Shape2ODConverter` that converts a `Cuboid` annotation to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Cuboid)

    def convert(
            self, annotation: Cuboid, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        super().convert(annotation, cas, object_designator)


class Sphere2ODConverter(Shape2ODConverter):
    """Extended `Shape2ODConverter` that converts a `Sphere` annotation to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Sphere)

    def convert(
            self, annotation: Sphere, cas: CAS, object_designator: ObjectDesignator
    ) -> None:
        size = ShapeSize(radius=annotation.radius)

        rk_logger = logging.getLogger(defs.PACKAGE_NAME)
        rk_logger.info(
            f"The center point of the Sphere annotation is currently not converted to OD."
        )

        object_designator.shape_size.append(size)
        super().convert(annotation, cas, object_designator)


class Location2ODConverter(Annotation2ODConverter):
    """Extended `Shape2ODConverter` that converts a `LocationAnnotation` to `ObjectDesignator` data."""

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, LocationAnnotation)

    def convert(
            self,
            annotation: LocationAnnotation,
            cas: CAS,
            object_designator: ObjectDesignator,
    ) -> None:
        object_designator.location = annotation.name
