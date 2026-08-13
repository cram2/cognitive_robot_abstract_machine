"""Annotation types for robokudo.

This module provides various annotation types used for object classification,
semantic information, location data, geometric shapes, and pose information.
All annotation types inherit from the base Annotation class.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Optional, Any, List

from robokudo.types.core import Annotation
from robokudo.types.cv import BoundingBox3D, Points3D
from robokudo.types.tf import (
    Position,
    Pose,
    StampedPose,
    StampedPosition,
    StampedTransform,
)
from semantic_digital_twin.world_description.geometry import (
    Shape as SemDTShape,
    Box as SemDTBox,
    Sphere as SemDTSphere,
    Cylinder as SemDTCylinder,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from semantic_digital_twin.world_description.world_entity import Region


@dataclass
class Classification(Annotation):
    """Classification annotation for objects.

    This class represents a classification result, including the type of classification
    (instance, class, or shape), the class name, confidence score, and class ID.
    """

    classification_type: str = ""
    """
    Type of classification (INSTANCE, CLASS, or SHAPE)
    """

    classname: str = ""
    """
    Name of the classified class
    """

    confidence: float = 0.0
    """
    Confidence score of the classification
    """

    class_id: int = 0
    """
    Numeric identifier for the class
    """


@dataclass
class SemanticColor(Annotation):
    """Semantic color annotation.

    This class represents semantic color information, including the color name
    and its ratio/proportion in the annotated object or region.
    """

    color: str = ""
    """
    Name of the color
    """

    ratio: float = 0.0
    """
    Proportion/ratio of this color in the object/region
    """


@dataclass
class LocationAnnotation(Annotation):
    """Location annotation for objects.

    This class represents a semantic location in the environment.
    """

    region: Region | None = None
    """
    Referenced semantic region
    """

    name: str = ""
    """
    Legacy fallback location name
    """


@dataclass
class Plane(Annotation):
    """Plane annotation for surface detection.

    This class represents a detected plane in 3D space, including its model
    parameters and inlier points.
    """

    model: List[float] = field(default_factory=list)
    """
    4-dimensional plane equation parameters [a, b, c, d] for ax + by + cz + d = 0
    """

    inliers: List[int] = field(default_factory=list)
    """
    List of pointcloud indices that belong to this plane
    """


@dataclass
class Shape(Annotation):
    """Base class for shape annotations.

    This class stores a SemDT geometry shape and inlier point indices.
    """

    inliers: List[int] = field(default_factory=list)
    """
    List of pointcloud indices that belong to this shape
    """

    geometry: SemDTShape = field(default_factory=SemDTBox)
    """
    SemDT geometry shape represented by this annotation
    """

    @property
    def shape_name(self) -> str:
        """Return the semantic name of the stored shape."""
        return self.geometry.__class__.__name__


@dataclass
class Cuboid(Shape):
    """Cuboid shape annotation.

    This class represents a cuboid shape as a SemDT box geometry.
    """

    geometry: SemDTBox = field(default_factory=SemDTBox)
    """
    SemDT box geometry represented by this annotation
    """


@dataclass
class Sphere(Shape):
    """Sphere shape annotation.

    This class represents a sphere shape as a SemDT sphere geometry.
    """

    geometry: SemDTSphere = field(default_factory=SemDTSphere)
    """
    SemDT sphere geometry represented by this annotation
    """


@dataclass
class Cylinder(Shape):
    """Cylinder shape annotation.

    This class represents a cylinder shape as a SemDT cylinder geometry.
    """

    geometry: SemDTCylinder = field(default_factory=SemDTCylinder)
    """
    SemDT cylinder geometry represented by this annotation
    """


@dataclass
class ColorHistogram(Annotation):
    """Color histogram annotation.

    This class usually represents a 2D color histogram, typically containing hue and
    saturation information.
    """

    hist: Optional[npt.NDArray] = None
    """
    2D histogram array
    """

    normalized: bool = False
    """
    Whether the histogram is normalized
    """


@dataclass
class PoseAnnotation(Pose, Annotation):
    """Pose annotation combining transform and annotation functionality.

    This class inherits from both Pose and Annotation to provide pose information
    as an annotation type.
    """

    ...


@dataclass
class PositionAnnotation(Position, Annotation):
    """Position annotation combining position and annotation functionality.

    This class inherits from both Position and Annotation to provide position
    information as an annotation type.
    """

    ...


@dataclass
class StampedPoseAnnotation(StampedPose, Annotation):
    """Timestamped pose annotation.

    This class combines timestamped pose information with annotation functionality.
    """

    ...


@dataclass
class StampedPositionAnnotation(StampedPosition, Annotation):
    """Timestamped position annotation.

    This class combines timestamped position information with annotation functionality.
    """

    ...


@dataclass
class StampedTransformAnnotation(StampedTransform, Annotation):
    """Timestamped transform annotation.

    This class combines timestamped transform information with annotation functionality.
    """

    ...


@dataclass
class Encoding(Annotation):
    """An abstract Encoding Type.

    This class represents various types of encodings such as feature vectors,
    latent space representations, or other variables.
    """

    encoding: Any = None
    """
    The encoded representation
    """


@dataclass
class BoundingBox3DAnnotation(BoundingBox3D, Annotation):
    """3D bounding box annotation.

    This class combines 3D bounding box functionality with annotation capabilities.
    """

    ...


@dataclass
class CloudAnnotation(Points3D, Annotation):
    """Point cloud annotation.

    This class combines 3D point cloud functionality with annotation capabilities.
    """

    ...


@dataclass
class SpatiallyNearestAnnotation(Annotation):
    """
    Annotation to describe the spatially nearest object to the camera.
    Supposed to be unique for an amount of objects or humans.
    """

    ...


@dataclass
class TextAnnotation(Annotation):
    """Text annotation."""

    text: str = ""
    """
    Text content of the annotation
    """
