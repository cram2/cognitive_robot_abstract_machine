"""Human-related annotation types for Robokudo.

This module provides types for representing various aspects of human detection
and analysis in computer vision applications. It includes:

* Face detection results
* Facial expressions and mimics
* Human attributes
* Activity recognition
* Body keypoint detection

The types support both 2D and 3D representations where applicable.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto

import open3d as o3d
from typing_extensions import Any

from robokudo.types.core import Annotation
from robokudo.types.cv import Points3D, ImageROI
from robokudo.types.tf import Pose


@dataclass
class FaceAnnotation(ImageROI, Points3D, Pose):
    """Face detection result combining 2D, 3D and pose information.

    Provides:
    * 2D region of interest in image
    * 3D point cloud of face
    * 6-DOF pose of face

    .. note::
       Does not contain identity information. Use Classification annotations
       on the parent HumanAnnotation for identity.
    """

    ...


@dataclass
class MimicAnnotation(Annotation):
    """Facial expression or mimic detection result.

    Represents detected facial expressions or mimics.
    """

    type: str = str("")
    """
    Type of expression/mimic detected
    """


@dataclass
class AttributeAnnotation(Annotation):
    """Generic human attribute annotation.

    Used for various human attributes such as:
    * Age
    * Height
    * Clothing
    * Gender
    * Other physical or visual characteristics
    """

    type: str = str("")
    """
    Type of attribute
    """


@dataclass
class ActivityAnnotation(Annotation):
    """Human activity detection result.

    Represents detected human activities and their context.
    Can be subclassed for specific activity types.

    Examples:
    * Pointing
    * Waving
    * Walking
    * Sitting
    """

    type: str = str("")
    """
    Type of activity detected
    """

    interaction_with: Any = None
    """
    Target of interaction if applicable
    """


@dataclass
class BodyKeypoint:
    """
    The position of one body part, such as the nose or the left knee, in an image.
    """

    x: float
    """
    Horizontal position in color image pixels.
    """

    y: float
    """
    Vertical position in color image pixels.
    """

    confidence: float
    """
    Confidence of the detector that the body part is at this position.
    """


@dataclass
class BodyKeypointCloud:
    """
    The depth measurements around a body keypoint.
    """

    keypoint: BodyKeypoint
    """
    The body keypoint the cloud is built around.
    """

    points: o3d.geometry.PointCloud
    """
    The colored points measured around the keypoint, empty if none were built for it.
    """


class KeypointKind(StrEnum):
    """
    What each keypoint of a :class:`KeypointAnnotation` is given as.
    """

    IMAGE_POSITION = auto()
    """
    Each keypoint is a :class:`BodyKeypoint`, its position in the color image.
    """

    POINT_CLOUD = auto()
    """
    Each keypoint is a :class:`BodyKeypointCloud`, the depth measurements around it.
    """


@dataclass
class KeypointAnnotation(Annotation):
    """
    The body keypoints of one human, indexed by
    :class:`~robokudo.annotators.human_and_pose.PoseKeypoint`.
    """

    keypoints: list[BodyKeypoint] | list[BodyKeypointCloud] = field(
        default_factory=list
    )
    """
    The keypoints, given as :attr:`kind` says.
    """

    kind: KeypointKind = field(kw_only=True)
    """
    What each of the :attr:`keypoints` is given as.
    """
