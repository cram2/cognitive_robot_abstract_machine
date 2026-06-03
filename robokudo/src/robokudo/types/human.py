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

from typing_extensions import Any, List

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
class KeypointAnnotation(Annotation):
    """Human body keypoint detection result.

    Represents detected keypoints like:
    * Joint positions
    * Body part locations
    * Facial landmarks

    Supports both 2D and 3D keypoint types.
    """

    KP_TYPE_2D: str = "2D"
    """
    Identifier for 2D keypoint sets
    """

    KP_TYPE_3D: str = "3D"
    """
    Identifier for 3D keypoint sets
    """

    keypoints: List = field(default_factory=list)
    """
    List of detected keypoints
    """

    type: str = None  # type of keypoints: e.g. 3D or 2D
    """
    Type of keypoints (2D or 3D)
    """
