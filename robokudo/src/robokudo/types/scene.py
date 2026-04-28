"""Scene representation types for Robokudo.

This module provides types for representing and analyzing elements in a scene.
It includes types for:

* Analyzable scene elements
* Object and human hypotheses
* Region definitions
* Part-whole relationships
* Occupancy annotations

The types support 3D point clouds, image ROIs, and nested annotations.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import open3d as o3d
from typing_extensions import TYPE_CHECKING, List, Any

from robokudo.types.core import IdentifiableAnnotation, Nameable, Annotation
from robokudo.types.cv import ImageROI

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AnalyzableAnnotation(IdentifiableAnnotation):
    """
    An Annotation that describes an entity observed in the scene, that shall be analyzed further.
    This is considered as a super class for everything that could potentially be necessary to look at.
    Examples for this are: Detected objects, Detected humans, but also subparts of detected objects.
    The latter includes specific features of an object or subcomponents that need their individual analysis.

    Analyzables can get their own Annotations, might have 3D/6D points and their own ROI.
    """

    annotations: List = field(default_factory=list)
    """
    List of associated annotations
    """

    points: o3d.geometry.PointCloud = None
    """
    Point cloud data
    """

    point_indices: List = field(default_factory=list)
    """
    Indices into point cloud
    """

    object_knowledge: Body | None = None
    """
    A reference to an SDT Body, if present
    """

    roi: ImageROI = field(default_factory=ImageROI)
    """
    Image region of interest
    """


@dataclass
class ObjectHypothesis(AnalyzableAnnotation):
    """Hypothesis about an object in the scene.

    Basic object hypothesis without additional specialization.
    Inherits all capabilities from AnalyzableAnnotation.
    """


@dataclass
class HumanHypothesis(ObjectHypothesis):
    """Hypothesis about a human in the scene.

    In contrast to ObjectHypothesis, we have many different aspects on a Object to annotate.
    Therefore, the semantics of self.points, self.centroid and self.roi on this Hypothesis are a bit different.
    These variables should always contain the maximum region of the person we have been able to detect so far.
    Examples:
      - When running only a face detection, they should be adjusted to the ROI for the face.
      - When running a full body detection + face detection,
        they should be adjusted to everything the full body detection and the face recognition was able to see.
        Basically a Union over the ROIs generated for both.
    In general, PersonHypotheses should rely more on Annotations and also annotate the individual detected features,
    like body parts or activities.
    """

    goal: Any = None
    """
    Goal associated with the human
    """

    result: Any = None
    """
    Result of human-related processing
    """


@dataclass
class RegionHypothesis(AnalyzableAnnotation, Nameable):
    """Hypothesis about a region in the scene.

    A region in the scene. It doesn't necessarily focus on one specific object, but is rather used as a data structure
    to analyze a specific part of the 3d space. Examples: A region above a sofa for free seat estimation,
    the space of a sink to check if it is filled with water, etc.
    """

    ...


@dataclass
class ParthoodHypothesis(AnalyzableAnnotation, Nameable):
    """Base class for part-whole relationships in objects.

    This annotation can be used to represent parts of an object.
    We mainly refer to 'features' and 'components'.
    A logo at a specific place on an object would be a feature, while a control unit on an electrical device is
    a component.

    Parthood Annotations are still a Hypothesis, because due to bad pose estimation we might actually not
    refer to the correct region of the parthood. Example: Pose estimation is off by 180 degrees around the up-axis
    and you are referring to a feature on the front face of an object.
    """


@dataclass
class ParthoodComponentHypothesis(ParthoodHypothesis): ...


@dataclass
class ParthoodFeatureHypothesis(ParthoodHypothesis):
    """Hypothesis about a feature part of an object.

    Represents surface or visual features like:

    * Logos
    * Text
    * Decorative elements
    * Surface patterns
    """

    ...


@dataclass
class OccupiedAnnotation(Annotation):
    """Annotation indicating region occupancy.

    This annotation can be used to indicate that a region is occupied.
    """

    occupied: bool = False
    """
    Indicator for whether the region is occupied or not.
    """
