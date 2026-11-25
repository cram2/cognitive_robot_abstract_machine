"""
Types for the elements of a scene that RoboKudo analyzes: object, human, region and part
hypotheses and occupancy annotations, each with its point cloud, image region and nested
annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import open3d as o3d
from semantic_digital_twin.reasoning.queries import annotation_class_by_label
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Human,
    UnknownObject,
)
from typing_extensions import TYPE_CHECKING, Any, TypeVar

from robokudo.types.annotation import Classification
from robokudo.types.core import Annotation, IdentifiableAnnotation, Nameable
from robokudo.types.cv import ImageROI

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body

AnnotationType = TypeVar("AnnotationType", bound=Annotation)


@dataclass(kw_only=True)
class AnalyzableAnnotation(IdentifiableAnnotation):
    """
    An Annotation that describes an entity observed in the scene, that shall be analyzed further.
    This is considered as a super class for everything that could potentially be necessary to look at.
    Examples for this are: Detected objects, Detected humans, but also subparts of detected objects.
    The latter includes specific features of an object or subcomponents that need their individual analysis.

    Analyzables can get their own Annotations, might have 3D/6D points and their own ROI.
    """

    annotations: list = field(default_factory=list)
    """
    List of associated annotations.
    """

    points: o3d.geometry.PointCloud = None
    """
    Point cloud data.
    """

    point_indices: list = field(default_factory=list)
    """
    Indices into point cloud.
    """

    object_knowledge: Body | None = None
    """
    A reference to an SDT Body, if present.
    """

    roi: ImageROI = field(default_factory=ImageROI)
    """
    Image region of interest.
    """

    def latest_annotation_of_type(
        self, *, annotation_type: type[AnnotationType]
    ) -> AnnotationType | None:
        """
        :param annotation_type: The type of the annotation.
        :return: The annotation of ``annotation_type`` added last to
            :attr:`annotations`, or None if there is none.
        """
        for annotation in reversed(self.annotations):
            if isinstance(annotation, annotation_type):
                return annotation
        return None


@dataclass(kw_only=True)
class ObjectHypothesis(AnalyzableAnnotation):
    """
    Hypothesis about an object in the scene.

    Basic object hypothesis without additional specialization. Inherits all capabilities
    from AnalyzableAnnotation.
    """

    @property
    def class_label(self) -> str | None:
        """
        The class name of the latest classification of this hypothesis, or None if it
        was not classified.
        """
        classification = self.latest_annotation_of_type(annotation_type=Classification)
        if classification is None or not classification.classname:
            return None
        return classification.classname

    def determine_semantic_class(
        self, *, minimum_class_confidence: float
    ) -> type[IsPerceivable]:
        """
        :param minimum_class_confidence: How confident a classification has to be.
        :return: The semantic annotation :attr:`class_label` names, if the latest
            classification is at least ``minimum_class_confidence`` confident; else
            :class:`UnknownObject`.
        """
        classification = self.latest_annotation_of_type(annotation_type=Classification)
        if (
            self.class_label is None
            or classification.confidence < minimum_class_confidence
        ):
            return UnknownObject
        semantic_class = annotation_class_by_label(self.class_label)
        if semantic_class is None:
            return UnknownObject
        return semantic_class


@dataclass(kw_only=True)
class HumanHypothesis(ObjectHypothesis):
    """
    Hypothesis about a human in the scene.

    Its :attr:`points` and :attr:`roi` cover the largest region of the person detected
    so far, such as the union of a face and a full body detection. Detected features
    such as body parts or activities are annotated individually.
    """

    goal: Any = None
    """
    Goal associated with the human.
    """

    result: Any = None
    """
    Result of human-related processing.
    """

    def determine_semantic_class(
        self, *, minimum_class_confidence: float
    ) -> type[IsPerceivable]:
        """
        Unlike :meth:`ObjectHypothesis.determine_semantic_class`, :class:`Human`
        whatever the classification says.
        """
        return Human


@dataclass(kw_only=True)
class RegionHypothesis(AnalyzableAnnotation, Nameable):
    """
    Hypothesis about a region in the scene.

    A part of the 3D space to analyze that is not necessarily one object, such as the
    region above a sofa for free seat estimation or a sink checked for water.
    """

    ...


@dataclass(kw_only=True)
class ParthoodHypothesis(AnalyzableAnnotation, Nameable):
    """
    Hypothesis about a part of an object: a feature, such as a logo, or a component,
    such as the control unit of an electrical device.

    An inaccurate object pose can place the part on the wrong region, for example on the
    front face when the pose is off by 180 degrees around the up axis.
    """


@dataclass(kw_only=True)
class ParthoodComponentHypothesis(ParthoodHypothesis): ...


@dataclass(kw_only=True)
class ParthoodFeatureHypothesis(ParthoodHypothesis):
    """
    Hypothesis about a surface or visual feature of an object, such as a logo, text, a
    decorative element or a surface pattern.
    """

    ...


@dataclass(kw_only=True)
class OccupiedAnnotation(Annotation):
    """
    Annotation indicating region occupancy.

    This annotation can be used to indicate that a region is occupied.
    """

    occupied: bool = False
    """
    Indicator for whether the region is occupied or not.
    """
