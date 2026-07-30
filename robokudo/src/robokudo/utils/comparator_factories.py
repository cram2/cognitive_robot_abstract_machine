from typing_extensions import Dict, Optional, Type

from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    ColorHistogram,
    PoseAnnotation,
    PositionAnnotation,
    SemanticColor,
    SIFTAnnotation,
    StampedPoseAnnotation,
    StampedPositionAnnotation,
    StampedTransformAnnotation,
)
from robokudo.types.core import Annotation
from robokudo.types.core import Type as RkType
from robokudo.types.cv import ImageROI, Rect
from robokudo.types.tf import (
    Pose,
    Position,
    StampedPose,
    StampedPosition,
    StampedTransform,
)
from robokudo.utils.comparators import (
    BboxComparator,
    ClassificationComparator,
    FeatureComparator,
    HistogramComparator,
    ImageROIComparator,
    PoseComparator,
    RoiComparator,
    SemanticColorComparator,
    SIFTComparator,
    TranslationComparator,
)


class FeatureComparatorFactory:
    """
    A factory for creating feature comparators.
    """

    annotator_comparators: Dict[Type[Annotation], Type[FeatureComparator]] = {
        BoundingBox3DAnnotation: BboxComparator,
        Classification: ClassificationComparator,
        ColorHistogram: HistogramComparator,
        SemanticColor: SemanticColorComparator,
        PositionAnnotation: TranslationComparator,
        StampedPositionAnnotation: TranslationComparator,
        PoseAnnotation: PoseComparator,
        StampedPoseAnnotation: PoseComparator,
        StampedTransformAnnotation: PoseComparator,
        SIFTAnnotation: SIFTComparator,
    }
    """
    Mapping of annotation types to feature comparators.
    """

    type_comparators: Dict[Type[RkType], Type[FeatureComparator]] = {
        Rect: RoiComparator,
        ImageROI: ImageROIComparator,
        Position: TranslationComparator,
        StampedPosition: TranslationComparator,
        Pose: PoseComparator,
        StampedPose: PoseComparator,
        StampedTransform: PoseComparator,
    }
    """
    Mapping of robokudo types to feature comparators.
    """

    @classmethod
    def for_annotation(
        cls, annotation: Type[Annotation], weight: float, **kwargs
    ) -> Optional[FeatureComparator]:
        """
        Get a feature comparator for the given annotation type and assign it the given
        weight.

        :param annotation: The annotation type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if
            the type is not supported.
        """
        if annotation not in cls.annotator_comparators:
            return None
        return FeatureComparatorFactory.annotator_comparators[annotation](
            weight, **kwargs
        )

    @classmethod
    def for_type(
        cls, rk_type: Type[RkType], weight: float, **kwargs
    ) -> Optional[FeatureComparator]:
        """
        Get a feature comparator for the given robokudo type and assign it the given
        weight.

        :param rk_type: The robokudo type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if
            the type is not supported.
        """
        if rk_type not in cls.type_comparators:
            return None
        return FeatureComparatorFactory.type_comparators[rk_type](weight, **kwargs)
