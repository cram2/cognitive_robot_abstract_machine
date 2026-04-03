from typing_extensions import Type, Dict, Optional

from robokudo.types.annotation import (
    Classification,
    BoundingBox3DAnnotation,
    ColorHistogram,
    SemanticColor,
    PositionAnnotation,
)
from robokudo.types.core import Annotation
from robokudo.types.core import Type as RkType
from robokudo.types.cv import ImageROI, Rect
from robokudo.utils.comparators import (
    FeatureComparator,
    BboxComparator,
    ClassificationComparator,
    HistogramComparator,
    SemanticColorComparator,
    ImageROIComparator,
    RoiComparator,
    TranslationComparator,
)


class FeatureComparatorFactory:
    """A factory for creating feature comparators."""

    annotator_comparators: Dict[Type[Annotation], Type[FeatureComparator]] = {
        BoundingBox3DAnnotation: BboxComparator,
        Classification: ClassificationComparator,
        ColorHistogram: HistogramComparator,
        SemanticColor: SemanticColorComparator,
        ImageROI: ImageROIComparator,
        PositionAnnotation: TranslationComparator,
    }
    """Mapping of annotation types to feature comparators."""

    type_comparators: Dict[Type[RkType], Type[FeatureComparator]] = {
        Rect: RoiComparator
    }

    @classmethod
    def for_annotation(
        cls, annotation: Type[Annotation], weight: float
    ) -> Optional[FeatureComparator]:
        """Get a feature comparator for the given annotation type and assign it the given weight.

        :param annotation: The annotation type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if the type is not supported.
        """
        if annotation not in cls.annotator_comparators:
            return None
        return FeatureComparatorFactory.annotator_comparators[annotation](weight)

    @classmethod
    def for_type(
        cls, rk_type: Type[RkType], weight: float
    ) -> Optional[FeatureComparator]:
        """Get a feature comparator for the given robokudo type and assign it the given weight.

        :param rk_type: The robokudo type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if the type is not supported.
        """
