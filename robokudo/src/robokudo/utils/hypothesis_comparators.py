from typing_extensions import Dict, Type, Union, Callable, TypeVar, Generic

from robokudo.cas import CAS
from robokudo.types.core import Annotation, Type as RkType
from robokudo.types.cv import ImageROI
from robokudo.types.scene import AnalyzableAnnotation, ObjectHypothesis
from robokudo.utils.comparator_factories import FeatureComparatorFactory
from robokudo.utils.comparators import FeatureComparator

THyothesis = TypeVar("THyothesis", bound=AnalyzableAnnotation)


class HypothesisComparator(Generic[THyothesis]):
    """A comparator that can compare two hypotheses."""

    def __init__(self, cas_fn: Callable[[], CAS]) -> None:
        self.cas_fn = cas_fn
        """Function to get a CAS instance, to be used for annotation filtering."""

        self.annotation_comparators: Dict[Type[Annotation], FeatureComparator] = {}
        """Feature comparators used by this hypothesis comparator to compare annotations."""

        self.type_comparators: Dict[Type[RkType], FeatureComparator] = {}
        """Feature comparators used by this hypothesis comparator to compare other robokudo types."""

    def compute_similarity(
        self, hypothesis1: THyothesis, hypothesis2: THyothesis
    ) -> float:
        """Compute the similarity between two hypotheses.

        :param hypothesis1: The first hypothesis to compare.
        :param hypothesis2: The second hypothesis to compare.
        :return: The similarity of the two hypotheses between 0.0 (no similarity) and 1.0 (identical).
        """
        raise NotImplementedError()

    def with_comparator_for(
        self,
        compared_type: Type[Union[Annotation, RkType]],
        weight: float,
        **kwargs,
    ) -> "HypothesisComparator":
        """Add a feature comparator for the given annotation using the given weight to the comparator.

        :param compared_type: The type that should be compared by the hypothesis comparator.
        :param weight: The weight the hypothesis comparator should use for the compared type.
        :return: The hypothesis comparator with the added comparator.
        :raises KeyError: If no comparator can be found for the given type.
        :raises ValueError: If the given type cannot be compared.
        """

        if issubclass(compared_type, Annotation):
            comparator = FeatureComparatorFactory.for_annotation(
                compared_type, weight, **kwargs
            )
            if not comparator:
                raise KeyError(f"No comparator found for annotation {compared_type}")

            self.annotation_comparators[compared_type] = comparator
        elif issubclass(compared_type, RkType):
            comparator = FeatureComparatorFactory.for_type(
                compared_type, weight, **kwargs
            )
            if not comparator:
                raise KeyError(f"No comparator found for annotation {compared_type}")
            self.type_comparators[compared_type] = comparator
        else:
            raise ValueError(f"Unknown comparator type {compared_type}")

        return self


class ObjectHypothesisComparator(HypothesisComparator[ObjectHypothesis]):
    def compute_similarity(
        self, hypothesis1: ObjectHypothesis, hypothesis2: ObjectHypothesis
    ) -> float:
        """Compute the similarity between two hypotheses.

        :param hypothesis1: The first hypothesis to compare.
        :param hypothesis2: The second hypothesis to compare.
        :return: The similarity of the two hypotheses between 0.0 (no similarity) and 1.0 (identical).
        """

        total_weight = 0.0
        total_similarity = 0.0

        # Compare ROI?
        if ImageROI in self.type_comparators:
            image_roi_comparator = self.type_comparators[ImageROI]
            similarity = image_roi_comparator.compute_similarity(
                hypothesis1.roi, hypothesis2.roi
            )
            total_weight += image_roi_comparator.weight
            total_similarity += similarity * image_roi_comparator.weight

        cas = self.cas_fn()

        # Compare annotations
        for annotation_type in self.annotation_comparators.keys():
            comparator = self.annotation_comparators[annotation_type]

            annotations1 = cas.filter_by_type(annotation_type, hypothesis1.annotations)
            annotations2 = cas.filter_by_type(annotation_type, hypothesis2.annotations)

            total_compared = 0
            for ann1 in annotations1:
                for ann2 in annotations2:
                    # Compare only annotations from the same source
                    if ann1.source == ann2.source:
                        similarity = comparator.compute_similarity(ann1, ann2)

                        total_weight += comparator.weight
                        total_similarity += similarity * comparator.weight

                        total_compared += 1
                        break

        if total_weight == 0.0:
            return 0.0

        return total_similarity / total_weight
