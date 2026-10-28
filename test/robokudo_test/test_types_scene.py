from __future__ import annotations

from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis

# %% latest annotation of a type


class TestLatestAnnotationOfType:
    def test_returns_the_annotation_added_last(self) -> None:
        first, second = PoseAnnotation(), PoseAnnotation()
        hypothesis = ObjectHypothesis(annotations=[first, second])

        assert hypothesis.latest_annotation_of_type(PoseAnnotation) is second

    def test_is_none_without_an_annotation_of_the_type(self) -> None:
        hypothesis = ObjectHypothesis(annotations=[PoseAnnotation()])

        assert hypothesis.latest_annotation_of_type(Classification) is None


# %% class label


class TestClassLabel:
    def test_is_the_class_name_of_the_latest_classification(self) -> None:
        latest = Classification(classname="cup")
        hypothesis = ObjectHypothesis(
            annotations=[Classification(classname="bowl"), latest]
        )

        assert hypothesis.class_label == latest.classname

    def test_is_none_without_a_classification(self) -> None:
        assert ObjectHypothesis().class_label is None

    def test_is_none_for_a_classification_without_a_class_name(self) -> None:
        hypothesis = ObjectHypothesis(annotations=[Classification()])

        assert hypothesis.class_label is None
