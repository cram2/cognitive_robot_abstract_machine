"""
Tests of the object hypothesis of :mod:`robokudo.types.scene`.
"""

from __future__ import annotations

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    Human,
    PerceivedObject,
)

from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis

# %% latest annotation of a type


class TestLatestAnnotationOfType:
    """
    :meth:`ObjectHypothesis.latest_annotation_of_type` finds the last annotation of a
    type.
    """

    def test_returns_the_annotation_added_last(self) -> None:
        """
        Of two annotations of the type, the one added second is returned.
        """
        first, second = PoseAnnotation(), PoseAnnotation()
        hypothesis = ObjectHypothesis(annotations=[first, second])

        assert (
            hypothesis.latest_annotation_of_type(annotation_type=PoseAnnotation)
            is second
        )

    def test_is_none_without_an_annotation_of_the_type(self) -> None:
        """
        A hypothesis without an annotation of the type has none to return.
        """
        hypothesis = ObjectHypothesis(annotations=[PoseAnnotation()])

        assert (
            hypothesis.latest_annotation_of_type(annotation_type=Classification) is None
        )


# %% class label


class TestClassLabel:
    """
    The class label of an object hypothesis.
    """

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


# %% semantic class


class TestSemanticClass:
    """
    The semantic annotation a hypothesis names, however confident its classification.
    """

    def test_is_the_semantic_annotation_the_class_label_names(self) -> None:
        hypothesis = ObjectHypothesis(
            annotations=[Classification(classname="cup", confidence=0.0)]
        )

        assert hypothesis.semantic_class is Cup

    def test_is_none_for_a_class_label_naming_no_semantic_annotation(self) -> None:
        hypothesis = ObjectHypothesis(annotations=[Classification(classname="fork")])

        assert hypothesis.semantic_class is None

    def test_is_none_without_a_classification(self) -> None:
        assert ObjectHypothesis().semantic_class is None

    def test_is_human_for_a_human_whatever_its_class_label(self) -> None:
        hypothesis = HumanHypothesis(annotations=[Classification(classname="cup")])

        assert hypothesis.semantic_class is Human


# %% believed semantic class


class TestBelievedSemanticClass:
    """
    The semantic annotation a hypothesis is believed to be, given how confident a
    classification has to be.
    """

    def test_is_the_semantic_class_of_a_confident_classification(self) -> None:
        minimum_class_confidence = 0.5
        hypothesis = ObjectHypothesis(
            annotations=[
                Classification(classname="cup", confidence=minimum_class_confidence)
            ]
        )

        assert (
            hypothesis.believed_semantic_class(
                minimum_class_confidence=minimum_class_confidence
            )
            is Cup
        )

    def test_is_a_perceived_object_below_the_minimum_confidence(self) -> None:
        hypothesis = ObjectHypothesis(
            annotations=[Classification(classname="cup", confidence=0.4)]
        )

        assert (
            hypothesis.believed_semantic_class(minimum_class_confidence=0.5)
            is PerceivedObject
        )

    def test_is_a_perceived_object_for_a_label_naming_no_semantic_annotation(
        self,
    ) -> None:
        hypothesis = ObjectHypothesis(
            annotations=[Classification(classname="fork", confidence=1.0)]
        )

        assert (
            hypothesis.believed_semantic_class(minimum_class_confidence=0.5)
            is PerceivedObject
        )

    def test_is_human_for_an_unclassified_human(self) -> None:
        assert (
            HumanHypothesis().believed_semantic_class(minimum_class_confidence=0.5)
            is Human
        )
