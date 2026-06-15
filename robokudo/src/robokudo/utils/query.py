from queue import Queue
from typing_extensions import Any

from py_trees.blackboard import Blackboard

from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from robokudo.identifier import BBIdentifier
from robokudo.types.annotation import (
    Classification,
    LocationAnnotation,
    SemanticColor,
    Shape,
)
from robokudo.types.scene import ObjectHypothesis


class ObjectHypothesisQueryMatcher:
    """Match ObjectHypotheses against requested ObjectDesignator attributes."""

    def __init__(self) -> None:
        self.query_matchers = {
            "uid": self.matches_uid,
            "type": self.matches_type,
            "shape": self.matches_shape,
            "color": self.matches_color,
            "location": self.matches_location,
            "size": self.matches_size,
            "attribute": self.matches_attribute,
            "description": self.matches_description,
        }

    @staticmethod
    def normalize_query_value(value: Any) -> str:
        """Normalize string-like query values for robust comparisons."""
        return str(value).strip().lower()

    @classmethod
    def normalized_values(cls, values: Any) -> set[str]:
        """Return normalized, non-empty string values from scalar or sequence input."""
        if values is None:
            return set()
        if isinstance(values, str):
            normalized_value = cls.normalize_query_value(values)
            return {normalized_value} if normalized_value else set()
        try:
            iterator = iter(values)
        except TypeError:
            normalized_value = cls.normalize_query_value(values)
            return {normalized_value} if normalized_value else set()

        normalized_values = set()
        for value in iterator:
            if not isinstance(value, str):
                try:
                    iter(value)
                except TypeError:
                    pass
                else:
                    normalized_values.update(cls.normalized_values(value))
                    continue
            normalized_value = cls.normalize_query_value(value)
            if normalized_value:
                normalized_values.add(normalized_value)
        return normalized_values

    @staticmethod
    def value_variants(value: Any) -> list[Any]:
        """Return common string-like representations for semantic names."""
        if value is None:
            return []

        variants = [value]
        name = getattr(value, "name", None)
        if name is not None:
            variants.append(name)
        prefix = getattr(value, "prefix", None)
        if prefix is not None and name is not None:
            variants.append(f"{prefix}:{name}")

        return variants

    @classmethod
    def any_requested_value_matches(
        cls, requested_values: Any, candidate_values: list[Any]
    ) -> bool:
        """Return true if any requested value matches any candidate value."""
        requested = cls.normalized_values(requested_values)
        if len(requested) == 0:
            return True

        candidates = set()
        for candidate_value in candidate_values:
            candidates.update(
                cls.normalized_values(cls.value_variants(candidate_value))
            )
        return not requested.isdisjoint(candidates)

    def matches(
        self, object_hypothesis: ObjectHypothesis, requested_object: ObjectDesignator
    ) -> bool:
        """Return true if an object hypothesis satisfies all supported query fields."""
        for field_name, matcher in self.query_matchers.items():
            requested_values = getattr(requested_object, field_name, None)
            if len(self.normalized_values(requested_values)) == 0:
                continue
            if not matcher(object_hypothesis, requested_values):
                return False
        return True

    def matches_uid(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match an ObjectDesignator uid request against the object hypothesis id."""
        return self.any_requested_value_matches(
            requested_values, [object_hypothesis.id]
        )

    def matches_type(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match requested object type against Classification annotations."""
        candidate_values = [
            annotation.classname
            for annotation in object_hypothesis.annotations
            if isinstance(annotation, Classification)
        ]
        return self.any_requested_value_matches(requested_values, candidate_values)

    def matches_shape(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match requested shapes against shape annotations."""
        candidate_values = []
        for annotation in object_hypothesis.annotations:
            if isinstance(annotation, Shape):
                candidate_values.append(annotation.shape_name)
                candidate_values.append(type(annotation).__name__)
        return self.any_requested_value_matches(requested_values, candidate_values)

    def matches_color(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match requested colors against SemanticColor annotations."""
        candidate_values = [
            annotation.color
            for annotation in object_hypothesis.annotations
            if isinstance(annotation, SemanticColor)
        ]
        return self.any_requested_value_matches(requested_values, candidate_values)

    def matches_location(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match requested location against LocationAnnotation annotations."""
        candidate_values = []
        for annotation in object_hypothesis.annotations:
            if not isinstance(annotation, LocationAnnotation):
                continue
            candidate_values.append(annotation.name)
            if annotation.region is not None:
                candidate_values.extend(self.value_variants(annotation.region.name))
        return self.any_requested_value_matches(requested_values, candidate_values)

    def matches_generic_annotation_field(
        self,
        object_hypothesis: ObjectHypothesis,
        requested_values: Any,
        field_name: str,
    ) -> bool:
        """Match requested values against annotations exposing the given field."""
        candidate_values = []
        for annotation in object_hypothesis.annotations:
            if hasattr(annotation, field_name):
                candidate_values.append(getattr(annotation, field_name))
        return self.any_requested_value_matches(requested_values, candidate_values)

    def matches_attribute(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match generic requested attributes when annotations expose an attribute field."""
        return self.matches_generic_annotation_field(
            object_hypothesis, requested_values, "attribute"
        )

    def matches_size(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match semantic size requests when annotations expose a size field."""
        return self.matches_generic_annotation_field(
            object_hypothesis, requested_values, "size"
        )

    def matches_description(
        self, object_hypothesis: ObjectHypothesis, requested_values: Any
    ) -> bool:
        """Match generic requested descriptions when annotations expose a description field."""
        return self.matches_generic_annotation_field(
            object_hypothesis, requested_values, "description"
        )


class QueryHandler(object):
    """
    QueryHandler provides an interface to interact with the ROS Action-based query interface.
    This wrapper eases the use of the various Blackboard variables devoted to the communication with the query interface.
    """

    @staticmethod
    def init_feedback_queue() -> None:
        """Initializes the feedback queue on the Blackboard."""
        blackboard = Blackboard()
        try:
            feedback_queue = blackboard.get(BBIdentifier.QUERY_FEEDBACK)
        except KeyError:
            feedback_queue = None
        if feedback_queue is None:
            feedback_queue = Queue()
            blackboard.set(BBIdentifier.QUERY_FEEDBACK, feedback_queue)

    @staticmethod
    def get_feedback_queue() -> Queue:
        """Retrieves and returns the feedback queue from the Blackboard."""
        QueryHandler.init_feedback_queue()

        blackboard = Blackboard()
        return blackboard.get(BBIdentifier.QUERY_FEEDBACK)

    @staticmethod
    def send_feedback(feedback: Query.Feedback) -> None:
        """Add a feedback part of the Query msg to the feedback queue, ready to be sent.

        :param feedback: The feedback message to send.
        """
        feedback_queue = QueryHandler.get_feedback_queue()
        feedback_queue.put(feedback)

    @staticmethod
    def send_feedback_str(feedback_str: str) -> None:
        """Add a simple string to the feedback to the feedback queue, ready to be sent.

        :param feedback_str: The string to send as feedback.
        """
        feedback_msg = Query.Feedback()
        feedback_msg.feedback = feedback_str
        QueryHandler.send_feedback(feedback_msg)

    @staticmethod
    def send_answer(result: Query.Result) -> None:
        """Raise a standard RoboKudo Query Result as a query answer to the blackboard.

        :param result: The result to raise to the blackboard.
        """
        if not isinstance(result, Query.Result):
            raise TypeError(
                f"Expected standard RoboKudo Query Result type. Got {type(result)}."
                "If you want to send other results, use QueryHandler.send_arbitrary_answer"
            )

        QueryHandler.send_arbitrary_answer(result)

    @staticmethod
    def send_arbitrary_answer(result: Any) -> None:
        """Raise any data as a query answer to the blackboard.

        :param result: The data to raise to the blackboard.
        """
        blackboard = Blackboard()
        blackboard.set(BBIdentifier.QUERY_ANSWER, result)

    @staticmethod
    def preempt_requested() -> bool:
        """Checks whether a preempt request is pending on the blackboard.

        :return: True if a preempt request is pending, False otherwise.
        """
        blackboard = Blackboard()
        try:
            is_requested = blackboard.get(BBIdentifier.QUERY_PREEMPT_REQUESTED)
        except KeyError:
            is_requested = False
        return is_requested

    @staticmethod
    def acknowledge_preempt_request() -> None:
        """Acknowledges the preempt request on the blackboard."""
        blackboard = Blackboard()
        blackboard.set(BBIdentifier.QUERY_PREEMPT_ACK, True)
