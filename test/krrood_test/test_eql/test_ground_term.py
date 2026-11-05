"""
Tests for how a :class:`~krrood.entity_query_language.ground_term.GroundTerm` identifies
the object it belongs to.
"""

from __future__ import annotations

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.entity_query_language.ground_term import CallTerm

from ..dataset.objects_compared_by_value import PointComparedByValue

# %% method receivers


def test_method_call_is_one_ground_term_whether_its_receiver_is_bound_or_passed():
    point = PointComparedByValue(1.0)

    assert CallTerm.from_call(point.shifted_by, CallArguments((1.0,))) == (
        CallTerm.from_call(PointComparedByValue.shifted_by, CallArguments((point, 1.0)))
    )


def test_method_calls_on_equal_receivers_are_different_ground_terms():
    first = PointComparedByValue(1.0)
    second = PointComparedByValue(1.0)

    assert CallTerm.from_call(first.shifted_by, CallArguments((1.0,))) != (
        CallTerm.from_call(second.shifted_by, CallArguments((1.0,)))
    )


def test_function_compares_its_arguments_by_value():
    def distance(first: PointComparedByValue, second: PointComparedByValue) -> float:
        return abs(first.coordinate - second.coordinate)

    origin = PointComparedByValue(0.0)

    assert CallTerm.from_call(
        distance, CallArguments((origin, PointComparedByValue(1.0)))
    ) == CallTerm.from_call(
        distance, CallArguments((PointComparedByValue(0.0), PointComparedByValue(1.0)))
    )
