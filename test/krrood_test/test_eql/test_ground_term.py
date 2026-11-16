"""
Tests for how a :class:`~krrood.entity_query_language.ground_term.GroundTerm` identifies
the object it belongs to.
"""

from __future__ import annotations

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.entity_query_language.ground_term import CallTerm, ObjectIdentity

from ..dataset.objects_compared_by_value import PointComparedByValue

# %% object identities


def test_identity_of_an_object_differs_from_the_object_itself():
    point = PointComparedByValue(coordinate=1.0)

    assert ObjectIdentity(identified_object=point) != point
    assert point != ObjectIdentity(identified_object=point)


def test_identities_of_one_object_are_equal_and_hash_alike():
    point = PointComparedByValue(coordinate=1.0)

    assert ObjectIdentity(identified_object=point) == ObjectIdentity(
        identified_object=point
    )
    assert hash(ObjectIdentity(identified_object=point)) == hash(
        ObjectIdentity(identified_object=point)
    )


def test_identities_of_equal_objects_differ():
    assert ObjectIdentity(
        identified_object=PointComparedByValue(coordinate=1.0)
    ) != ObjectIdentity(identified_object=PointComparedByValue(coordinate=1.0))


# %% method receivers


def test_method_call_is_one_ground_term_whether_its_receiver_is_bound_or_passed():
    point = PointComparedByValue(coordinate=1.0)

    assert CallTerm.from_call(
        callee=point.shifted_by, arguments=CallArguments(keyword={"distance": 1.0})
    ) == (
        CallTerm.from_call(
            callee=PointComparedByValue.shifted_by,
            arguments=CallArguments(positional=(point,), keyword={"distance": 1.0}),
        )
    )


def test_method_calls_on_equal_receivers_are_different_ground_terms():
    first = PointComparedByValue(coordinate=1.0)
    second = PointComparedByValue(coordinate=1.0)

    assert CallTerm.from_call(
        callee=first.shifted_by, arguments=CallArguments(keyword={"distance": 1.0})
    ) != (
        CallTerm.from_call(
            callee=second.shifted_by, arguments=CallArguments(keyword={"distance": 1.0})
        )
    )


def test_function_compares_its_arguments_by_value():
    def distance(first: PointComparedByValue, second: PointComparedByValue) -> float:
        return abs(first.coordinate - second.coordinate)

    origin = PointComparedByValue(coordinate=0.0)

    assert CallTerm.from_call(
        callee=distance,
        arguments=CallArguments(
            positional=(origin, PointComparedByValue(coordinate=1.0))
        ),
    ) == CallTerm.from_call(
        callee=distance,
        arguments=CallArguments(
            positional=(
                PointComparedByValue(coordinate=0.0),
                PointComparedByValue(coordinate=1.0),
            )
        ),
    )
