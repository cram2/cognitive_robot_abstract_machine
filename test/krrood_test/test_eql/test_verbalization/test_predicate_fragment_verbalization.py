"""
Tests for predicate verbalization via a required :class:`VerbalizationFragment`.

A :class:`Verbalizable` predicate builds its surface from the typed clause vocabulary (``clause`` /
``Noun`` / ``Verb`` / ``Copula``), so the result composes with the rest of the pipeline: a wrapping
``Not`` negates it inline — *"a Location is not reachable for a Robot"* — instead of the opaque
*"not (a Location is reachable for a Robot)"* a flat string blob would force.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import inference, not_, variable
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.verbalization._example_domain import (
    IsReachable,
    Location,
    Robot,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


def test_verbalizable_predicate_renders_affirmatively():
    assert (
        verbalize_expression(
            inference(IsReachable)(
                location=variable(Location, []), body=variable(Robot, [])
            )
        )
        == "a Location is reachable for a Robot"
    )


def test_wrapping_not_negates_the_predicate_inline():
    """
    A ``Not`` over a verbalizable predicate flips its copula in place rather than
    wrapping the whole clause in *"not (...)"*.
    """
    assert (
        verbalize_expression(
            Not(IsReachable(variable(Location, []), variable(Robot, [])))
        )
        == "a Location is not reachable for a Robot"
    )
