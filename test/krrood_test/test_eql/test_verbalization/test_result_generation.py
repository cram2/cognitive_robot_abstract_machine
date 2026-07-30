"""
Tests for :class:`VerbalizationResultGenerator`.

``verbalization_results.py`` itself is regenerated for the real ``krrood`` package by
``conftest.py`` on every test run, so there is no separate test asserting it matches
what the generator produces -- it always does, by construction. What remains worth
testing here is the generation logic itself, against a small controlled domain.
"""

from __future__ import annotations

import krrood
from krrood.entity_query_language.predicate import HasType, HasTypes
from krrood.entity_query_language.testing.result_generation import (
    VerbalizationResultGenerator,
)
from krrood.entity_query_language.testing.result_verification import (
    VerbalizationResultsOfPackage,
)
from krrood.entity_query_language.verbalization import _example_domain

# %% generation against a small, controlled domain


def test_generated_results_pass_their_own_snapshot_verification():
    """
    Feeding the generator's ``covered_results()`` into a fresh snapshot passes both
    verification assertions -- the same coverage and wording checks a hand-written entry
    has to pass, against the real objects the generator produces.
    """
    snapshot = VerbalizationResultsOfPackage(package=_example_domain, results=())
    generator = VerbalizationResultGenerator(snapshot=snapshot)

    round_trip_snapshot = VerbalizationResultsOfPackage(
        package=_example_domain, results=generator.covered_results()
    )
    round_trip_snapshot.assert_results_cover_every_callable()
    round_trip_snapshot.assert_declared_results_render_as_stated()


# %% literal example values instead of placeholder variables


def test_placeholder_example_value_names_itself_instead_of_a_placeholder():
    """
    ``HasType``/``HasTypes`` override ``_example_operand_values_`` because ``types_`` is
    never bound to a symbolic operand in real usage -- only ever a literal.

    Their placeholder rendering names the literal's own value, the same way it would for
    a real query, instead of falling back to a generic *"a Type"* placeholder.
    """
    snapshot = VerbalizationResultsOfPackage(package=krrood, results=())

    assert snapshot.rendered_result(HasType) == "a variable is of type Integer"
    assert snapshot.rendered_result(HasTypes) == "a variable is of type Integer or Text"
