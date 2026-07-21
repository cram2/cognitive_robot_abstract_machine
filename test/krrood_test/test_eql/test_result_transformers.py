"""Tests for the query result-transformer pipeline and its inspectability.

Ordering and quantification are composed result-stream stages on the compiled query, exposed through
``_result_stages_`` so a meta-inspector can identify how a query is ordered and quantified without
traversing the expression tree or evaluating it.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import entity, variable, an, the
from krrood.entity_query_language.query.quantifiers import An, The
from krrood.entity_query_language.query.result_transformers import (
    Ordering,
    Quantification,
)


def test_result_stages_reports_the_quantifier_kind():
    """The pipeline exposes the quantifier kind, so an inspector can tell ``an`` from ``the`` without
    evaluating or traversing the expression tree."""
    value = variable(int, [1])

    an_stages = an(entity(value))._result_stages_
    assert any(
        isinstance(stage, Quantification) and stage.quantifier_type is An
        for stage in an_stages
    )

    the_stages = the(entity(value))._result_stages_
    assert any(
        isinstance(stage, Quantification) and stage.quantifier_type is The
        for stage in the_stages
    )


def test_result_stages_includes_an_ordering_stage_when_ordered():
    """An ordered query exposes an :class:`Ordering` stage carrying its direction."""
    value = variable(int, [3, 1, 2])
    ordered_query = an(entity(value).ordered_by(value, descending=True))

    assert any(
        isinstance(stage, Ordering) and stage.descending
        for stage in ordered_query._result_stages_
    )


def test_result_stages_has_no_ordering_stage_when_unordered():
    """An unordered query exposes no ordering stage."""
    value = variable(int, [3, 1, 2])

    assert not any(
        isinstance(stage, Ordering) for stage in an(entity(value))._result_stages_
    )
