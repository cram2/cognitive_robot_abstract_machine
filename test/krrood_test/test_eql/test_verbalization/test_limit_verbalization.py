"""
Tests for verbalizing a query's ``limit(n)`` (with optional ``ordered_by``) as a ranking
phrase on the selection.

A ``limit`` reshapes the selection's leading phrase rather than adding a trailing
clause: *"the first two Robots"*, *"the top three Employees by salary"*, *"the Employee
with the highest salary"*. When a limit is present the standalone *"ordered by … from
highest to lowest"* clause is suppressed; ordering *without* a limit keeps it. Ordering
by an attribute of the selection (vs. the selection itself) changes where the
superlative attaches.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.pipeline import (
    VerbalizationPipeline,
    verbalize_expression,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    HierarchicalRenderer,
)

from ...dataset.department_and_employee import Employee


def _int():
    return variable(int, [1, 2, 3])


# ── the ranking table: (n, direction, key-relation) → leading phrase ─────────


def test_limit_without_order_first_singular():
    assert verbalize_expression(entity(_int()).limit(1)) == "Find the first int"


def test_limit_without_order_first_plural():
    assert verbalize_expression(entity(_int()).limit(2)) == "Find the first two ints"


def test_descending_self_singular_is_highest():
    v = _int()
    q = entity(v).ordered_by(v, descending=True).limit(1)
    assert verbalize_expression(q) == "Find the highest int"


def test_descending_self_plural_is_top_n():
    v = _int()
    q = entity(v).ordered_by(v, descending=True).limit(3)
    assert verbalize_expression(q) == "Find the top three ints"


def test_ascending_self_singular_is_lowest():
    v = _int()
    q = entity(v).ordered_by(v, descending=False).limit(1)
    assert verbalize_expression(q) == "Find the lowest int"


def test_ascending_self_plural_is_bottom_n():
    v = _int()
    q = entity(v).ordered_by(v, descending=False).limit(3)
    assert verbalize_expression(q) == "Find the bottom three ints"


def test_descending_attribute_singular_is_with_the_highest():
    e = variable(Employee, [])
    q = entity(e).ordered_by(e.salary, descending=True).limit(1)
    assert verbalize_expression(q) == "Find the Employee with the highest salary"


def test_descending_attribute_plural_is_top_n_by():
    e = variable(Employee, [])
    q = entity(e).ordered_by(e.salary, descending=True).limit(3)
    assert verbalize_expression(q) == "Find the top three Employees by salary"


def test_ascending_attribute_singular_is_with_the_lowest():
    e = variable(Employee, [])
    q = entity(e).ordered_by(e.salary, descending=False).limit(1)
    assert verbalize_expression(q) == "Find the Employee with the lowest salary"


def test_ascending_attribute_plural_is_bottom_n_by():
    e = variable(Employee, [])
    q = entity(e).ordered_by(e.salary, descending=False).limit(3)
    assert verbalize_expression(q) == "Find the bottom three Employees by salary"


# ── ordered-by clause: suppressed under limit, kept without it ───────────────


def test_limit_suppresses_the_ordered_by_clause():
    e = variable(Employee, [])
    text = verbalize_expression(
        entity(e).ordered_by(e.salary, descending=True).limit(3)
    )
    assert "ordered by" not in text
    assert "from highest to lowest" not in text


def test_ordering_without_limit_reports_the_ordered_listing():
    """
    An unranked ordered query is a listing, not a search: it reports the whole (plural)
    population and keeps the ordered-by clause.
    """
    e = variable(Employee, [])
    text = verbalize_expression(an(entity(e).ordered_by(e.salary, descending=True)))
    assert text == "Report Employees ordered by their salaries from highest to lowest"


def test_conditioned_ordering_still_reports():
    """
    Report-ness and conditions are orthogonal: a filtered ordered query is still a
    report (a filtered listing), so it keeps "Report" and the plural subject alongside
    its restriction.
    """
    e = variable(Employee, [])
    text = verbalize_expression(an(entity(e).where(e.salary > 5).ordered_by(e.salary)))
    assert (
        text
        == "Report Employees whose salaries are greater than 5, ordered by their salaries from lowest to highest"
    )


# ── composition with WHERE (the ranking selection stays a referring subject) ──


def test_ranking_composes_with_where_clause():
    e = variable(Employee, [])
    q = (
        entity(e)
        .where(e.salary > 1000)
        .ordered_by(e.starting_salary, descending=True)
        .limit(3)
    )
    assert verbalize_expression(q) == (
        "Find the top three Employees by starting_salary whose salaries are greater than 1000"
    )


def test_plural_ranking_subject_pronominalises_as_their():
    e = variable(Employee, [])
    q = (
        entity(e)
        .where(e.department.name == "Sales")
        .ordered_by(e.salary, descending=True)
        .limit(3)
    )
    text = verbalize_expression(q)
    assert "the top three Employees by salary" in text
    assert "their department" in text  # plural subject → "their", not "its"


def test_singular_ranking_subject_pronominalises_as_its():
    e = variable(Employee, [])
    q = (
        entity(e)
        .where(e.department.name == "Sales")
        .ordered_by(e.salary, descending=True)
        .limit(1)
    )
    text = verbalize_expression(q)
    assert "the Employee with the highest salary" in text
    assert "its department" in text


# ── hierarchical rendering ───────────────────────────────────────────────────


def test_ranking_hierarchical_rendering():
    e = variable(Employee, [])
    q = (
        entity(e)
        .where(e.salary > 1000)
        .ordered_by(e.starting_salary, descending=True)
        .limit(3)
    )
    text = VerbalizationPipeline(HierarchicalRenderer(PlainFormatter())).verbalize(q)
    assert text == (
        "Find the top three Employees by starting_salary\n"
        "  - whose salaries are greater than 1000"
    )


# ── unit: morphology + form selection ─────────────────────────────────────────


def test_cardinal_words():
    assert morphology.cardinal(1) == "one"
    assert morphology.cardinal(2) == "two"
    assert morphology.cardinal(3) == "three"


@pytest.mark.parametrize(
    "n, direction, relation, expected",
    [
        ("NONE", "n1", "SELF", "LeadingRankForm"),
        ("DESCENDING", "n1", "SELF", "LeadingRankForm"),
        ("DESCENDING", "n3", "SELF", "LeadingRankForm"),
        ("OTHER_relation", "n3", "OTHER", "LeadingRankForm"),
        ("DESCENDING", "n1", "ATTRIBUTE", "AttributeSuperlativeForm"),
        ("DESCENDING", "n3", "ATTRIBUTE", "AttributeRankedByForm"),
        ("ASCENDING", "n1", "ATTRIBUTE", "AttributeSuperlativeForm"),
        ("ASCENDING", "n3", "ATTRIBUTE", "AttributeRankedByForm"),
    ],
)
def test_ranking_form_selection(n, direction, relation, expected):
    """
    The registry picks the right form per (direction, count, key-relation) — OCP seam.
    """
    from krrood.entity_query_language.verbalization.grammar.query.planner import (
        SortDirection,
        RankingKeyRelation,
        RankingPlan,
    )
    from krrood.entity_query_language.verbalization.grammar.query.ranking import (
        RankingForm,
        RankingRequest,
    )

    direction_value = {
        "NONE": SortDirection.NONE,
        "ASCENDING": SortDirection.ASCENDING,
        "DESCENDING": SortDirection.DESCENDING,
        "OTHER_relation": SortDirection.DESCENDING,
    }[n]
    relation_value = {
        "SELF": RankingKeyRelation.SELF,
        "ATTRIBUTE": RankingKeyRelation.ATTRIBUTE,
        "OTHER": RankingKeyRelation.OTHER,
    }[relation]
    count = 1 if direction == "n1" else 3
    plan = RankingPlan(
        limit_number=count,
        direction=direction_value,
        relation=relation_value,
        order_key=None,
    )
    form = RankingForm.most_applicable(RankingRequest(plan=plan))
    assert form.__name__ == expected
