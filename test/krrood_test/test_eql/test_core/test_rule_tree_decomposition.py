"""Tests for :mod:`krrood.entity_query_language.rules.decomposition`."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import (
    add,
    alternative,
    entity,
    refinement,
    variable,
)
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.decomposition import DecomposedRuleTree

# %% mimic


@dataclass(kw_only=True)
class LabelledThing:
    """A case whose rules each conclude a distinct label, so a rule is known by
    its label."""

    size: int = 0
    """The value the rule conditions test."""

    label: str | None = None
    """The concluded attribute."""


def _concluded_label(condition: SymbolicExpression) -> Any:
    """The label the rule with this condition concludes."""
    return next(
        conclusion.right._value_
        for conclusion in condition._conclusions_
        if isinstance(conclusion, Add)
    )


# %% rule order


def test_rule_conditions_keep_sibling_branches_of_each_kind_in_the_order_added():
    """The root's alternatives wrap it from outside, so they come before its
    refinements; among siblings of one kind the order they were added in is
    kept."""
    thing = variable(LabelledThing, domain=[])
    query = entity(thing).where(thing.size > 0)
    with query:
        add(thing.label, "base")
        with refinement(thing.size > 1):
            add(thing.label, "first refinement")
        with refinement(thing.size > 2):
            add(thing.label, "second refinement")
            with alternative(thing.size > 3):
                add(thing.label, "alternative inside the second refinement")
        with alternative(thing.size < 0):
            add(thing.label, "first alternative")
        with alternative(thing.size < -1):
            add(thing.label, "second alternative")
    query.build()

    rule_tree = DecomposedRuleTree.from_condition(condition=query._conditions_root_)

    assert [
        _concluded_label(condition) for condition in rule_tree.rule_conditions()
    ] == [
        "base",
        "first alternative",
        "second alternative",
        "first refinement",
        "second refinement",
        "alternative inside the second refinement",
    ]


def test_a_bare_condition_decomposes_into_itself_without_branches():
    """A condition no conclusion selector wraps is its own base condition."""
    thing = variable(LabelledThing, domain=[])
    condition = thing.size > 0

    rule_tree = DecomposedRuleTree.from_condition(condition=condition)

    assert rule_tree.base_condition is condition
    assert rule_tree.branches == []
