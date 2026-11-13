"""Tests for the condition and branch emitter families of
:mod:`krrood.entity_query_language.source_emission`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.code_generation.ast_helpers import ast_unparse_expression
from krrood.code_generation.exceptions import NoApplicableSourceEmitterError
from krrood.entity_query_language.factories import and_, not_, or_, variable
from krrood.entity_query_language.rules.conclusion_selector import ConclusionSelector
from krrood.entity_query_language.source_emission import (
    BranchSourceEmitter,
    ConditionEmissionContext,
    ConditionSourceEmitter,
    ConjunctionSourceEmitter,
    RefinementBranchSourceEmitter,
)
from krrood.patterns.specificity_ranking import concrete_subclasses


@dataclass(kw_only=True)
class Door:
    """A case type whose attributes the emitted conditions read."""

    is_open: bool = False
    """Whether the door is open."""

    width: float = 0.0
    """The door's width."""


# %% conditions


def test_emitted_condition_reads_as_the_authored_condition():
    """Comparisons, attribute accesses, literals and connectives are emitted in
    the syntax they are authored in."""
    door = variable(Door, domain=[])
    condition = and_(door.is_open == True, or_(door.width > 0.5, not_(door.width < 2)))
    context = ConditionEmissionContext(variable_names_by_id={door._id_: "door"})

    source = ast_unparse_expression(
        ConditionSourceEmitter.emit_source(subject=condition, context=context)
    )

    assert source == (
        "and_(door.is_open == True, or_(door.width > 0.5, not_(door.width < 2)))"
    )


def test_emitting_a_variable_without_a_source_name_raises():
    """A variable the context gives no name has no emitted form."""
    door = variable(Door, domain=[])
    context = ConditionEmissionContext(variable_names_by_id={})

    with pytest.raises(NoApplicableSourceEmitterError) as error:
        ConditionSourceEmitter.emit_source(subject=door, context=context)
    assert error.value.subject is door
    assert error.value.family is ConditionSourceEmitter


def test_condition_emitter_belongs_to_the_condition_family():
    """An emitter's family is the family base it descends from."""
    assert ConjunctionSourceEmitter.family is ConditionSourceEmitter


# %% branches


def test_branch_emitter_belongs_to_the_branch_family():
    """A branch emitter is chosen among the branch emitters, not the condition
    emitters."""
    assert RefinementBranchSourceEmitter.family is BranchSourceEmitter


def test_every_conclusion_selector_has_a_branch_emitter():
    """A rule tree using any conclusion selector EQL defines can be emitted."""
    eql_conclusion_selectors = {
        selector
        for selector in concrete_subclasses(ConclusionSelector)
        if selector.__module__ == ConclusionSelector.__module__
    }

    assert {
        emitter.emitted_type for emitter in BranchSourceEmitter.alternatives()
    } == eql_conclusion_selectors
