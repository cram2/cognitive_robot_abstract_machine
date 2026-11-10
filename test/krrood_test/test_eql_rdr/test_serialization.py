"""
Self-contained tests for the rule-tree unparser
(:mod:`krrood.entity_query_language.rdr.serialization`).

Builds rule trees directly from core EQL primitives and feeds ``rdr_to_python()`` a
minimal stand-in for the parts of :class:`EQLSingleClassRDR` it actually reads, so this
test module -- and the DAG-unparsing slice it covers -- stays testable independently of
the rest of the RDR engine. ``save_rdr``/``save_rdr_with_case``/``load_rdr`` operate on a
real :class:`EQLSingleClassRDR` and are covered by the later, dedicated serialization
test suite once the engine slice lands.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import pytest
from typing_extensions import Any, Optional

from krrood.entity_query_language.factories import add, alternative, entity, refinement, variable
from krrood.entity_query_language.rdr.backward_inference import what_do_we_know_about
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
from krrood.entity_query_language.rdr.exceptions import EmptyRuleTreeError
from krrood.entity_query_language.rdr.serialization import (
    _RDR_MODULE_TEMPLATE_NAME,
    _TEMPLATES_DIRECTORY,
    RDR_CASE_TYPE_NAME,
    RDR_CASE_VARIABLE_NAME,
    RDR_CONCLUSION_ATTRIBUTE_NAME,
    RDR_CORNER_CASES_NAME,
    RDR_QUERY_NAME,
    rdr_to_python,
    walk_rules_in_emission_order,
)


class Species(Enum):
    MAMMAL = "mammal"
    BIRD = "bird"


@dataclass(unsafe_hash=True)
class Animal:
    """Minimal RDR classification target used only by this test module."""

    name: str
    has_fur: bool = False
    can_fly: bool = False
    habitat: Optional[Species] = None


@dataclass
class _SerializableRuleTree:
    """Minimal stand-in for the :class:`EQLSingleClassRDR` attributes ``rdr_to_python`` reads."""

    query: Any
    case_type: type
    case_variable: Any
    conclusion_attribute_name: str
    corner_cases: CornerCaseStore = field(default_factory=CornerCaseStore)


def _flat_tree_rdr():
    animal = variable(Animal, domain=[])
    query = entity(animal).where(animal.has_fur == True)
    with query:
        add(animal.species, Species.MAMMAL)
        with refinement(animal.can_fly == True):
            add(animal.species, Species.BIRD)
    query.build()
    return _SerializableRuleTree(query, Animal, animal, "species")


def _alternative_chain_rdr():
    animal = variable(Animal, domain=[])
    query = entity(animal).where(animal.has_fur == True)
    with query:
        add(animal.species, Species.MAMMAL)
        with alternative(animal.can_fly == True):
            add(animal.species, Species.BIRD)
    query.build()
    return _SerializableRuleTree(query, Animal, animal, "species")


def _exec_generated_module(source: str) -> dict:
    """Execute generated source in a fresh namespace, as the real module loader would."""
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    return namespace


# %% rdr_to_python: structure of the generated source


def test_generated_source_rebuilds_an_equivalent_rule_tree():
    source = rdr_to_python(_flat_tree_rdr())

    namespace = _exec_generated_module(source)

    assert namespace[RDR_CASE_TYPE_NAME] is Animal
    assert namespace[RDR_CONCLUSION_ATTRIBUTE_NAME] == "species"
    assert namespace[RDR_CORNER_CASES_NAME] == {}


def test_generated_tree_has_the_same_backward_inference_knowledge_as_the_original():
    original = _flat_tree_rdr()
    source = rdr_to_python(original)
    namespace = _exec_generated_module(source)
    rebuilt_root = namespace[RDR_QUERY_NAME]._conditions_root_

    original_root = original.query._conditions_root_
    for value in (Species.MAMMAL, Species.BIRD):
        original_knowledge = what_do_we_know_about(original_root, value)
        rebuilt_knowledge = what_do_we_know_about(rebuilt_root, value)
        assert rebuilt_knowledge.is_satisfiable() == original_knowledge.is_satisfiable()
        assert len(rebuilt_knowledge.sufficient_condition_sets) == len(
            original_knowledge.sufficient_condition_sets
        )


def test_generated_tree_evaluates_a_bird_case_the_same_as_the_original():
    original = _flat_tree_rdr()
    source = rdr_to_python(original)
    namespace = _exec_generated_module(source)
    rebuilt_variable = namespace[RDR_CASE_VARIABLE_NAME]
    rebuilt_root = namespace[RDR_QUERY_NAME]._conditions_root_

    bat = Animal("bat", has_fur=True, can_fly=True)
    knowledge = what_do_we_know_about(rebuilt_root, Species.BIRD)
    assert knowledge.sufficient_condition_sets[0].evaluate_against(
        rebuilt_variable, bat
    )


def test_raises_when_the_rdr_has_no_rules():
    empty_rdr = _SerializableRuleTree(None, Animal, variable(Animal, domain=[]), "species")

    with pytest.raises(EmptyRuleTreeError):
        rdr_to_python(empty_rdr)


def test_generated_source_rebuilds_an_equivalent_alternative_chain():
    original = _alternative_chain_rdr()
    source = rdr_to_python(original)
    namespace = _exec_generated_module(source)
    rebuilt_root = namespace[RDR_QUERY_NAME]._conditions_root_

    original_root = original.query._conditions_root_
    for value in (Species.MAMMAL, Species.BIRD):
        original_knowledge = what_do_we_know_about(original_root, value)
        rebuilt_knowledge = what_do_we_know_about(rebuilt_root, value)
        assert rebuilt_knowledge.is_satisfiable() == original_knowledge.is_satisfiable()
        assert len(rebuilt_knowledge.sufficient_condition_sets) == len(
            original_knowledge.sufficient_condition_sets
        )


def _template_docstring() -> str:
    """The docstring the module template opens with, read through the same constants that
    locate it for rendering so the expected text has one source rather than a retyped copy.
    Everything from the first placeholder on is dropped, since a Jinja template is not
    parseable Python."""
    template_source = (Path(_TEMPLATES_DIRECTORY) / _RDR_MODULE_TEMPLATE_NAME).read_text()
    return ast.get_docstring(ast.parse(template_source[: template_source.index("{{")]))


# %% rdr_to_python: the generated module states how it may be edited


def test_generated_source_opens_with_the_template_docstring():
    """The header telling a reader how a generated module may be edited reaches the
    output, rather than only living in the template."""
    source = rdr_to_python(_flat_tree_rdr())

    assert ast.get_docstring(ast.parse(source)) == _template_docstring()


# %% rdr_to_python: corner cases are embedded in the generated source


def test_recorded_corner_case_is_embedded_in_the_generated_source():
    rdr = _flat_tree_rdr()
    rule_node = walk_rules_in_emission_order(rdr.query._conditions_root_)[0]
    corner_case = Animal("cat", has_fur=True, can_fly=False)
    rdr.corner_cases.record_corner_case(rule_node, corner_case)

    source = rdr_to_python(rdr)
    namespace = _exec_generated_module(source)

    assert corner_case in namespace[RDR_CORNER_CASES_NAME].values()


def test_recorded_corner_case_with_an_enum_field_round_trips_through_the_generated_source():
    rdr = _flat_tree_rdr()
    rule_node = walk_rules_in_emission_order(rdr.query._conditions_root_)[0]
    corner_case = Animal("cat", has_fur=True, can_fly=False, habitat=Species.MAMMAL)
    rdr.corner_cases.record_corner_case(rule_node, corner_case)

    source = rdr_to_python(rdr)
    namespace = _exec_generated_module(source)

    assert corner_case in namespace[RDR_CORNER_CASES_NAME].values()
