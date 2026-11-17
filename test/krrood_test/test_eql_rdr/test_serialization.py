"""
Tests for
:meth:`krrood.entity_query_language.rdr.serialization.RippleDownRulesSerializer.to_source`, on rule
trees built from core EQL primitives and a stand-in for the parts of
:class:`EQLSingleClassRDR` it reads.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum

import pytest
from typing_extensions import Any, Optional

from krrood.entity_query_language import factories
from krrood.entity_query_language.factories import (
    add,
    alternative,
    entity,
    refinement,
    variable,
)
from krrood.entity_query_language.rdr.backward_inference import (
    get_conclusion_sufficient_conditions_from_a_rule_tree,
)
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
from krrood.entity_query_language.rdr.exceptions import EmptyRuleTreeError
from krrood.entity_query_language.rdr.serialization import (
    RippleDownRulesModuleAttribute,
    RippleDownRulesSerializer,
)
from krrood.entity_query_language.rules.decomposition import DecomposedRuleTree


class Species(Enum):
    MAMMAL = "mammal"
    BIRD = "bird"


@dataclass(unsafe_hash=True)
class Animal:
    """
    Minimal RDR classification target used only by this test module.
    """

    name: str
    has_fur: bool = False
    can_fly: bool = False
    habitat: Optional[Species] = None


@dataclass(unsafe_hash=True)
class Entity:
    """
    A case type whose lower-camel-case name is also the name of a factory the generated
    module calls.
    """

    has_fur: bool = False
    """
    Whether the entity has fur.
    """

    species: Optional[Species] = None
    """
    What the rules conclude.
    """


@dataclass
class _SerializableRuleTree:
    """
    Minimal stand-in for the :class:`EQLSingleClassRDR` attributes
    :class:`RippleDownRulesSerializer` reads.
    """

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


def _entity_named_case_type_rdr():
    """
    A one-rule tree over :class:`Entity`, whose case variable's natural name is the
    name of a factory.
    """
    entity_case = variable(Entity, domain=[])
    query = entity(entity_case).where(entity_case.has_fur == True)
    with query:
        add(entity_case.species, Species.MAMMAL)
    query.build()
    return _SerializableRuleTree(query, Entity, entity_case, "species")


def _exec_generated_module(source: str) -> dict:
    """
    Execute generated source in a fresh namespace, as the real module loader would.
    """
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    return namespace


# %% RippleDownRulesSerializer.to_source: structure of the generated source


def test_generated_source_rebuilds_an_equivalent_rule_tree():
    """
    The generated module defines the attributes loading reads, with the RDR's values.
    """
    source = RippleDownRulesSerializer(rdr=_flat_tree_rdr()).to_source()

    namespace = _exec_generated_module(source)

    assert namespace[RippleDownRulesModuleAttribute.CASE_TYPE] is Animal
    assert namespace[RippleDownRulesModuleAttribute.CONCLUSION_ATTRIBUTE] == "species"
    assert namespace[RippleDownRulesModuleAttribute.CORNER_CASES] == {}


def test_generated_tree_has_the_same_backward_inference_knowledge_as_the_original():
    """
    Backward inference finds as many sufficient conditions in the rebuilt tree as in the
    original.
    """
    original = _flat_tree_rdr()
    source = RippleDownRulesSerializer(rdr=original).to_source()
    namespace = _exec_generated_module(source)
    rebuilt_root = namespace[RippleDownRulesModuleAttribute.QUERY]._conditions_root_

    original_root = original.query._conditions_root_
    for value in (Species.MAMMAL, Species.BIRD):
        original_knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(
            original_root, value
        )
        rebuilt_knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(
            rebuilt_root, value
        )
        assert rebuilt_knowledge.is_satisfiable() == original_knowledge.is_satisfiable()
        assert len(rebuilt_knowledge.sufficient_condition_sets) == len(
            original_knowledge.sufficient_condition_sets
        )


def test_generated_tree_evaluates_a_bird_case_the_same_as_the_original():
    """
    A case the refinement concludes for satisfies the rebuilt tree's condition for that
    conclusion.
    """
    original = _flat_tree_rdr()
    source = RippleDownRulesSerializer(rdr=original).to_source()
    namespace = _exec_generated_module(source)
    rebuilt_variable = namespace[RippleDownRulesModuleAttribute.CASE_VARIABLE]
    rebuilt_root = namespace[RippleDownRulesModuleAttribute.QUERY]._conditions_root_

    bat = Animal("bat", has_fur=True, can_fly=True)
    knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(
        rebuilt_root, Species.BIRD
    )
    assert knowledge.sufficient_condition_sets[0].evaluate_against(
        shared_variable=rebuilt_variable, case=bat
    )


def _species_concluded_for(root, case_variable, case) -> set[Species]:
    """
    The conclusions whose sufficient conditions the case satisfies in the rule tree.
    """
    concluded = set()
    for value in Species:
        knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(root, value)
        if any(
            condition_set.evaluate_against(shared_variable=case_variable, case=case)
            for condition_set in knowledge.sufficient_condition_sets
        ):
            concluded.add(value)
    return concluded


@pytest.mark.parametrize("rdr_factory", [_flat_tree_rdr, _alternative_chain_rdr])
@pytest.mark.parametrize("has_fur", [False, True])
@pytest.mark.parametrize("can_fly", [False, True])
def test_generated_tree_concludes_the_same_as_the_original_for_every_case(
    rdr_factory, has_fur, can_fly
):
    """
    Whatever case is given, the rebuilt tree concludes exactly what the original does.
    """
    original = rdr_factory()
    namespace = _exec_generated_module(
        RippleDownRulesSerializer(rdr=original).to_source()
    )
    case = Animal("subject", has_fur=has_fur, can_fly=can_fly)

    original_conclusions = _species_concluded_for(
        original.query._conditions_root_, original.case_variable, case
    )
    rebuilt_conclusions = _species_concluded_for(
        namespace[RippleDownRulesModuleAttribute.QUERY]._conditions_root_,
        namespace[RippleDownRulesModuleAttribute.CASE_VARIABLE],
        case,
    )

    assert rebuilt_conclusions == original_conclusions


def test_a_case_type_named_like_a_factory_does_not_shadow_the_factory():
    """
    The case variable never takes the name of a factory the module calls.
    """
    original = _entity_named_case_type_rdr()

    namespace = _exec_generated_module(
        RippleDownRulesSerializer(rdr=original).to_source()
    )

    assert namespace[RippleDownRulesModuleAttribute.CASE_TYPE] is Entity
    assert _species_concluded_for(
        namespace[RippleDownRulesModuleAttribute.QUERY]._conditions_root_,
        namespace[RippleDownRulesModuleAttribute.CASE_VARIABLE],
        Entity(has_fur=True),
    ) == {Species.MAMMAL}


def test_raises_when_the_rdr_has_no_rules():
    """
    An RDR without rules has nothing to write.
    """
    empty_rdr = _SerializableRuleTree(
        None, Animal, variable(Animal, domain=[]), "species"
    )

    with pytest.raises(EmptyRuleTreeError):
        RippleDownRulesSerializer(rdr=empty_rdr).to_source()


def test_generated_source_rebuilds_an_equivalent_alternative_chain():
    """
    An alternative chain rebuilds with the same backward-inference knowledge.
    """
    original = _alternative_chain_rdr()
    source = RippleDownRulesSerializer(rdr=original).to_source()
    namespace = _exec_generated_module(source)
    rebuilt_root = namespace[RippleDownRulesModuleAttribute.QUERY]._conditions_root_

    original_root = original.query._conditions_root_
    for value in (Species.MAMMAL, Species.BIRD):
        original_knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(
            original_root, value
        )
        rebuilt_knowledge = get_conclusion_sufficient_conditions_from_a_rule_tree(
            rebuilt_root, value
        )
        assert rebuilt_knowledge.is_satisfiable() == original_knowledge.is_satisfiable()
        assert len(rebuilt_knowledge.sufficient_condition_sets) == len(
            original_knowledge.sufficient_condition_sets
        )


def _template_docstring() -> str:
    """
    The docstring the module template opens with, read through the same constants that
    locate it for rendering.

    Everything from the first placeholder on is dropped, since a Jinja template is not
    parseable Python.
    """
    template_source = (
        RippleDownRulesSerializer.templates_directory
        / RippleDownRulesSerializer.module_template_name
    ).read_text()
    return ast.get_docstring(ast.parse(template_source[: template_source.index("{{")]))


# %% RippleDownRulesSerializer.to_source: the generated module states how it may be edited


def test_generated_source_opens_with_the_template_docstring():
    """
    The header telling a reader how a generated module may be edited reaches the output,
    rather than only living in the template.
    """
    source = RippleDownRulesSerializer(rdr=_flat_tree_rdr()).to_source()

    assert ast.get_docstring(ast.parse(source)) == _template_docstring()


def test_generated_source_imports_only_the_factories_it_calls():
    """
    A flat tree of refinements imports no factory it does not call, such as
    :func:`~krrood.entity_query_language.factories.alternative`.
    """
    source = RippleDownRulesSerializer(rdr=_flat_tree_rdr()).to_source()

    imported_factory_names = {
        imported.name
        for statement in ast.parse(source).body
        if isinstance(statement, ast.ImportFrom)
        and statement.module == factories.__name__
        for imported in statement.names
    }
    assert imported_factory_names == {
        factory.__name__ for factory in (variable, entity, add, refinement)
    }


# %% RippleDownRulesSerializer.to_source: corner cases are embedded in the generated source


def test_recorded_corner_case_is_embedded_in_the_generated_source():
    """
    A recorded corner case is part of the generated module.
    """
    rdr = _flat_tree_rdr()
    rule_node = DecomposedRuleTree.from_condition(
        condition=rdr.query._conditions_root_
    ).rule_conditions()[0]
    corner_case = Animal("cat", has_fur=True, can_fly=False)
    rdr.corner_cases.record_corner_case(rule_node, corner_case)

    source = RippleDownRulesSerializer(rdr=rdr).to_source()
    namespace = _exec_generated_module(source)

    assert (
        corner_case in namespace[RippleDownRulesModuleAttribute.CORNER_CASES].values()
    )


def test_recorded_corner_case_with_an_enum_field_round_trips_through_the_generated_source():
    """
    A corner case with an enum field evaluates back to an equal case.
    """
    rdr = _flat_tree_rdr()
    rule_node = DecomposedRuleTree.from_condition(
        condition=rdr.query._conditions_root_
    ).rule_conditions()[0]
    corner_case = Animal("cat", has_fur=True, can_fly=False, habitat=Species.MAMMAL)
    rdr.corner_cases.record_corner_case(rule_node, corner_case)

    source = RippleDownRulesSerializer(rdr=rdr).to_source()
    namespace = _exec_generated_module(source)

    assert (
        corner_case in namespace[RippleDownRulesModuleAttribute.CORNER_CASES].values()
    )
