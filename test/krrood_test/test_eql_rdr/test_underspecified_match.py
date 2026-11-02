"""
Self-contained tests for the underspecified-match adapter
(:mod:`krrood.entity_query_language.rdr.underspecified`).

Builds real ``Match`` objects directly via the public ``an(Type)(...).from_(domain)``
factory rather than through :class:`EQLSingleClassRDR`, so this test module -- and the
underspecified-inference slice it covers -- stays testable independently of the rest of
the RDR engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import List, Optional

from krrood.entity_query_language.factories import an
from krrood.entity_query_language.rdr.exceptions import (
    MultipleInferenceTargets,
    NoInferenceTarget,
    UndeclaredInferenceTarget,
    UnsupportedInferenceTarget,
)
from krrood.entity_query_language.rdr.underspecified import RippleDownRulesMatchParser


@dataclass(unsafe_hash=True)
class Animal:
    """
    Minimal RDR classification target used only by this test module.
    """

    name: str
    has_fur: bool = False
    species: Optional[str] = None
    diet: Optional[str] = None


@dataclass(unsafe_hash=True)
class TaggedAnimal:
    """
    A case type with an unbounded-iterable field, to exercise the unsupported-target
    guard.
    """

    name: str
    tags: List[str] = field(default_factory=list)


class PlainAnimal:
    """
    A case type that is not a dataclass, so it declares no fields.
    """

    def __init__(self, name: str, species: str | None = None) -> None:
        self.name = name
        self.species = species


@dataclass(unsafe_hash=True)
class AnimalWithDerivedLabel:
    """
    A case type whose ``label`` is a property rather than a declared field.
    """

    name: str

    @property
    def label(self) -> str:
        """
        The animal's name, upper-cased.
        """
        return self.name.upper()


def _animals():
    return [
        Animal("cat", has_fur=True, species="mammal", diet="carnivore"),
        Animal("snake", has_fur=False, species="reptile", diet="carnivore"),
    ]


# %% case_type / variable


def test_case_type_and_variable_reflect_the_underlying_match():
    match = an(Animal)(species=...).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    assert parser.case_type is Animal
    assert parser.variable is match._variable_


# %% inference_targets / single_target / target_attribute_name


def test_single_ellipsis_attribute_is_the_sole_inference_target():
    match = an(Animal)(has_fur=True, species=...).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    assert len(parser.inference_targets) == 1
    assert parser.single_target() is parser.inference_targets[0]
    assert parser.target_attribute_name == "species"


def test_no_ellipsis_attribute_raises_no_inference_target():
    match = an(Animal)(has_fur=True).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    with pytest.raises(NoInferenceTarget):
        parser.single_target()


def test_multiple_ellipsis_attributes_raise_multiple_inference_targets():
    match = an(Animal)(species=..., diet=...).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    with pytest.raises(MultipleInferenceTargets) as error:
        parser.single_target()
    assert set(error.value.attribute_names) == {"species", "diet"}


def test_unbounded_iterable_ellipsis_attribute_raises_unsupported_inference_target():
    match = an(TaggedAnimal)(tags=...).from_([TaggedAnimal("cat", tags=["fluffy"])])
    parser = RippleDownRulesMatchParser(match)

    with pytest.raises(UnsupportedInferenceTarget) as error:
        parser.single_target()
    assert error.value.case_type is TaggedAnimal
    assert error.value.attribute_name == "tags"


def test_a_target_on_a_case_type_that_is_not_a_dataclass_is_undeclared():
    match = an(PlainAnimal)(species=...).from_([PlainAnimal("cat")])
    parser = RippleDownRulesMatchParser(match)

    with pytest.raises(UndeclaredInferenceTarget) as error:
        parser.single_target()
    assert error.value.case_type is PlainAnimal
    assert error.value.attribute_name == "species"


def test_a_target_that_is_not_a_field_of_the_case_type_is_undeclared():
    match = an(AnimalWithDerivedLabel)(label=...).from_([AnimalWithDerivedLabel("cat")])
    parser = RippleDownRulesMatchParser(match)

    with pytest.raises(UndeclaredInferenceTarget) as error:
        parser.single_target()
    assert error.value.case_type is AnimalWithDerivedLabel
    assert error.value.attribute_name == "label"


# %% filtered_cases


def test_filtered_cases_keeps_only_instances_matching_the_concrete_constraints():
    match = an(Animal)(has_fur=True, species=...).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    cases = list(parser.filtered_cases())

    assert cases == [Animal("cat", has_fur=True, species="mammal", diet="carnivore")]


def test_filtered_cases_with_no_concrete_constraints_yields_the_whole_domain():
    match = an(Animal)(species=...).from_(_animals())
    parser = RippleDownRulesMatchParser(match)

    cases = list(parser.filtered_cases())

    assert cases == _animals()
