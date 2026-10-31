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
    UnsupportedInferenceTarget,
)
from krrood.entity_query_language.rdr.underspecified import RDRMatchParser


@dataclass(unsafe_hash=True)
class Animal:
    """Minimal RDR classification target used only by this test module."""

    name: str
    has_fur: bool = False
    species: Optional[str] = None
    diet: Optional[str] = None


@dataclass(unsafe_hash=True)
class TaggedAnimal:
    """A case type with an unbounded-iterable field, to exercise the unsupported-target guard."""

    name: str
    tags: List[str] = field(default_factory=list)


def _animals():
    return [
        Animal("cat", has_fur=True, species="mammal", diet="carnivore"),
        Animal("snake", has_fur=False, species="reptile", diet="carnivore"),
    ]


# %% case_type / variable


def test_case_type_and_variable_reflect_the_underlying_match():
    match = an(Animal)(species=...).from_(_animals())
    parser = RDRMatchParser(match)

    assert parser.case_type is Animal
    assert parser.variable is match._variable_


# %% inference_targets / single_target / target_attribute_name


def test_single_ellipsis_attribute_is_the_sole_inference_target():
    match = an(Animal)(has_fur=True, species=...).from_(_animals())
    parser = RDRMatchParser(match)

    assert len(parser.inference_targets) == 1
    assert parser.single_target() is parser.inference_targets[0]
    assert parser.target_attribute_name == "species"


def test_no_ellipsis_attribute_raises_no_inference_target():
    match = an(Animal)(has_fur=True).from_(_animals())
    parser = RDRMatchParser(match)

    with pytest.raises(NoInferenceTarget):
        parser.single_target()


def test_multiple_ellipsis_attributes_raise_multiple_inference_targets():
    match = an(Animal)(species=..., diet=...).from_(_animals())
    parser = RDRMatchParser(match)

    with pytest.raises(MultipleInferenceTargets) as error:
        parser.single_target()
    assert set(error.value.attribute_names) == {"species", "diet"}


def test_unbounded_iterable_ellipsis_attribute_raises_unsupported_inference_target():
    match = an(TaggedAnimal)(tags=...).from_([TaggedAnimal("cat", tags=["fluffy"])])
    parser = RDRMatchParser(match)

    with pytest.raises(UnsupportedInferenceTarget) as error:
        parser.single_target()
    assert error.value.case_type is TaggedAnimal
    assert error.value.attribute_name == "tags"


# %% filtered_cases


def test_filtered_cases_keeps_only_instances_matching_the_concrete_constraints():
    match = an(Animal)(has_fur=True, species=...).from_(_animals())
    parser = RDRMatchParser(match)

    cases = list(parser.filtered_cases())

    assert cases == [Animal("cat", has_fur=True, species="mammal", diet="carnivore")]


def test_filtered_cases_with_no_concrete_constraints_yields_the_whole_domain():
    match = an(Animal)(species=...).from_(_animals())
    parser = RDRMatchParser(match)

    cases = list(parser.filtered_cases())

    assert cases == _animals()
