"""
Self-contained tests for corner-case provenance
(:mod:`krrood.entity_query_language.rdr.corner_case`).

Uses plain dataclasses for the recorded cases and raw core-EQL ``Variable`` instances
(any ``SymbolicExpression`` will do -- only ``._id_`` is used) as stand-ins for rule
condition nodes, so this test module -- and the corner-case provenance slice it covers
-- stays testable independently of the rest of the RDR engine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum

import pytest

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.rdr.corner_case import (
    AsdictCaseSerializer,
    CornerCaseStore,
)
from krrood.entity_query_language.rdr.exceptions import CaseNotSerializableError


class Species(Enum):
    CAT = "cat"
    DOG = "dog"


@dataclass(unsafe_hash=True)
class Owner:
    """
    Inner dataclass of a nested dataclass, used to exercise recursive (de)serialization.
    """

    name: str


@dataclass
class Animal:
    """
    Flat-plus-nested dataclass used as the recorded corner case.
    """

    name: str
    species: Species
    age: int
    owner: Owner


@dataclass
class Unserializable:
    """
    A dataclass with a field type AsdictCaseSerializer does not support.
    """

    tags: list


def _rex() -> Animal:
    return Animal("Rex", Species.DOG, 3, Owner("Alice"))


# %% AsdictCaseSerializer.to_source


def test_to_source_emits_eval_able_constructor_source_for_a_nested_dataclass():
    serializer = AsdictCaseSerializer()

    case_source = serializer.to_source(_rex())

    rebuilt = eval(
        case_source.source,
        {case_type.__name__: case_type for case_type in (Animal, Species, Owner)},
    )
    assert rebuilt == _rex()
    assert case_source.referenced_types == {Animal, Species, Owner}


def test_to_source_raises_for_a_non_dataclass_value():
    serializer = AsdictCaseSerializer()

    with pytest.raises(CaseNotSerializableError) as error:
        serializer.to_source({"not": "a dataclass"})
    assert error.value.value == {"not": "a dataclass"}


def test_to_source_raises_for_an_unsupported_field_type():
    serializer = AsdictCaseSerializer()

    with pytest.raises(CaseNotSerializableError):
        serializer.to_source(Unserializable(tags=["a", "b"]))


# %% AsdictCaseSerializer.from_data


def test_from_data_reconstructs_a_nested_dataclass_from_an_asdict_style_mapping():
    serializer = AsdictCaseSerializer()
    data = dataclasses.asdict(_rex())

    rebuilt = serializer.from_data(data, Animal)

    assert rebuilt == _rex()


# %% CornerCaseStore.record / get


def test_get_returns_the_case_recorded_for_a_node():
    animal = variable(Animal, domain=[])
    store = CornerCaseStore()
    case = _rex()

    store.record(animal, case)

    assert store.get(animal._id_) is case


def test_get_returns_none_for_a_node_with_no_recorded_case():
    animal = variable(Animal, domain=[])
    store = CornerCaseStore()

    assert store.get(animal._id_) is None


def test_get_returns_none_for_a_none_node_id():
    store = CornerCaseStore()

    assert store.get(None) is None


# %% CornerCaseStore.to_ordered_sources / from_ordered_cases


def test_to_ordered_sources_only_includes_nodes_with_a_recorded_case():
    with_case = variable(Animal, domain=[])
    without_case = variable(Animal, domain=[])
    store = CornerCaseStore()
    store.record(with_case, _rex())

    sources = store.to_ordered_sources([with_case, without_case])

    assert set(sources) == {0}
    assert Animal in sources[0].referenced_types


def test_from_ordered_cases_rebuilds_a_store_keyed_by_node_id():
    with_case = variable(Animal, domain=[])
    without_case = variable(Animal, domain=[])
    case = _rex()

    rebuilt = CornerCaseStore.from_ordered_cases([with_case, without_case], {0: case})

    assert rebuilt.get(with_case._id_) == case
    assert rebuilt.get(without_case._id_) is None
