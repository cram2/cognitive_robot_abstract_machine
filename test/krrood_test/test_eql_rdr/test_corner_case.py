"""
Self-contained tests for corner-case provenance
(:mod:`krrood.entity_query_language.rdr.corner_case`).

Uses plain dataclasses for the recorded cases and raw core-EQL ``Variable`` instances
(any ``SymbolicExpression`` will do -- only ``._id_`` is used) as stand-ins for rule
condition nodes, so this test module -- and the corner-case provenance slice it covers
-- stays testable independently of the rest of the RDR engine.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
from test.krrood_test.dataset.nested_dataclass_records import Animal, rex

# %% CornerCaseStore.record_corner_case / get_corner_case_for_node


def test_get_corner_case_for_node_returns_the_case_recorded_for_a_node():
    animal = variable(Animal, domain=[])
    store = CornerCaseStore()
    case = rex()

    store.record_corner_case(animal, case)

    assert store.get_corner_case_for_node(animal._id_) is case


def test_get_corner_case_for_node_returns_none_for_a_node_with_no_recorded_case():
    animal = variable(Animal, domain=[])
    store = CornerCaseStore()

    assert store.get_corner_case_for_node(animal._id_) is None


def test_get_corner_case_for_node_returns_none_for_a_none_node_id():
    store = CornerCaseStore()

    assert store.get_corner_case_for_node(None) is None


# %% CornerCaseStore.to_ordered_sources / from_ordered_cases


def test_to_ordered_sources_only_includes_nodes_with_a_recorded_case():
    with_case = variable(Animal, domain=[])
    without_case = variable(Animal, domain=[])
    store = CornerCaseStore()
    store.record_corner_case(with_case, rex())

    sources = store.to_ordered_sources([with_case, without_case])

    assert set(sources) == {0}
    assert sources[0] == store.serializer.to_source(rex())


def test_from_ordered_cases_rebuilds_a_store_keyed_by_node_id():
    with_case = variable(Animal, domain=[])
    without_case = variable(Animal, domain=[])
    case = rex()

    rebuilt = CornerCaseStore.from_ordered_cases([with_case, without_case], {0: case})

    assert rebuilt.get_corner_case_for_node(with_case._id_) == case
    assert rebuilt.get_corner_case_for_node(without_case._id_) is None
