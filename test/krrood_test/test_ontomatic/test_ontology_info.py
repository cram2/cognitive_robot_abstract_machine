"""
Ordering the classes and properties of an ontology by what they depend on.
"""

from __future__ import annotations

from copy import deepcopy

import rdflib

from krrood.ontomatic.ontology_to_python.ontology_info import (
    ClassInfo,
    OntologyInfo,
    PropertyInfo,
)
from krrood.ontomatic.ontology_to_python.property_type import PropertyType

# %% topological order


def test_an_item_comes_after_the_items_it_depends_on():
    """
    Every item follows the items it depends on.
    """
    items = {"a": {"bases": ["b"]}, "b": {"bases": []}, "c": {"bases": ["a"]}}
    assert OntologyInfo.topological_order(items=items, dependency_key="bases") == [
        "b",
        "a",
        "c",
    ]


def test_items_in_a_cycle_follow_the_rest_in_alphabetical_order():
    """
    Items that depend on each other in a cycle come last, in alphabetical order.
    """
    items = {"x": {"bases": ["y"]}, "y": {"bases": ["x"]}, "z": {"bases": []}}
    assert OntologyInfo.topological_order(items=items, dependency_key="bases") == [
        "z",
        "x",
        "y",
    ]


def test_a_dependency_outside_the_items_is_ignored():
    """
    A dependency on a name that is not among the items does not hold an item back.
    """
    items = {"a": {"bases": ["Symbol"]}}
    assert OntologyInfo.topological_order(items=items, dependency_key="bases") == ["a"]


# %% copies


def test_a_copied_class_shares_its_ontology():
    """
    A deep copy of a class copies its own information but keeps the ontology it belongs
    to.
    """
    ontology = OntologyInfo(graph=rdflib.Graph())
    original = ClassInfo(
        name="Person", uri="http://example.org/person#Person", ontology=ontology
    )
    copy = deepcopy(original)
    assert (copy.ontology is ontology, copy is not original) == (True, True)


def test_a_copied_property_shares_its_ontology():
    """
    A deep copy of a property copies its own information but keeps the ontology it
    belongs to.
    """
    ontology = OntologyInfo(graph=rdflib.Graph())
    original = PropertyInfo(
        name="knows",
        uri="http://example.org/person#knows",
        type=PropertyType.OBJECT_PROPERTY,
        ontology=ontology,
    )
    copy = deepcopy(original)
    assert (copy.ontology is ontology, copy is not original) == (True, True)
