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


def test_a_copied_property_of_an_ontology_shares_only_the_ontology():
    """
    A deep copy of a property the ontology holds keeps the ontology but none of the
    property's own lists.
    """
    original = PropertyInfo(
        name="knows",
        uri="http://example.org/person#knows",
        type=PropertyType.OBJECT_PROPERTY,
        superproperties=["relatedTo"],
    )
    ontology = OntologyInfo(
        graph=rdflib.Graph(), original_properties={original.name: original}
    )
    copy = ontology.properties[original.name]
    assert (
        copy.ontology is ontology,
        copy.superproperties == original.superproperties,
        copy.superproperties is original.superproperties,
    ) == (True, True, False)


# %% identity


def test_a_class_is_equal_only_to_itself():
    """
    Two classes with the same information are two distinct classes, as they are hashed
    by their identity.
    """
    assert ClassInfo(name="Person", uri="http://example.org/person#Person") != (
        ClassInfo(name="Person", uri="http://example.org/person#Person")
    )


def test_a_property_is_equal_only_to_itself():
    """
    Two properties with the same information are two distinct properties, as they are
    hashed by their identity.
    """
    assert PropertyInfo(
        name="knows",
        uri="http://example.org/person#knows",
        type=PropertyType.OBJECT_PROPERTY,
    ) != PropertyInfo(
        name="knows",
        uri="http://example.org/person#knows",
        type=PropertyType.OBJECT_PROPERTY,
    )


def test_an_ontology_is_equal_only_to_itself():
    """
    Two ontologies of the same graph are two distinct ontologies, as they are hashed by
    their identity.
    """
    graph = rdflib.Graph()
    assert OntologyInfo(graph=graph) != OntologyInfo(graph=graph)
