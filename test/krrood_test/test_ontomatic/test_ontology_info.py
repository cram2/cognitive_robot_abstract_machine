"""
Ordering the classes and properties of an ontology by what they depend on.
"""

from __future__ import annotations

from krrood.ontomatic.ontology_to_python.ontology_info import OntologyInfo

# %% topological order


def test_an_item_comes_after_the_items_it_depends_on():
    """
    Every item follows the items it depends on.
    """
    items = {"a": {"bases": ["b"]}, "b": {"bases": []}, "c": {"bases": ["a"]}}
    assert OntologyInfo.topological_order(items=items, dep_key="bases") == [
        "b",
        "a",
        "c",
    ]


def test_items_in_a_cycle_follow_the_rest_in_alphabetical_order():
    """
    Items that depend on each other in a cycle come last, in alphabetical order.
    """
    items = {"x": {"bases": ["y"]}, "y": {"bases": ["x"]}, "z": {"bases": []}}
    assert OntologyInfo.topological_order(items=items, dep_key="bases") == [
        "z",
        "x",
        "y",
    ]


def test_a_dependency_outside_the_items_is_ignored():
    """
    A dependency on a name that is not among the items does not hold an item back.
    """
    items = {"a": {"bases": ["Symbol"]}}
    assert OntologyInfo.topological_order(items=items, dep_key="bases") == ["a"]
