"""
Tests pinning the semantics of ``match`` on *optional* collection-valued attributes
(``Optional[set[X]]``, ``set[X] | None``, ``Optional[list[X]]``).

An optional collection behaves like a collection with ``None`` standing for the empty
collection:

* ``attr=value`` means ``value`` is an element, so an owner whose attribute is ``None``
  never matches.
* ``attr=pattern`` means some element matches the pattern, so an owner whose attribute is
  ``None`` never matches, and an unconstrained pattern excludes it without raising.
* A symbolic value whose type is the element type means membership.
* ``attr=None`` and a whole-collection literal keep equality.
* Flattening an optional collection inside a plain query treats ``None`` as empty.

The domain classes have names no other module uses: the class diagram resolves a string
annotation such as ``set[ShelfItem]`` by class name, so a same-named class imported by
another module could otherwise take its place.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterable, List, Optional, Type

import pytest

from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    exists,
    flat_variable,
    for_all,
    variable,
)
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(eq=False)
class ShelfItem(Symbol):
    weight: float


@dataclass(eq=False)
class OptionalSetBag(Symbol):
    name: str
    items: Optional[set[ShelfItem]] = None


@dataclass(eq=False)
class UnionSetBag(Symbol):
    name: str
    items: set[ShelfItem] | None = None


@dataclass(eq=False)
class OptionalListBag(Symbol):
    name: str
    items: Optional[list[ShelfItem]] = None


BAG_CLASSES_AND_CONTAINERS = [
    pytest.param(OptionalSetBag, set, id="Optional[set]"),
    pytest.param(UnionSetBag, set, id="set-or-None"),
    pytest.param(OptionalListBag, list, id="Optional[list]"),
]


def names_of(bags: Iterable[Symbol]) -> List[str]:
    """
    :param bags: The matched bags.
    :return: Their sorted names.
    """
    return sorted(bag.name for bag in bags)


# %% Builders


def build_bags(bag_class: Type[Symbol], container_type: type):
    """
    :param bag_class: The bag class to instantiate.
    :param container_type: ``set`` or ``list``, the collection type of the bag's items.
    :return: ``(light, heavy, bags)`` where the bags are "both" holding the light (1.0)
        and heavy (2.0) item, "light" holding only the light item, and "none" whose
        items are ``None``.
    """
    light, heavy = ShelfItem(1.0), ShelfItem(2.0)
    bags = [
        bag_class("both", container_type([light, heavy])),
        bag_class("light", container_type([light])),
        bag_class("none", None),
    ]
    return light, heavy, bags


# %% Element semantics on an optional collection


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_value_matches_owners_whose_optional_collection_contains_it(
    bag_class, container_type
):
    """
    ``items=heavy`` matches the bags holding the heavy item and does not raise on the
    bag whose collection is ``None``.
    """
    _, heavy, bags = build_bags(bag_class, container_type)
    matches = a(bag_class)(items=heavy).from_(bags).evaluate()
    assert names_of(matches) == ["both"]


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_value_in_no_optional_collection_matches_nothing(bag_class, container_type):
    """
    An item no bag holds matches no bag, including the one whose collection is ``None``.
    """
    _, _, bags = build_bags(bag_class, container_type)
    matches = a(bag_class)(items=ShelfItem(9.0)).from_(bags).evaluate()
    assert list(matches) == []


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_nested_pattern_matches_when_some_element_of_an_optional_collection_matches(
    bag_class, container_type
):
    """
    ``items=a(ShelfItem)(weight=2.0)`` matches the bags with a heavy item and does not
    raise while building or evaluating the pattern.
    """
    _, _, bags = build_bags(bag_class, container_type)
    pattern = a(bag_class)(items=a(ShelfItem)(weight=2.0))
    assert names_of(pattern.from_(bags).evaluate()) == ["both"]


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_unconstrained_pattern_excludes_the_owner_with_a_none_collection(
    bag_class, container_type
):
    """
    ``items=a(ShelfItem)()`` matches every bag with at least one item, once each, and
    excludes the bag whose collection is ``None`` instead of raising.
    """
    _, _, bags = build_bags(bag_class, container_type)
    pattern = a(bag_class)(items=a(ShelfItem)())
    assert names_of(pattern.from_(bags).evaluate()) == ["both", "light"]


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_variable_of_the_element_type_means_membership_in_an_optional_collection(
    bag_class, container_type
):
    """
    A symbolic value whose type is the element type stands for one element, so it must
    be in the optional collection; the bag with ``None`` never matches.
    """
    light, _, bags = build_bags(bag_class, container_type)
    element = variable(ShelfItem, domain=[light])
    matches = a(bag_class)(items=element).from_(bags).evaluate()
    assert names_of(matches) == ["both", "light"]


# %% Regression guards: unchanged semantics


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_none_keeps_equality_with_none(bag_class, container_type):
    """
    ``items=None`` matches only the bag whose collection is ``None``.
    """
    _, _, bags = build_bags(bag_class, container_type)
    matches = a(bag_class)(items=None).from_(bags).evaluate()
    assert names_of(matches) == ["none"]


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_whole_collection_literal_keeps_equality(bag_class, container_type):
    """
    A collection literal equal to the whole collection of one bag matches only that bag.
    """
    light, _, bags = build_bags(bag_class, container_type)
    matches = a(bag_class)(items=container_type([light])).from_(bags).evaluate()
    assert names_of(matches) == ["light"]


# %% Flattening an optional collection inside a plain query


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_exists_over_a_flattened_none_collection_is_false(bag_class, container_type):
    """
    ``exists`` over the flattened items of a ``None`` collection is false, so that bag
    is left out and nothing is raised.
    """
    _, _, bags = build_bags(bag_class, container_type)
    bag = variable(bag_class, domain=bags)
    query = an(
        entity(bag).where(
            exists(item := flat_variable(bag.items), item.weight > 0.0),
        )
    )
    assert names_of(query.evaluate()) == ["both", "light"]


@pytest.mark.parametrize("bag_class, container_type", BAG_CLASSES_AND_CONTAINERS)
def test_for_all_over_a_flattened_none_collection_is_vacuously_true(
    bag_class, container_type
):
    """
    ``for_all`` over the flattened items of a ``None`` collection holds vacuously, so
    the bag with ``None`` is kept when every element satisfies the condition trivially.
    """
    _, _, bags = build_bags(bag_class, container_type)
    bag = variable(bag_class, domain=bags)
    query = an(
        entity(bag).where(
            for_all(item := flat_variable(bag.items), item.weight > 0.0),
        )
    )
    assert names_of(query.evaluate()) == ["both", "light", "none"]
