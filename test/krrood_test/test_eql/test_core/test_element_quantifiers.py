"""
Tests for the patterns :func:`every_is` and :func:`none_is`, which state how many elements of a
collection attribute match a pattern, as :func:`a` states that some element does.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.exceptions import ElementQuantifierWithoutAttribute
from krrood.entity_query_language.factories import a, every_is, none_is

from ...dataset.collection_attribute_paths import Colour, Fruit, Product, Shelf
from ...dataset.journeys import Journey, Route, Stop

# %% fixtures


@dataclass
class Store:
    """
    Shelves of fruits and other products, one of them empty.
    """

    empty_shelf: Shelf
    """
    A shelf without products.
    """

    red_fruit_shelf: Shelf
    """
    A shelf holding only red fruits.
    """

    mixed_shelf: Shelf
    """
    A shelf holding a red fruit and a green product that is no fruit.
    """

    green_shelf: Shelf
    """
    A shelf holding only a green product that is no fruit.
    """

    @property
    def shelves(self) -> list[Shelf]:
        """
        :return: Every shelf of the store.
        """
        return [
            self.empty_shelf,
            self.red_fruit_shelf,
            self.mixed_shelf,
            self.green_shelf,
        ]


@pytest.fixture
def store() -> Store:
    """
    :return: Shelves of red fruits and green products, one of them empty.
    """
    apple = Fruit("apple", Colour.RED, 0.2)
    cherry = Fruit("cherry", Colour.RED, 0.01)
    lime = Product("lime", Colour.GREEN, 0.1)
    return Store(
        empty_shelf=Shelf("empty"),
        red_fruit_shelf=Shelf("red fruit", [apple, cherry]),
        mixed_shelf=Shelf("mixed", [lime, apple]),
        green_shelf=Shelf("green", [lime]),
    )


# %% every element


def test_every_element_matches_the_pattern(store):
    """
    Every product of a shelf is red, vacuously so for the empty shelf.
    """
    shelves = a(Shelf)(products=every_is(Product)(colour=Colour.RED)).from_(
        domain=store.shelves
    )

    assert shelves.tolist() == [store.empty_shelf, store.red_fruit_shelf]


def test_every_element_is_of_the_pattern_type(store):
    """
    The type of :func:`every_is` is stated of every element, not used to pick some.
    """
    shelves = a(Shelf)(products=every_is(Fruit)).from_(domain=store.shelves)

    assert shelves.tolist() == [store.empty_shelf, store.red_fruit_shelf]


def test_every_is_on_a_single_value_matches_like_a(store):
    """
    On a field holding one object, :func:`every_is` is the same as :func:`a`.
    """
    harbour = Stop(name="harbour")
    journeys = [Journey(name="by sea", route=Route(stops=[harbour])), Journey(name="overland", route=Route())]

    every_route = a(Journey)(route=every_is(Route)(stops=[harbour])).from_(domain=journeys)
    a_route = a(Journey)(route=a(Route)(stops=[harbour])).from_(domain=journeys)

    assert every_route.tolist() == a_route.tolist() == [journeys[0]]


# %% no element


def test_no_element_matches_the_pattern(store):
    """
    No product of a shelf is red, vacuously so for the empty shelf.
    """
    shelves = a(Shelf)(products=none_is(Product)(colour=Colour.RED)).from_(
        domain=store.shelves
    )

    assert shelves.tolist() == [store.empty_shelf, store.green_shelf]


def test_none_is_on_a_single_value_negates_the_pattern(store):
    """
    On a field holding one object, :func:`none_is` states that the object does not match.
    """
    harbour = Stop(name="harbour")
    journeys = [Journey(name="by sea", route=Route(stops=[harbour])), Journey(name="overland", route=Route())]

    result = a(Journey)(route=none_is(Route)(stops=[harbour])).from_(domain=journeys)

    assert result.tolist() == [journeys[1]]


# %% conditions on the pattern


def test_condition_on_every_element(store):
    """
    A condition given to :func:`every_is` with :meth:`where` holds of every element.
    """
    light = every_is(Product)
    light.where(light.weight < 0.15)

    shelves = a(Shelf)(products=light).from_(domain=store.shelves)

    assert shelves.tolist() == [store.empty_shelf, store.green_shelf]


def test_condition_on_no_element(store):
    """
    A condition given to :func:`none_is` with :meth:`where` holds of no element.
    """
    heavy = none_is(Product)
    heavy.where(heavy.weight > 0.15)

    shelves = a(Shelf)(products=heavy).from_(domain=store.shelves)

    assert shelves.tolist() == [store.empty_shelf, store.green_shelf]


# %% misuse


@pytest.mark.parametrize("quantifier", [every_is, none_is])
def test_element_quantifier_outside_an_attribute_is_rejected(store, quantifier):
    """
    A pattern that is not the value of an attribute has no elements to quantify.
    """
    shelves = quantifier(Shelf).from_(domain=store.shelves)

    with pytest.raises(ElementQuantifierWithoutAttribute):
        shelves.tolist()
