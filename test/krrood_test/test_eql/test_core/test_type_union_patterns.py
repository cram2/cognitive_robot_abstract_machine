"""
Tests for patterns over a union of classes, such as ``a(Fruit | Vegetable)``, which
match an object of any one of the classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.backends import EntityQueryLanguageGenerativeBackend
from krrood.entity_query_language.factories import a, every_is, none_is

from ...dataset.collection_attribute_paths import (
    Colour,
    Fruit,
    Product,
    Shelf,
    Vegetable,
)

# %% fixtures


@dataclass
class Pantry:
    """
    Shelves of fruits, vegetables and other products, one of them empty.
    """

    apple: Fruit
    """
    A red fruit.
    """

    pea: Vegetable
    """
    A green vegetable.
    """

    salt: Product
    """
    A product that is neither a fruit nor a vegetable.
    """

    empty_shelf: Shelf
    """
    A shelf without products.
    """

    produce_shelf: Shelf
    """
    A shelf holding only a fruit and a vegetable.
    """

    mixed_shelf: Shelf
    """
    A shelf holding salt and a fruit.
    """

    salt_shelf: Shelf
    """
    A shelf holding only salt.
    """

    @property
    def shelves(self) -> list[Shelf]:
        """
        :return: Every shelf of the pantry.
        """
        return [
            self.empty_shelf,
            self.produce_shelf,
            self.mixed_shelf,
            self.salt_shelf,
        ]


@pytest.fixture
def pantry() -> Pantry:
    """
    :return: Shelves of fruits, vegetables and salt, one of them empty.
    """
    apple = Fruit("apple", Colour.RED, 0.2)
    pea = Vegetable("pea", Colour.GREEN, 0.01)
    salt = Product("salt", Colour.GREEN, 1.0)
    return Pantry(
        apple=apple,
        pea=pea,
        salt=salt,
        empty_shelf=Shelf("empty"),
        produce_shelf=Shelf("produce", [apple, pea]),
        mixed_shelf=Shelf("mixed", [salt, apple]),
        salt_shelf=Shelf("salt", [salt]),
    )


# %% matching a union


def test_union_pattern_matches_an_object_of_any_class_of_the_union(pantry):
    """
    A pattern over a union selects the objects of each of its classes.
    """
    products = a(Fruit | Vegetable).from_(domain=[pantry.apple, pantry.pea, pantry.salt])

    assert products.tolist() == [pantry.apple, pantry.pea]


def test_some_element_of_a_union(pantry):
    """
    Some product of a shelf is a fruit or a vegetable.
    """
    shelves = a(Shelf)(products=a(Fruit | Vegetable)).from_(domain=pantry.shelves)

    assert set(shelves.tolist()) == {pantry.produce_shelf, pantry.mixed_shelf}


def test_every_element_of_a_union(pantry):
    """
    Every product of a shelf is a fruit or a vegetable, vacuously so for the empty
    shelf.
    """
    shelves = a(Shelf)(products=every_is(Fruit | Vegetable)).from_(domain=pantry.shelves)

    assert shelves.tolist() == [pantry.empty_shelf, pantry.produce_shelf]


def test_no_element_of_a_union(pantry):
    """
    No product of a shelf is a fruit or a vegetable, vacuously so for the empty shelf.
    """
    shelves = a(Shelf)(products=none_is(Fruit | Vegetable)).from_(domain=pantry.shelves)

    assert shelves.tolist() == [pantry.empty_shelf, pantry.salt_shelf]


# %% keyword arguments


def test_union_pattern_takes_a_field_every_class_has(pantry):
    """
    A field of the common base class constrains the objects of every class of the
    union: no shelf holds a green fruit or vegetable, except the produce shelf.
    """
    shelves = a(Shelf)(products=none_is(Fruit | Vegetable)(colour=Colour.GREEN)).from_(
        domain=pantry.shelves
    )

    assert shelves.tolist() == [
        pantry.empty_shelf,
        pantry.mixed_shelf,
        pantry.salt_shelf,
    ]


def test_union_pattern_on_a_field_of_one_class_matches_like_its_base_class(pantry):
    """
    A field that only one class of the union has is matched as by a pattern on the
    common base class: only objects having the field can match it.
    """
    pantry.pea.grows_underground = True
    products = [pantry.apple, pantry.pea, pantry.salt]

    union = a(Fruit | Vegetable)(grows_underground=True).from_(domain=products)
    base = a(Product)(grows_underground=True).from_(domain=products)

    assert union.tolist() == base.tolist() == [pantry.pea]


def test_condition_on_no_element_of_a_union(pantry):
    """
    A condition given to a union pattern with :meth:`where` holds together with the
    union: no fruit or vegetable on the shelf is heavy, though the salt may be.
    """
    heavy = none_is(Fruit | Vegetable)
    heavy.where(heavy.weight > 0.1)

    shelves = a(Shelf)(products=heavy).from_(domain=pantry.shelves)

    assert shelves.tolist() == [pantry.empty_shelf, pantry.salt_shelf]


# %% generation


def test_generation_yields_no_object_outside_the_union():
    """
    Generation constructs the common base class, so it yields nothing for a union
    pattern rather than an object of a class outside the union.
    """
    produce = a(Fruit | Vegetable)(name="apple", colour=Colour.RED, weight=0.2)

    assert list(produce.evaluate(backend=EntityQueryLanguageGenerativeBackend())) == []
