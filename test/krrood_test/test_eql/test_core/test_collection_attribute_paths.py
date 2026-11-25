"""
Tests for reading an attribute of the elements of a collection through a path, such as
``shelf.products.colour``, without naming the element with ``variable_from``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    CollectionElement,
)
from krrood.entity_query_language.exceptions import AmbiguousCollectionAttribute
from krrood.entity_query_language.factories import (
    a,
    an,
    count,
    entity,
    exists,
    for_all,
    not_,
    or_,
    set_of,
    variable,
    variable_from,
)

from ...dataset.collection_attribute_paths import (
    Colour,
    Product,
    Shelf,
    Tally,
    TallyMark,
    Warehouse,
)

# %% fixtures


@dataclass
class Store:
    """
    Shelves whose products cover every case a path through a collection meets.
    """

    apple: Product
    """
    A red product.
    """

    cherry: Product
    """
    Another red product.
    """

    lime: Product
    """
    A green product.
    """

    empty_shelf: Shelf
    """
    A shelf without products.
    """

    red_shelf: Shelf
    """
    A shelf holding only red products, more than one.
    """

    mixed_shelf: Shelf
    """
    A shelf holding a green and a red product.
    """

    green_shelf: Shelf
    """
    A shelf holding only a green product.
    """

    @property
    def shelves(self) -> list[Shelf]:
        """
        Every shelf of the store.
        """
        return [self.empty_shelf, self.red_shelf, self.mixed_shelf, self.green_shelf]


@pytest.fixture
def store() -> Store:
    """
    Shelves of red and green products, one of them empty.
    """
    apple = Product("apple", Colour.RED, 0.2)
    cherry = Product("cherry", Colour.RED, 0.01)
    lime = Product("lime", Colour.GREEN, 0.1)
    return Store(
        apple=apple,
        cherry=cherry,
        lime=lime,
        empty_shelf=Shelf("empty"),
        red_shelf=Shelf("red", [apple, cherry]),
        mixed_shelf=Shelf("mixed", [lime, apple]),
        green_shelf=Shelf("green", [lime]),
    )


# %% building the path


def test_attribute_only_elements_have_is_read_on_the_element(store):
    """
    An attribute the list lacks is read on the collection element.
    """
    shelf = variable(Shelf, store.shelves)
    colour = shelf.products.colour
    assert isinstance(colour, Attribute)
    assert isinstance(colour._child_, CollectionElement)


def test_same_path_gives_same_element(store):
    """
    Two attributes read through one path share its element node.
    """
    shelf = variable(Shelf, store.shelves)
    assert shelf.products.colour._child_ is shelf.products.name._child_


def test_attribute_of_collection_class_stays_on_collection(store):
    """
    An attribute of the list class is read on the list itself.
    """
    shelf = variable(Shelf, store.shelves)
    copy_method = shelf.products.copy
    assert copy_method._child_ is shelf.products


def test_index_reads_collection(store):
    """
    Indexing selects one element of the collection, not each element.
    """
    shelf = variable(Shelf, store.shelves)
    first_colour = an(entity(shelf).where(shelf.products[0].colour == Colour.GREEN))
    assert first_colour.tolist() == [store.mixed_shelf, store.green_shelf]


def test_name_of_collection_class_and_element_type_is_ambiguous():
    """
    A name both the list and its elements have cannot be resolved.
    """
    tally = variable(Tally, [Tally([TallyMark(3)])])
    with pytest.raises(AmbiguousCollectionAttribute):
        _ = tally.marks.count


def test_mapping_attribute_does_not_read_elements(store):
    """
    A dictionary field is not read element by element.
    """
    shelf = variable(Shelf, store.shelves)
    assert shelf.weight_by_product.keys._child_ is shelf.weight_by_product


# %% filtering on elements


def test_condition_on_element_keeps_each_owner_once(store):
    """
    A shelf with two red products is reported once.
    """
    shelf = variable(Shelf, store.shelves)
    red = an(entity(shelf).where(shelf.products.colour == Colour.RED))
    assert red.tolist() == [store.red_shelf, store.mixed_shelf]


def test_conditions_on_same_path_constrain_same_element(store):
    """
    Two conditions on one path must hold for one product.
    """
    shelf = variable(Shelf, store.shelves)
    green_apple = an(
        entity(shelf).where(
            shelf.products.colour == Colour.GREEN, shelf.products.name == "apple"
        )
    )
    assert green_apple.tolist() == []


def test_variable_over_the_collection_names_another_element(store):
    """
    A variable over the products ranges over them independently of the path.
    """
    shelf = variable(Shelf, store.shelves)
    other_product = variable_from(shelf.products)
    green_and_apple = an(
        entity(shelf).where(
            shelf.products.colour == Colour.GREEN, other_product.name == "apple"
        )
    )
    assert green_and_apple.tolist() == [store.mixed_shelf]


def test_match_root_reads_elements(store):
    """
    A path rooted at a match reads the elements as well.
    """
    shelf = a(Shelf).from_(domain=store.shelves)
    green = an(entity(shelf).where(shelf.products.colour == Colour.GREEN))
    assert green.tolist() == [store.mixed_shelf, store.green_shelf]


def test_path_through_nested_collections(store):
    """
    A path through two collections reads the inner elements.
    """
    warehouses = [
        Warehouse("empty", [store.empty_shelf]),
        Warehouse("green", [store.green_shelf, store.empty_shelf]),
    ]
    warehouse = variable(Warehouse, warehouses)
    green = an(
        entity(warehouse).where(warehouse.shelves.products.colour == Colour.GREEN)
    )
    assert green.tolist() == [warehouses[1]]


# %% empty collections


def test_empty_collection_keeps_owner_in_disjunction(store):
    """
    An empty shelf passes a disjunction through its other operand.
    """
    shelf = variable(Shelf, store.shelves)
    red_or_named_empty = an(
        entity(shelf).where(
            or_(shelf.products.colour == Colour.RED, shelf.name == "empty")
        )
    )
    assert red_or_named_empty.tolist() == [
        store.empty_shelf,
        store.red_shelf,
        store.mixed_shelf,
    ]


def test_negated_condition_holds_on_empty_collection(store):
    """
    The negation of a condition on an absent product holds.
    """
    shelf = variable(Shelf, store.shelves)
    not_red = an(entity(shelf).where(not_(shelf.products.colour == Colour.RED)))
    assert not_red.tolist() == [store.empty_shelf, store.mixed_shelf, store.green_shelf]


def test_inequality_agrees_with_negated_equality(store):
    """
    Inequality and negated equality give the same shelves.
    """
    shelf = variable(Shelf, store.shelves)
    not_red = an(entity(shelf).where(not_(shelf.products.colour == Colour.RED)))
    other_shelf = variable(Shelf, store.shelves)
    unequal_to_red = an(
        entity(other_shelf).where(other_shelf.products.colour != Colour.RED)
    )
    assert unequal_to_red.tolist() == not_red.tolist()


def test_arithmetic_on_absent_element_satisfies_no_comparison(store):
    """
    Arithmetic on an absent product compares false.
    """
    shelf = variable(Shelf, store.shelves)
    light = an(entity(shelf).where(shelf.products.weight * 1000 < 50))
    assert light.tolist() == [store.red_shelf]


def test_absent_optional_collection_has_no_element(store):
    """
    An optional collection left as None has an absent element.
    """
    store.green_shelf.spare_products = [store.cherry]
    shelf = variable(Shelf, store.shelves)
    spare_red = an(entity(shelf).where(shelf.spare_products.colour == Colour.RED))
    assert spare_red.tolist() == [store.green_shelf]


def test_universal_quantifier_over_empty_collection_holds(store):
    """
    Every product of an empty shelf is green, vacuously.
    """
    shelf = variable(Shelf, store.shelves)
    product = variable_from(shelf.products)
    all_green = an(
        entity(shelf).where(for_all(product, product.colour == Colour.GREEN))
    )
    assert all_green.tolist() == [store.empty_shelf, store.green_shelf]


def test_negated_existential_quantifier_over_empty_collection_holds(store):
    """
    An empty shelf has no red product.
    """
    shelf = variable(Shelf, store.shelves)
    product = variable_from(shelf.products)
    no_red = an(
        entity(shelf).where(not_(exists(product, product.colour == Colour.RED)))
    )
    assert no_red.tolist() == [store.empty_shelf, store.green_shelf]


# %% paths inside quantifiers


@pytest.fixture
def warehouses(store) -> list[Warehouse]:
    """
    Warehouses whose shelves cover every case of a quantifier over shelves.
    """
    return [
        Warehouse("every shelf has red", [store.red_shelf, store.mixed_shelf]),
        Warehouse("one shelf without red", [store.red_shelf, store.green_shelf]),
        Warehouse("no shelves"),
        Warehouse("empty shelf", [store.empty_shelf]),
    ]


def test_universal_quantifier_body_reads_some_element_per_value(warehouses):
    """
    A path in a universal body asks for some product of each shelf.
    """
    warehouse = variable(Warehouse, warehouses)
    shelf = variable_from(warehouse.shelves)
    every_shelf_has_red = an(
        entity(warehouse).where(for_all(shelf, shelf.products.colour == Colour.RED))
    )
    assert every_shelf_has_red.tolist() == [warehouses[0], warehouses[2]]


def test_existential_quantifier_body_reads_some_element(warehouses):
    """
    A path in an existential body asks for some product of some shelf.
    """
    warehouse = variable(Warehouse, warehouses)
    shelf = variable_from(warehouse.shelves)
    some_shelf_has_green = an(
        entity(warehouse).where(exists(shelf, shelf.products.colour == Colour.GREEN))
    )
    assert some_shelf_has_green.tolist() == [warehouses[0], warehouses[1]]


# %% selecting elements


def test_selected_path_gives_each_satisfying_element(store):
    """
    Selecting the path gives each product that satisfies the condition.
    """
    shelf = variable(Shelf, store.shelves)
    red_names = an(
        entity(shelf.products.name).where(shelf.products.colour == Colour.RED)
    )
    assert red_names.tolist() == [store.apple.name, store.cherry.name, store.apple.name]


def test_selected_path_alone_skips_empty_collection(store):
    """
    Selecting the path alone reports nothing for an empty shelf.
    """
    shelf = variable(Shelf, store.shelves)
    names = an(entity(shelf.products.name))
    assert names.tolist() == [
        store.apple.name,
        store.cherry.name,
        store.lime.name,
        store.apple.name,
        store.lime.name,
    ]


def test_selected_path_next_to_owner_is_none_for_empty_collection(store):
    """
    Selecting the path next to its shelf reports None for an empty shelf.
    """
    shelf = variable(Shelf, store.shelves)
    product_name = shelf.products.name
    rows = an(set_of(shelf, product_name).where(shelf.name == "empty")).tolist()
    assert [(row[shelf], row[product_name]) for row in rows] == [
        (store.empty_shelf, None)
    ]


def test_count_of_elements_is_zero_for_empty_collection(store):
    """
    Counting the products of an empty shelf gives zero.
    """
    shelf = variable(Shelf, store.shelves)
    product_count = count(shelf.products.name)
    rows = an(set_of(shelf, product_count).grouped_by(shelf)).tolist()
    assert {row[shelf].name: row[product_count] for row in rows} == {
        shelf_.name: len(shelf_.products) for shelf_ in store.shelves
    }


# %% domains read through a path


def test_domain_read_through_a_path_skips_the_absent_element(warehouses):
    """
    A warehouse without shelves gives a variable over its shelves' products no value.
    """
    warehouse = variable(Warehouse, warehouses)
    product = variable_from(warehouse.shelves.products)

    rows = set_of(warehouse, product).tolist()

    assert [(row[warehouse], row[product]) for row in rows] == [
        (warehouse_, product_)
        for warehouse_ in warehouses
        for shelf_ in warehouse_.shelves
        for product_ in shelf_.products
    ]
