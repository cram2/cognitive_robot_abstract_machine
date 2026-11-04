"""
Tests pinning how the generative backend constructs an object from a pattern stated for
a collection-valued attribute.

The element built from a nested pattern, or a plain element value, given for an
attribute declared as a collection is wrapped in the field's collection type
(``list[X]`` gives ``[x]``, ``set[X]`` gives ``{x}``, ``tuple[X, ...]`` gives ``(x,)``,
``Optional[set[X]]`` gives ``{x}``). A whole-collection literal stays as given, and a
single-valued attribute is untouched. Each constructed object satisfies the pattern it
was generated from.

The domain classes have names no other module uses: the class diagram resolves a string
annotation such as ``list[CrateItem]`` by class name, so a same-named class imported by
another module could otherwise take its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from krrood.entity_query_language.backends import EntityQueryLanguageGenerativeBackend
from krrood.entity_query_language.factories import a, an, variable_from
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(eq=False)
class CrateItem(Symbol):
    weight: float


@dataclass(eq=False)
class ListCrate(Symbol):
    items: list[CrateItem] = field(default_factory=list)
    label: str = ""


@dataclass(eq=False)
class SetCrate(Symbol):
    items: set[CrateItem] = field(default_factory=set)
    label: str = ""


@dataclass(eq=False)
class TupleCrate(Symbol):
    items: tuple[CrateItem, ...] = ()
    label: str = ""


@dataclass(eq=False)
class OptionalSetCrate(Symbol):
    items: Optional[set[CrateItem]] = None
    label: str = ""


@dataclass(eq=False)
class SingleItemCrate(Symbol):
    item: Optional[CrateItem] = None
    label: str = ""


# %% Helpers


def generate(pattern) -> List[Symbol]:
    """
    :param pattern: A pattern to generate instances from.
    :return: The instances constructed by the generative backend.
    """
    return list(pattern.evaluate(backend=EntityQueryLanguageGenerativeBackend()))


def weights_of(items) -> List[float]:
    """
    :param items: Items, in any collection.
    :return: Their sorted weights.
    """
    return sorted(item.weight for item in items)


def item_pattern(crate_class):
    """
    :param crate_class: A crate class with an ``items`` attribute.
    :return: A pattern generating one crate per weight 1.0 and 2.0 around a built item,
        labelled "x".
    """
    return a(crate_class)(
        items=a(CrateItem)(weight=variable_from([1.0, 2.0])), label="x"
    )


# %% Pattern for an element is wrapped in the collection type


def test_pattern_for_a_list_attribute_constructs_a_list():
    """
    The item built for a ``list[CrateItem]`` attribute is held in a list.
    """
    crates = generate(item_pattern(ListCrate))
    assert all(type(crate.items) is list for crate in crates)


def test_pattern_for_a_list_attribute_constructs_one_element_per_solution():
    """
    Each generated list holds exactly the one item built for that solution.
    """
    crates = generate(item_pattern(ListCrate))
    assert sorted(weights_of(crate.items) for crate in crates) == [[1.0], [2.0]]


def test_pattern_for_a_set_attribute_constructs_a_set():
    """
    The item built for a ``set[CrateItem]`` attribute is held in a set.
    """
    crates = generate(item_pattern(SetCrate))
    assert all(type(crate.items) is set for crate in crates)


def test_pattern_for_a_set_attribute_constructs_one_element_per_solution():
    """
    Each generated set holds exactly the one item built for that solution.
    """
    crates = generate(item_pattern(SetCrate))
    assert sorted(weights_of(crate.items) for crate in crates) == [[1.0], [2.0]]


def test_pattern_for_a_tuple_attribute_constructs_a_tuple():
    """
    The item built for a ``tuple[CrateItem, ...]`` attribute is held in a tuple.
    """
    crates = generate(item_pattern(TupleCrate))
    assert all(type(crate.items) is tuple for crate in crates)


def test_pattern_for_a_tuple_attribute_constructs_one_element_per_solution():
    """
    Each generated tuple holds exactly the one item built for that solution.
    """
    crates = generate(item_pattern(TupleCrate))
    assert sorted(weights_of(crate.items) for crate in crates) == [[1.0], [2.0]]


def test_pattern_for_an_optional_set_attribute_constructs_a_set():
    """
    The item built for an ``Optional[set[CrateItem]]`` attribute is held in a set.
    """
    crates = generate(item_pattern(OptionalSetCrate))
    assert all(type(crate.items) is set for crate in crates)


def test_pattern_for_an_optional_set_attribute_constructs_one_element_per_solution():
    """
    Each generated optional set holds exactly the one item built for that solution.
    """
    crates = generate(item_pattern(OptionalSetCrate))
    assert sorted(weights_of(crate.items) for crate in crates) == [[1.0], [2.0]]


def test_other_attributes_of_the_pattern_are_still_set():
    """
    Wrapping the element does not disturb the other attributes of the generated object.
    """
    crates = generate(item_pattern(ListCrate))
    assert [crate.label for crate in crates] == ["x", "x"]


# %% Plain element value is wrapped in the collection type


def test_element_value_for_a_list_attribute_is_wrapped_in_a_list():
    """
    A plain item given for a ``list[CrateItem]`` attribute, combined with a generated
    attribute, is held in a list.
    """
    existing = CrateItem(5.0)
    pattern = a(ListCrate)(items=existing, label=variable_from(["x", "y"]))
    crates = generate(pattern)
    assert [crate.items for crate in crates] == [[existing], [existing]]


def test_element_value_for_a_set_attribute_is_wrapped_in_a_set():
    """
    A plain item given for a ``set[CrateItem]`` attribute, combined with a generated
    attribute, is held in a set.
    """
    existing = CrateItem(5.0)
    pattern = a(SetCrate)(items=existing, label=variable_from(["x", "y"]))
    crates = generate(pattern)
    assert [crate.items for crate in crates] == [{existing}, {existing}]


def test_element_value_for_a_tuple_attribute_is_wrapped_in_a_tuple():
    """
    A plain item given for a ``tuple[CrateItem, ...]`` attribute, combined with a
    generated attribute, is held in a tuple.
    """
    existing = CrateItem(5.0)
    pattern = a(TupleCrate)(items=existing, label=variable_from(["x", "y"]))
    crates = generate(pattern)
    assert [crate.items for crate in crates] == [(existing,), (existing,)]


def test_element_value_for_an_optional_set_attribute_is_wrapped_in_a_set():
    """
    A plain item given for an ``Optional[set[CrateItem]]`` attribute, combined with a
    generated attribute, is held in a set.
    """
    existing = CrateItem(5.0)
    pattern = an(OptionalSetCrate)(items=existing, label=variable_from(["x", "y"]))
    crates = generate(pattern)
    assert [crate.items for crate in crates] == [{existing}, {existing}]


# %% Unchanged construction


def test_whole_list_literal_stays_as_given():
    """
    A list literal given for a ``list[CrateItem]`` attribute is the constructed list,
    not nested in another list.
    """
    first, second = CrateItem(1.0), CrateItem(2.0)
    crates = generate(a(ListCrate)(items=[first, second], label="x"))
    assert [crate.items for crate in crates] == [[first, second]]


def test_whole_set_literal_stays_as_given():
    """
    A set literal given for a ``set[CrateItem]`` attribute is the constructed set.
    """
    first, second = CrateItem(1.0), CrateItem(2.0)
    crates = generate(a(SetCrate)(items={first, second}, label="x"))
    assert [crate.items for crate in crates] == [{first, second}]


def test_single_valued_attribute_receives_the_built_element_itself():
    """
    A pattern for a single-valued attribute constructs the bare item, not a collection.
    """
    pattern = a(SingleItemCrate)(
        item=a(CrateItem)(weight=variable_from([1.0, 2.0])), label="x"
    )
    crates = generate(pattern)
    assert all(type(crate.item) is CrateItem for crate in crates)


# %% Constructed objects satisfy their own pattern


def test_constructed_list_crate_satisfies_the_pattern_it_came_from():
    """
    Evaluating the generating pattern over a constructed list crate matches it.
    """
    crates = generate(item_pattern(ListCrate))
    for crate in crates:
        assert len(list(item_pattern(ListCrate).from_([crate]).evaluate())) == 1


def test_constructed_set_crate_satisfies_the_pattern_it_came_from():
    """
    Evaluating the generating pattern over a constructed set crate matches it.
    """
    crates = generate(item_pattern(SetCrate))
    for crate in crates:
        assert len(list(item_pattern(SetCrate).from_([crate]).evaluate())) == 1


def test_constructed_tuple_crate_satisfies_the_pattern_it_came_from():
    """
    Evaluating the generating pattern over a constructed tuple crate matches it.
    """
    crates = generate(item_pattern(TupleCrate))
    for crate in crates:
        assert len(list(item_pattern(TupleCrate).from_([crate]).evaluate())) == 1


def test_constructed_optional_set_crate_satisfies_the_pattern_it_came_from():
    """
    Evaluating the generating pattern over a constructed optional set crate matches it.
    """
    crates = generate(item_pattern(OptionalSetCrate))
    for crate in crates:
        assert len(list(item_pattern(OptionalSetCrate).from_([crate]).evaluate())) == 1
