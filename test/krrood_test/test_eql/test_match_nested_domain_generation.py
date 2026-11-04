"""
Tests pinning how the deterministic generative backend handles a nested pattern ranging
over a domain: the generated object takes an object of that domain satisfying the nested
pattern, instead of a newly constructed one.

The domain classes have names no other module uses: the class diagram resolves a string
annotation such as ``set[ShelfBook]`` by class name, so a same-named class imported by
another module could otherwise take its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from krrood.entity_query_language.backends import EntityQueryLanguageGenerativeBackend
from krrood.entity_query_language.factories import a, variable_from
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(eq=False)
class ShelfBook(Symbol):
    weight: float


@dataclass(eq=False)
class BookStand(Symbol):
    book: ShelfBook | None = None
    label: str = ""


@dataclass(eq=False)
class BookBox(Symbol):
    books: set[ShelfBook] = field(default_factory=set)
    label: str = ""


@dataclass(eq=False)
class BookRow(Symbol):
    books: list[ShelfBook] = field(default_factory=list)
    label: str = ""


# %% Helpers


def generate(pattern) -> List[Symbol]:
    """
    :param pattern: A pattern to generate instances from.
    :return: The instances constructed by the deterministic generative backend.
    """
    return list(pattern.evaluate(backend=EntityQueryLanguageGenerativeBackend()))


def light_and_heavy_books() -> tuple[ShelfBook, ShelfBook]:
    """
    :return: A book of weight 1.0 and one of weight 5.0.
    """
    return ShelfBook(1.0), ShelfBook(5.0)


# %% The generated object takes an object of the nested domain


def test_single_valued_attribute_takes_each_object_of_the_domain():
    """
    One stand is generated per book of the domain, holding that very book.
    """
    light, heavy = light_and_heavy_books()
    pattern = a(BookStand)(book=a(ShelfBook)().from_([light, heavy]), label="x")
    held = [stand.book for stand in generate(pattern)]
    assert sorted(held, key=lambda book: book.weight) == [light, heavy]


def test_nested_conditions_select_the_objects_of_the_domain():
    """
    Only the books of the domain satisfying the nested pattern are taken.
    """
    light, heavy = light_and_heavy_books()
    pattern = a(BookStand)(book=a(ShelfBook)(weight=5.0).from_([light, heavy]))
    assert [stand.book for stand in generate(pattern)] == [heavy]


def test_no_object_of_the_domain_satisfying_the_pattern_generates_nothing():
    """
    When no book of the domain satisfies the nested pattern, nothing is generated.
    """
    light, _ = light_and_heavy_books()
    pattern = a(BookStand)(book=a(ShelfBook)(weight=5.0).from_([light]))
    assert generate(pattern) == []


def test_collection_attribute_holds_the_object_of_the_domain():
    """
    The book taken for a ``set[ShelfBook]`` attribute is held in a set.
    """
    _, heavy = light_and_heavy_books()
    pattern = a(BookBox)(books=a(ShelfBook)().from_([heavy]))
    assert [box.books for box in generate(pattern)] == [{heavy}]


def test_list_of_patterns_takes_the_domain_object_at_its_position():
    """
    In a list of patterns, a pattern with a domain takes an object of it, and one
    without a domain is constructed.
    """
    light, _ = light_and_heavy_books()
    pattern = a(BookRow)(
        books=[
            a(ShelfBook)().from_([light]),
            a(ShelfBook)(weight=variable_from([2.0])),
        ]
    )
    [row] = generate(pattern)
    assert (row.books[0], row.books[1].weight) == (light, 2.0)


def test_other_attributes_are_still_generated():
    """
    The pattern's other variables are enumerated as before, each combined with the
    object taken from the domain.
    """
    _, heavy = light_and_heavy_books()
    pattern = a(BookStand)(
        book=a(ShelfBook)().from_([heavy]), label=variable_from(["a", "b"])
    )
    stands = generate(pattern)
    assert sorted((stand.label, stand.book) for stand in stands) == [
        ("a", heavy),
        ("b", heavy),
    ]
