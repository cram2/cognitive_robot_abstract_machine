"""
Tests for how ``==`` compares two collections: sequences by order, length and
duplicates, any other pair of collections as sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from krrood.entity_query_language.factories import an, entity, variable
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(eq=False)
class Shelf(Symbol):
    name: str
    ordered_books: list[str] = field(default_factory=list)
    book_set: set[str] = field(default_factory=set)


# %% Equality of collections


@pytest.mark.parametrize(
    "compared_value, expected_names",
    [
        (["a", "b"], ["forward"]),
        (("b", "a"), ["backward"]),
        (["a"], ["single"]),
    ],
    ids=["same order", "tuple against list", "no duplicates"],
)
def test_sequences_are_equal_by_order_length_and_duplicates(
    compared_value, expected_names
):
    """
    A list or tuple equals another only with the same elements in the same order.
    """
    shelves = [
        Shelf("forward", ["a", "b"]),
        Shelf("backward", ["b", "a"]),
        Shelf("repeated", ["a", "a"]),
        Shelf("single", ["a"]),
    ]
    shelf = variable(Shelf, domain=shelves)
    query = an(entity(shelf).where(shelf.ordered_books == compared_value))
    assert sorted(match.name for match in query.evaluate()) == expected_names


def test_a_set_is_equal_to_a_sequence_as_a_set():
    """
    A set compared with a list is compared as a set, ignoring the list's order.
    """
    shelves = [Shelf("both", book_set={"a", "b"}), Shelf("one", book_set={"a"})]
    shelf = variable(Shelf, domain=shelves)
    query = an(entity(shelf).where(shelf.book_set == ["b", "a"]))
    assert [match.name for match in query.evaluate()] == ["both"]
