"""
Tests for the type a flattened variable gives its elements, read off the annotation of
the collection it flattens.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import flat_variable, variable
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(kw_only=True, eq=False)
class Label(Symbol):
    text: str


@dataclass(kw_only=True, eq=False)
class Tag(Symbol):
    text: str


@dataclass(kw_only=True, eq=False)
class LabelledCollections(Symbol):
    """
    Holds labels and returns them in collections of differently annotated shapes.
    """

    label_set: set[Label]
    """
    The labels held.
    """

    def label_set_copy(self) -> set[Label]:
        """
        :return: The labels, as a set annotated with its element type.
        """
        return set(self.label_set)

    def label_sequence(self) -> tuple[Label, ...]:
        """
        :return: The labels, as a variable-length tuple.
        """
        return tuple(self.label_set)

    def weight_by_label(self) -> dict[Label, float]:
        """
        :return: A weight for each label, keyed by the label.
        """
        return {label: 1.0 for label in self.label_set}

    def label_and_tag(self) -> tuple[Label, Tag]:
        """
        :return: A pair whose positions have different types.
        """
        return Label(text="first"), Tag(text="second")


# %% Element type of a flattened collection


@pytest.mark.parametrize(
    "collection_of",
    [
        lambda collections: collections.label_set_copy(),
        lambda collections: collections.label_sequence(),
        lambda collections: collections.weight_by_label(),
    ],
    ids=["set", "variable-length tuple", "dictionary"],
)
def test_flattened_collection_has_the_element_type(collection_of):
    """
    A set and a variable-length tuple yield their element type, and a dictionary yields
    its key type, which is what iterating it gives.
    """
    collections = variable(LabelledCollections, domain=[])
    assert flat_variable(collection_of(collections))._type_ is Label


def test_flattened_tuple_with_mixed_element_types_has_their_common_base_class():
    """
    A tuple whose positions have different types yields their lowest common base class,
    as an attribute annotated with a union of those types does.
    """
    collections = variable(LabelledCollections, domain=[])
    assert flat_variable(collections.label_and_tag())._type_ is Symbol
