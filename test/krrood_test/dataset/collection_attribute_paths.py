"""
Classes whose fields hold collections, for reading attributes of the elements through a
path such as ``shelf.products.colour``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from krrood.symbol_graph.symbol_graph import Symbol

# %% elements


class Colour(Enum):
    """
    The colour of a product.
    """

    RED = 1
    GREEN = 2


@dataclass(eq=False)
class Product(Symbol):
    """
    An element held in collections.
    """

    name: str
    """
    The name of the product.
    """

    colour: Colour
    """
    The colour of the product.
    """

    weight: float
    """
    The weight of the product in kilograms.
    """


# %% collections


@dataclass(eq=False)
class Shelf(Symbol):
    """
    Holds products in a list, and possibly spare ones.
    """

    name: str
    """
    The name of the shelf.
    """

    products: list[Product] = field(default_factory=list)
    """
    The products on the shelf.
    """

    spare_products: list[Product] | None = None
    """
    The spare products kept for the shelf, if any are kept.
    """

    weight_by_product: dict[Product, float] = field(default_factory=dict)
    """
    The weight of each product, keyed by the product.
    """


@dataclass(eq=False)
class Warehouse(Symbol):
    """
    Holds shelves, which in turn hold products.
    """

    name: str
    """
    The name of the warehouse.
    """

    shelves: list[Shelf] = field(default_factory=list)
    """
    The shelves of the warehouse.
    """


# %% name shared by the collection and its elements


@dataclass(eq=False)
class TallyMark(Symbol):
    """
    An element with a field named like a method of :class:`list`.
    """

    count: int
    """
    How many strokes the mark has.
    """


@dataclass(eq=False)
class Tally(Symbol):
    """
    Holds tally marks in a list.
    """

    marks: list[TallyMark] = field(default_factory=list)
    """
    The marks of the tally.
    """


# %% subtypes of an element


@dataclass(eq=False)
class Fruit(Product):
    """
    A product that is a fruit.
    """


@dataclass(eq=False)
class Vegetable(Product):
    """
    A product that is a vegetable.
    """

    grows_underground: bool = False
    """
    Whether the eaten part grows below the ground, as a carrot does.
    """
