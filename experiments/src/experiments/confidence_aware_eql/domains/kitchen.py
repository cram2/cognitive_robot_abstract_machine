from dataclasses import dataclass
from enum import Enum

from krrood.symbol_graph.symbol_graph import Symbol


class Material(Enum):
    """The material a kitchen object is made of."""

    CERAMIC = 0
    """Fired clay, used for cups and plates."""

    GLASS = 1
    """Transparent glass, used for pitchers and drinking glasses."""

    METAL = 2
    """Metal, used for pots and pans."""


@dataclass(unsafe_hash=True)
class KitchenObject(Symbol):
    """A graspable kitchen object described by the features a robot reasons about.

    The fields are the features the confidence check scores: mass, characteristic
    size, and material. Being a :class:`Symbol` lets the same object take part in
    entity-query-language rule evaluation.
    """

    weight: float
    """Mass of the object in kilograms."""

    size: float
    """Characteristic dimension of the object in metres."""

    material: Material
    """The material the object is made of."""
