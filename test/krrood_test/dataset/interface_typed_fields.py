"""
Values of two dimensions behind one interface, stored through alternative mappings that
share an abstract base mapping for the interface, and a class whose fields are typed with
the interface.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import Self

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.patterns.interface import Interface

# %% the interface and its implementations


class LabelledPosition(Interface):
    """
    A position in some number of dimensions, carrying the name of what it is measured
    from.
    """

    origin_name: str
    """
    The name of what the position is measured from.
    """


@dataclass
class LabelledPosition2D(LabelledPosition):
    """
    A position given by two coordinates.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """

    origin_name: str
    """
    The name of what the position is measured from.
    """


@dataclass
class LabelledPosition3D(LabelledPosition):
    """
    A position given by three coordinates.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """

    z: float
    """
    The third coordinate.
    """

    origin_name: str
    """
    The name of what the position is measured from.
    """


# %% a class whose fields are typed with the interface


@dataclass
class HolderOfLabelledPositions:
    """
    Holds positions of any dimension.
    """

    position: LabelledPosition
    """
    The position this holder is about.
    """

    route: list[LabelledPosition] = field(default_factory=list)
    """
    Positions passed through, in order.
    """


# %% the mapping hierarchy


@dataclass(eq=False)
class LabelledPositionMapping(AlternativeMapping[LabelledPosition], ABC):
    """
    What every stored position keeps, whatever its dimension.
    """

    origin_name: str
    """
    The name of what the position is measured from.
    """


@dataclass(eq=False)
class LabelledPosition2DMapping(
    LabelledPositionMapping, AlternativeMapping[LabelledPosition2D]
):
    """
    A stored position of two coordinates.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """

    @classmethod
    def from_domain_object(cls, obj: LabelledPosition2D) -> Self:
        return cls(origin_name=obj.origin_name, x=obj.x, y=obj.y)

    def to_domain_object(self) -> LabelledPosition2D:
        return LabelledPosition2D(self.x, self.y, self.origin_name)


@dataclass(eq=False)
class LabelledPosition3DMapping(
    LabelledPositionMapping, AlternativeMapping[LabelledPosition3D]
):
    """
    A stored position of three coordinates.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """

    z: float
    """
    The third coordinate.
    """

    @classmethod
    def from_domain_object(cls, obj: LabelledPosition3D) -> Self:
        return cls(origin_name=obj.origin_name, x=obj.x, y=obj.y, z=obj.z)

    def to_domain_object(self) -> LabelledPosition3D:
        return LabelledPosition3D(self.x, self.y, self.z, self.origin_name)
