"""
Classes that inherit mapped classes next to their primary base: diamonds sharing a
mapped ancestor, mixins without a mapped ancestor, and interfaces, together with classes
whose fields are typed with those bases.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% a diamond: two bases sharing a mapped ancestor


@dataclass(kw_only=True)
class InventoryItem:
    """
    Something an inventory keeps track of.
    """

    label: str
    """
    The name the item is listed under.
    """


@dataclass(kw_only=True)
class Hinge:
    """
    A hinge a door turns on.
    """

    stiffness: float
    """
    How strongly the hinge resists turning, in newton metres per radian.
    """


@dataclass(kw_only=True)
class Furnishing(InventoryItem):
    """
    A piece of furniture.
    """

    leg_count: int
    """
    How many legs the piece stands on.
    """


@dataclass(kw_only=True)
class HasHinges(InventoryItem):
    """
    An item with doors turning on hinges.
    """

    hinges: list[Hinge] = field(default_factory=list)
    """
    The hinges of the item's doors.
    """

    hinge_count: int = 0
    """
    How many hinges the item has, counted when it was listed.
    """


@dataclass(kw_only=True)
class Closet(Furnishing, HasHinges):
    """
    A piece of furniture with hinged doors, primarily a furnishing.
    """

    shelf_count: int
    """
    How many shelves the closet holds.
    """


@dataclass(kw_only=True)
class WalkInCloset(Closet):
    """
    A closet large enough to walk into, inheriting its hinged doors.
    """

    floor_area: float
    """
    The area of its floor, in square metres.
    """


# %% a mixin without a mapped ancestor


@dataclass(kw_only=True)
class Inspected:
    """
    Something an inspector has checked.
    """

    inspector: str
    """
    The name of the inspector.
    """


@dataclass(kw_only=True)
class InspectedCloset(Closet, Inspected):
    """
    A closet that has been inspected.
    """


# %% interfaces


class ReferenceFrame(Interface):
    """
    Something offsets can be measured from.
    """

    @abstractmethod
    def describe_origin(self) -> str:
        """
        :return: A sentence naming where offsets measured from this frame start.
        """


@dataclass(kw_only=True)
class LandmarkFrame(ReferenceFrame):
    """
    A frame fixed at a landmark, primarily a frame.
    """

    landmark_name: str
    """
    The name of the landmark.
    """

    def describe_origin(self) -> str:
        return f"at {self.landmark_name}"


@dataclass(kw_only=True)
class FramedFurnishing(Furnishing, ReferenceFrame):
    """
    A piece of furniture offsets can be measured from, primarily a furnishing.
    """

    def describe_origin(self) -> str:
        return f"at {self.label}"


TagT = TypeVar("TagT")
"""
The type of the tag a tagged frame carries.
"""


class TaggedFrame(Generic[TagT], SubClassSafeGeneric, Interface):
    """
    A frame carrying a tag of a type each implementation chooses.

    .. note:: Its generic base makes :func:`dataclasses.is_dataclass` true for it,
        although it is no dataclass.
    """

    @abstractmethod
    def tag(self) -> TagT:
        """
        :return: The tag of the frame.
        """


@dataclass(kw_only=True)
class TaggedFurnishing(Furnishing, TaggedFrame[str]):
    """
    A piece of furniture tagged with its label.
    """

    def tag(self) -> str:
        return self.label


# %% classes whose fields are typed with those bases


@dataclass(kw_only=True)
class Inventory:
    """
    Items kept track of through the bases they were listed under.
    """

    hinged: HasHinges | None = None
    """
    An item with hinged doors.
    """

    inspected: list[Inspected] = field(default_factory=list)
    """
    Items that have been inspected.
    """

    frame: ReferenceFrame | None = None
    """
    The frame the inventory's offsets are measured from.
    """

    frames: list[ReferenceFrame] = field(default_factory=list)
    """
    Further frames offsets are measured from.
    """

    tagged_frame: TaggedFrame | None = None
    """
    A frame carrying a tag.
    """
