"""
Anything a :class:`~semantic_digital_twin.world.World` can be created from, including
the parsers that read one from a world description format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% world source


@dataclass
class WorldSource(ABC):
    """
    A description of a world that can be created anew whenever it is needed, such as a
    world model file, a map written in code or a specification of a whole world.
    """

    @abstractmethod
    def create_world(self) -> World:
        """
        Create the world this source describes.

        Every call creates a new world, so no two results share a world entity
        identifier.

        :return: The created world.
        """


# %% world model parser


@dataclass
class WorldModelParser(WorldSource, ABC):
    """
    Base for every parser that turns a world description format into a
    :class:`~semantic_digital_twin.world.World`.

    Declares only keyword-only fields, so each format keeps its own payload field (the
    description text or the path it is read from) as its first constructor parameter.
    """

    use_visual_as_collision_backup: bool = field(default=False, kw_only=True)
    """
    Whether a body that the description gives no collision geometry collides with its
    visual geometry instead.

    Cosmetic parts such as covers are often drawn but never described for contact, and
    some scenes keep their collision geometry in a file that is not loaded; either way
    the drawn geometry is the best stand-in for the real shape. A body that already has
    collision geometry is left alone, and each format decides what counts as having
    none.
    """

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: str, prefix: str | None = None) -> Self:
        """
        Create a parser for the world described by a file.

        Subclasses may accept further optional parameters for the aspects their format
        supports.

        :param file_path: The path of the file to parse.
        :param prefix: The prefix for every name used in the parsed world.
        :return: The parser for the described world.
        """
