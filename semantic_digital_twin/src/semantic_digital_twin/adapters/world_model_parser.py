from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Optional, Self

from semantic_digital_twin.world import World

# %% world model parser


@dataclass
class WorldModelParser(ABC):
    """
    Base for every parser that turns a world description format into a
    :class:`~semantic_digital_twin.world.World`.

    Declares no fields, so each format keeps its own payload field (the description text
    or the path it is read from) as its first constructor parameter.

    Every parse produces freshly created world entities, so a parser is the way to obtain
    a world that shares no identifiers with any previously parsed one.
    """

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: str, prefix: Optional[str] = None) -> Self:
        """
        Create a parser for the world described by a file.

        Subclasses may accept further optional parameters for the aspects their format
        supports.

        :param file_path: The path of the file to parse.
        :param prefix: The prefix for every name used in the parsed world.
        :return: The parser for the described world.
        """

    @abstractmethod
    def parse(self) -> World:
        """
        Build the world described by this parser's source.

        :return: The parsed world.
        """
