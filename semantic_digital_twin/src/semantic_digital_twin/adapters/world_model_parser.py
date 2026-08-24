from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Optional, Self, Type

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection

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


# %% joint description


@dataclass
class JointDescription:
    """
    A joint that has been read from a description and validated, but not yet turned into
    a connection.

    Reading a joint can fail on constructs a parser does not support, while creating a
    connection modifies a world. Keeping the two apart lets a failure surface before a
    world has been touched, shared by every format whose joints are described this way
    (SDF, USD, ...).
    """

    connection_type: Type[Connection]
    """
    The type of connection the joint becomes.
    """

    parent: Body
    """
    The body the joint moves the child relative to.
    """

    child: Body
    """
    The body the joint moves.
    """

    parent_T_connection: HomogeneousTransformationMatrix
    """
    The pose of the joint relative to the parent body.
    """

    connection_T_child: HomogeneousTransformationMatrix
    """
    The pose of the child body relative to the joint.
    """

    name: Optional[PrefixedName] = None
    """
    The name of the connection the joint becomes, ``None`` to let the connection
    generate a default name from its parent and child.
    """

    axis: Optional[Vector3] = None
    """
    The axis the joint moves along, in the frame of the joint.
    """

    limits: Optional[DegreeOfFreedomLimits] = None
    """
    The limits of the joint's degree of freedom.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    The dynamic properties of the joint.
    """
