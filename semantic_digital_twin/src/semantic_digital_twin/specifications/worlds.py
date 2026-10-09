"""
Specifications of whole worlds: an environment, the robots in it and the objects around
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Self

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.world_model_parser import WorldModelParser
from semantic_digital_twin.specifications.base import SpawnSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.package_resolver import PathResolver

# %% world specifications


@dataclass
class WorldSpecification:
    """
    World-independent description of a world: an environment, the robots in it, and
    objects around them.

    The environment is described by the parser that builds it (obtained from a model
    file with :meth:`from_urdf`, :meth:`from_mjcf` or :meth:`from_gazebo`). Applying it
    (:meth:`to_domain_object`) parses the environment anew, merges every robot into it,
    then spawns all starting objects, and returns the augmented environment world.
    """

    world_parser: WorldModelParser | None = None
    """
    The parser building the environment the robots and starting objects are added to.

    ``None`` describes an environment holding nothing but its root body.
    """

    robots: list[RobotSpecification] = field(default_factory=list)
    """
    The robots merged into the environment, each with its own localization and start
    pose.
    """

    objects: list[SpawnSpecification] = field(default_factory=list)
    """
    Specifications spawned relative to the world root once the robots are in place.
    """

    @classmethod
    def from_urdf(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from a URDF file.

        :param file_path: Path to the environment URDF. This is never a robot
            description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param path_resolver: Resolver for mesh/package paths referenced by the URDF.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        world_parser = URDFParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    @classmethod
    def from_mjcf(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        mimic_joints: dict[str, str] | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from an MJCF (MuJoCo XML)
        file.

        :param file_path: Path to the environment MJCF. This is never a robot
            description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param mimic_joints: Mapping of joint names to the joints they mimic.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        from semantic_digital_twin.adapters.mjcf import MJCFParser

        world_parser = MJCFParser(
            file_path=file_path,
            mimic_joints=mimic_joints or {},
            prefix=prefix,
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    @classmethod
    def from_gazebo(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from a Gazebo SDF world or
        model file.

        :param file_path: Path to the environment world or model file. This is never a
            robot description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param path_resolver: Resolver for the ``model://`` and mesh URIs the file
            references. Defaults to one that searches next to the file.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        from semantic_digital_twin.adapters.gazebo import GazeboParser

        world_parser = GazeboParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    def to_domain_object(self) -> World:
        """
        Materialize a new World from this specification.

        The environment is parsed anew, so the method can be applied repeatedly and no
        two results share an entity identifier. Every robot is merged into the world
        first, then all ``objects`` are spawned relative to the world root.

        :return: The augmented environment world.
        """
        if self.world_parser is not None:
            world = self.world_parser.parse()
        else:
            world = World.create_with_root_body()
        for robot_specification in self.robots:
            robot_specification.spawn(world)

        for object_specification in self.objects:
            object_specification.spawn(world)

        return world
