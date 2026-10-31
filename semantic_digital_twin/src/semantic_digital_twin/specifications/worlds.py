"""
Specifications of whole worlds: an environment, the robots in it and the objects around
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Self, TYPE_CHECKING


from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.base import SpawnSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_source import WorldSource

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.package_resolver import PathResolver

# %% world specifications


@dataclass
class WorldSpecification(WorldSource):
    """
    World-independent description of a world: an environment, the robots in it, and
    objects around them.
    """

    world_source: WorldSource | None = None
    """
    Describes the world the robots and starting objects are added to.

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
        world_source = URDFParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_source=world_source,
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
        use_visual_as_collision_backup: bool = False,
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
        :param use_visual_as_collision_backup: Whether a body with no geom that takes
            part in contact collides with the geoms it does have.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        from semantic_digital_twin.adapters.mjcf import MJCFParser

        world_source = MJCFParser(
            file_path=file_path,
            mimic_joints=mimic_joints or {},
            prefix=prefix,
            use_visual_as_collision_backup=use_visual_as_collision_backup,
        )
        return cls(
            world_source=world_source,
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

        world_source = GazeboParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_source=world_source,
            robots=robots or [],
            objects=objects or [],
        )

    def create_world(self) -> World:
        """
        Differs from
        :meth:`~semantic_digital_twin.world_source.WorldSource.create_world` in
        merging every robot into a new world of :attr:`world_source` first, then
        spawning all :attr:`objects` relative to the world root.

        :return: The world of :attr:`world_source`, holding the robots and objects.
        """
        if self.world_source is not None:
            world = self.world_source.create_world()
        else:
            world = World.create_with_root_body()
        for robot_specification in self.robots:
            robot_specification.spawn(world)

        for object_specification in self.objects:
            object_specification.spawn(world)

        return world
