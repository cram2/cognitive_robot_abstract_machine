from __future__ import annotations

import logging
import os
from dataclasses import field, dataclass
from pathlib import Path
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from .detectors.base import SegmindContext
from .episode_player import EpisodePlayer

logger = logging.getLogger(__name__)


@dataclass
class EpisodeSegmentation(TickingExecutorExtension):
    """
    Segments an episode into events by running detector nodes against a shared
    :class:`~segmind.detectors.base.SegmindContext` while an episode player steps the
    world.
    """

    player: EpisodePlayer | None = None
    """
    The episode player responsible for stepping the world, None if the world is stepped
    by someone else.
    """

    def extend_context(self, context: StatechartContext) -> None:
        context.ensure_extension(SegmindContext())

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.detect_holes(executor.context)
        if self.player:
            self.player.start()

    @staticmethod
    def detect_holes(context: StatechartContext) -> None:
        """
        Collects the bodies of the world that have "hole" in their name as the holes of
        the :class:`~segmind.detectors.base.SegmindContext`.

        :param context: The context holding the world and the segmind context.
        """
        segmind_context = context.require_extension(SegmindContext)
        segmind_context.holes.clear()
        for body in context.world.bodies:
            if "hole" in body.name.name:
                segmind_context.holes.append(body)


@dataclass
class EpisodeSceneLoader:
    """
    Loads the models of an episode's scene into a world.
    """

    world: World
    """
    The world the models are loaded into.
    """

    ignored_objects: list[str] = field(default_factory=list)
    """
    Names of the models that are not loaded.
    """

    fixed_objects: list[str] = field(default_factory=list)
    """
    Names of the models that are fixed to the root of the world.
    """

    def spawn_scene(self, models_dir, file_resolver: FileUriResolver | None = None):
        """
        Spawns the scene from the given directory.

        :param models_dir: The directory containing the models to spawn.
        :param file_resolver: The file resolver to use for resolving file URIs.
        """
        directory = Path(models_dir)
        for file in directory.glob("*.urdf"):
            self._load_urdf(file, file_resolver)
        for file in directory.glob("*.stl"):
            self._load_stl(file)

    def _load_urdf(self, file: Path, file_resolver: FileUriResolver | None = None):
        """
        Loads an URDF file into the simulation world.

        :param file: The path to the URDF file to load.
        :param file_resolver: The file resolver to use for resolving file URIs.
        """
        obj_name = file.stem
        if obj_name in self.ignored_objects:
            return
        resolver_kwargs = (
            {"path_resolver": FileUriResolver(base_directory=str(file.parent))}
            if file_resolver is not None
            else {}
        )
        obj_world = URDFParser.from_file(str(file), **resolver_kwargs).create_world()
        connection = (
            FixedConnection(parent=self.world.root, child=obj_world.root)
            if obj_name in self.fixed_objects
            else None
        )
        with self.world.modify_world():
            self.world.merge_world(obj_world, *([connection] if connection else []))

    def _load_stl(self, file: Path):
        """
        Loads an STL file into the simulation world.

        :param file: The path to the STL file to load.
        """
        mesh = Mesh.from_file(str(file))
        new_body = Body(
            name=PrefixedName(file.stem),
            visual=ShapeCollection([mesh]),
            collision=ShapeCollection([mesh]),
        )
        with self.world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                world=self.world,
                parent=self.world.root,
                child=new_body,
            )
            self.world.add_connection(connection)
