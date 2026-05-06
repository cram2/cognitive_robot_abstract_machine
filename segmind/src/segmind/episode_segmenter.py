import logging
import os
from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, List
from giskardpy.executor import Executor
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from .detectors.base import DetectorStateChart, SegmindContext
from .episode_player import EpisodePlayer

logger = logging.getLogger(__name__)

@dataclass
class EpisodeSegmenterExecutor(Executor):
    """
    Handles the segmentation of episodes by controlling the execution of a
    detector statechart and maintaining interactive control cycles.

    This class orchestrates interaction between the detector statechart,
    the simulation player, and the context, enabling episode segmentation
    and tick-based interactions. It allows for spawning scenes, managing
    holes, and ensuring state model updates during execution.
    """

    player: EpisodePlayer | None = None
    """
    The episode player responsible for stepping the world. This can be None if no player is used.
    """

    statechart: DetectorStateChart = field(init=False)
    """
    The detector statechart that drives the episode execution.
    """

    ignored_objects: Optional[List[str]] = field(default_factory=list)
    """
    A list of objects that should be ignored during the episode.
    """

    fixed_objects: Optional[List[str]] = field(default_factory=list)
    """
    A list of objects that should be fixed during the episode.
    """


    def __post_init__(self):
        """
        Adds the SegmindContext extension to the context.
        """
        super().__post_init__()
        self.context.add_extension(SegmindContext())


    def start(self):
        """
        Starts the episode player.
        """
        if self.player:
            self.player.start()


    def compile(self, motion_statechart: DetectorStateChart):
        """
        Compiles the provided statechart and initializes the episode segmenter for execution.
        """
        super().compile(motion_statechart)
        self.detect_holes()
        if self.player:
            self.player.start()


    def detect_holes(self):
        """
        Iterates through objects in the world's context and appends objects with
        "hole" in their name to the list of holes.
        """
        segmind_context = self.context.require_extension(SegmindContext)
        segmind_context.holes.clear()
        for o in self.context.world.bodies:
            if "hole" in o.name.name:
                segmind_context.holes.append(o)

    def spawn_scene(self, models_dir, file_resolver: Optional[FileUriResolver] = None):
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

    def _load_urdf(self, file: Path, file_resolver: Optional[FileUriResolver] = None):
        """
        Loads an URDF file into the simulation world.

        :param file: The path to the URDF file to load.
        :param file_resolver: The file resolver to use for resolving file URIs.
        """
        obj_name = file.stem
        if obj_name in self.ignored_objects:
            return
        try:
            resolver_kwargs = (
                {"path_resolver": FileUriResolver(base_directory=str(file.parent))}
                if file_resolver is not None
                else {}
            )
            obj_world = URDFParser.from_file(str(file), **resolver_kwargs).parse()
            connection = (
                FixedConnection(parent=self.context.world.root, child=obj_world.root)
                if obj_name in self.fixed_objects
                else None
            )
            with self.context.world.modify_world():
                self.context.world.merge_world(obj_world, *([connection] if connection else []))
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"File issue with {file}: {e}")

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
        with self.context.world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                world=self.context.world,
                parent=self.context.world.root,
                child=new_body,
            )
            self.context.world.add_connection(connection)
