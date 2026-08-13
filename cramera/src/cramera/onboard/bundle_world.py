"""
Bundling a Gazebo/SDF or MJCF source into the self-contained URDF format the web viewer
already loads.

Where :class:`cramera.onboard.bundle_urdf.BundleReport` rewrites the mesh references of
an existing URDF, this builds a URDF from scratch: the adapter has already resolved every
shape and pose into a :class:`~semantic_digital_twin.world.World`, so bundling only has
to serialize that world.
"""

from __future__ import annotations

import dataclasses
import os

from dataclasses import dataclass

from semantic_digital_twin.adapters.gazebo import GazeboParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world import World
from typing_extensions import Callable, ClassVar, Dict, Optional

from cramera.onboard.bundle_urdf import BundleReport, MeshReference
from cramera.onboard.world_to_urdf import UrdfDocument


@dataclass
class BundledWorld:
    """
    A world parsed from a non-URDF source and written into a scene bundle as URDF.
    """

    GAZEBO_MESH_DIRECTORY: ClassVar[str] = "gazebo"
    """
    Directory a bundled Gazebo mesh nests under, so meshes from differently named
    models cannot collide.
    """

    MJCF_MESH_DIRECTORY: ClassVar[str] = "mjcf"
    """
    The same, for MJCF sources.
    """

    @classmethod
    def of_gazebo_source(
        cls,
        source: str,
        name: str,
        output_directory: str,
        hints: Optional[Dict[str, str]] = None,
    ) -> BundleReport:
        """
        Bundle one Gazebo/SDF world or model, with every mesh it references.

        :param source: Path or URI of the world/model file to bundle.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree go into.
        :param hints: Resolutions recorded while a demo ran.
        """
        return cls._of_parsed_world(
            source,
            name,
            output_directory,
            hints,
            lambda path: GazeboParser.from_file(path).parse(),
            cls.GAZEBO_MESH_DIRECTORY,
            "Gazebo",
        )

    @classmethod
    def of_mjcf_source(
        cls,
        source: str,
        name: str,
        output_directory: str,
        hints: Optional[Dict[str, str]] = None,
    ) -> BundleReport:
        """
        Bundle one MJCF robot or scene, with every mesh it references.

        :param source: Path or URI of the MJCF file to bundle.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree go into.
        :param hints: Resolutions recorded while a demo ran.
        """
        return cls._of_parsed_world(
            source,
            name,
            output_directory,
            hints,
            lambda path: MJCFParser(path).parse(),
            cls.MJCF_MESH_DIRECTORY,
            "MJCF",
        )

    @classmethod
    def _of_parsed_world(
        cls,
        source: str,
        name: str,
        output_directory: str,
        hints: Optional[Dict[str, str]],
        parse: Callable[[str], World],
        mesh_subdirectory: str,
        format_name: str,
    ) -> BundleReport:
        """
        Resolve a source file, parse it into a world and serialize that world as URDF.

        :param source: Path or URI of the file to bundle.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree go into.
        :param hints: Resolutions recorded while a demo ran.
        :param parse: Turns the resolved source path into a world.
        :param mesh_subdirectory: Directory bundled meshes nest under.
        :param format_name: The source format, used in the not-found message.
        :raises FileNotFoundError: If the source itself cannot be found.
        """
        source_path = MeshReference(source).resolve(hints=hints) or source
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                "%s source not found: %s (from %s)"
                % (format_name, source_path, source)
            )
        report = UrdfDocument.of_world(
            parse(source_path), name, output_directory, mesh_subdirectory
        )
        return dataclasses.replace(report, source=source_path)
