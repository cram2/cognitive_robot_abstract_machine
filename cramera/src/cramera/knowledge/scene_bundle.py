"""
Reading the active scene bundle (scene.json, trajectory.json, the robot URDF).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from coraplex.datastructures.enums import JointType
from typing_extensions import Any, Dict, List, Optional
from urdf_parser_py.urdf import URDF

from cramera import paths
from cramera.generated_json import GeneratedJson
from cramera.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SceneBundle:
    """
    The active scene's parsed ``scene.json``/``trajectory.json``.
    """

    scene: Dict[str, Any]
    """
    Parsed ``scene.json``, or ``{}`` when no scene is active or it is unreadable.
    """

    trajectory: Dict[str, Any]
    """
    Parsed ``trajectory.json``, or ``{}`` when absent or unreadable.
    """

    @classmethod
    def active_name(cls) -> Optional[str]:
        """
        The active scene: ``CRAMERA_SCENE``, else the scenes-index default.
        """
        environment_override = os.environ.get("CRAMERA_SCENE")
        if environment_override:
            return environment_override
        index_path = paths.scenes_directory() / "index.json"
        if not index_path.is_file():
            return None
        index = GeneratedJson(index_path).read()
        if not isinstance(index, dict):
            return None
        return cls.default_of_index(index)

    @classmethod
    def default_of_index(cls, index: Dict[str, Any]) -> Optional[str]:
        """
        The scene a scenes index opens on: the default it declares, or its first scene
        when that default is not one of the scenes it declares.

        A bundle can outlive the default recorded beside it -- a scene renamed or
        dropped upstream leaves a name that resolves to nothing -- and a scene the
        bundle does have is a better answer than none.

        :param index: Parsed ``index.json``.
        """
        declared = [
            scene["name"]
            for scene in index.get("scenes", [])
            if isinstance(scene, dict) and scene.get("name")
        ]
        default = index.get("default")
        if default in declared:
            return default
        if not declared:
            return None
        if default:
            logger.warning(
                "the scenes index names '%s' as its default but does not declare it; "
                "opening '%s' instead",
                default,
                declared[0],
            )
        return declared[0]

    @classmethod
    def directory_of(cls, scene: Optional[str] = None) -> Optional[Path]:
        """
        Directory of a scene bundle, or None when no scene is named, active, or found in
        any scenes root (see :func:`cramera.paths.resolve_scene_directory`).

        :param scene: Name of the scene, or None for the active one.
        """
        name = scene or cls.active_name()
        return paths.resolve_scene_directory(name) if name else None

    @classmethod
    def of_active_scene(cls) -> SceneBundle:
        """
        The active scene's scene/trajectory bundle, or an empty one without a scene.
        """
        return cls.of_scene(None)

    @classmethod
    def of_scene(cls, scene: Optional[str]) -> SceneBundle:
        """
        One scene's scene/trajectory bundle, or an empty one when it has none.

        :param scene: Name of the scene to read, or None for the active one.
        """
        directory = cls.directory_of(scene)
        if not directory:
            return cls({}, {})
        scene = GeneratedJson(directory / "scene.json").read()
        if not isinstance(scene, dict):
            return cls({}, {})
        trajectory = GeneratedJson(
            directory / scene.get("trajectory", "trajectory.json")
        ).read()
        return cls(scene, trajectory if isinstance(trajectory, dict) else {})

    @classmethod
    def declared_presets(cls, scene: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        The ready-made queries a bundle ships for itself, or none.

        A bundle recorded from a demo knows which questions are worth asking about that
        demo far better than anything generated from the scene's contents can.

        :param scene: Name of the scene to read, or None for the active one.
        """
        directory = cls.directory_of(scene)
        if not directory or not (directory / "presets.json").is_file():
            return []
        declared = GeneratedJson(directory / "presets.json").read()
        if not isinstance(declared, dict):
            return []
        presets = declared.get("presets")
        return presets if isinstance(presets, list) else []


@dataclass
class UrdfJoint:
    """
    One joint of a parsed URDF, as needed by the kinematic-tree view.
    """

    name: str
    """
    Joint name.
    """

    type: JointType
    """
    URDF joint type, e.g. ``REVOLUTE``, ``PRISMATIC``, ``FIXED``.
    """

    parent: str
    """
    Name of the parent link.
    """

    child: str
    """
    Name of the child link.
    """


@dataclass
class ParsedUrdf:
    """
    A scene robot's URDF, parsed into its kinematic-tree shape.
    """

    links: List[str]
    """
    Every link name found in the URDF.
    """

    joints: List[UrdfJoint]
    """
    Every joint found in the URDF.
    """

    @classmethod
    def of_scene(cls, scene_name: Optional[str] = None) -> ParsedUrdf:
        """
        Parse a scene's robot URDF into its kinematic tree.

        :param scene_name: Name of the scene to parse, or None for the active one.
        """
        scene = SceneBundle.of_scene(scene_name).scene
        robot_model = next(
            (model for model in scene.get("models", []) if model.get("robot")), None
        )
        directory = SceneBundle.directory_of(scene_name)
        if not robot_model or not directory:
            return cls([], [])
        urdf_path = directory / robot_model["urdf"]
        if not urdf_path.is_file():
            return cls([], [])
        robot = URDF.from_xml_string(urdf_path.read_text(encoding="utf-8"))
        return cls(
            links=[link.name for link in robot.links],
            joints=[
                UrdfJoint(
                    name=joint.name,
                    type=JointType[joint.type.upper()],
                    parent=joint.parent,
                    child=joint.child,
                )
                for joint in robot.joints
            ],
        )
