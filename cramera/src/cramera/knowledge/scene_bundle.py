"""
Reading the active scene bundle (scene.json, trajectory.json, the robot URDF).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from coraplex.datastructures.enums import JointType
from typing_extensions import Any, Dict, List, Optional

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
        return index.get("default") if isinstance(index, dict) else None

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

        A regex parse suffices because the bundled URDFs are flat.

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
        text = urdf_path.read_text(encoding="utf-8", errors="replace")
        links = re.findall(r'<link\s+name="([^"]+)"', text)
        joints = []
        for joint in re.finditer(
            r'<joint\s+name="([^"]+)"\s+type="([^"]+)">(.*?)</joint>', text, re.S
        ):
            body = joint.group(3)
            parent = re.search(r'<parent\s+link="([^"]+)"', body)
            child = re.search(r'<child\s+link="([^"]+)"', body)
            if parent and child:
                joints.append(
                    UrdfJoint(
                        name=joint.group(1),
                        type=JointType[joint.group(2).upper()],
                        parent=parent.group(1),
                        child=child.group(1),
                    )
                )
        return cls(links, joints)
