"""
Reading the active scene bundle (scene.json, trajectory.json, the robot URDF).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from cram_viz import get_logger, paths

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


@dataclass
class UrdfJoint:
    """
    One joint of a parsed URDF, as needed by the kinematic-tree view.
    """

    name: str
    """
    Joint name.
    """

    type: str
    """
    URDF joint type, e.g. ``revolute``, ``prismatic``, ``fixed``.
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


def scene_name() -> Optional[str]:
    """
    The active scene: ``CRAM_VIZ_SCENE``, else the scenes-index default.
    """
    environment_override = os.environ.get("CRAM_VIZ_SCENE")
    if environment_override:
        return environment_override
    index_path = paths.scenes_dir() / "index.json"
    if not index_path.is_file():
        return None
    index = _read_json(index_path)
    return index.get("default") if isinstance(index, dict) else None


def _read_json(path: Path) -> Any:
    """
    Read a JSON file, treating unreadable or corrupt content as absent.

    Scene bundles and the scan cache are generated artifacts that a failed run can leave
    half-written; the viewer degrades instead of refusing to start.

    :param path: Path of the JSON file to read.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as error:
        logger.warning("ignoring unreadable %s: %s", path, error)
        return None


def scene_dir() -> Optional[Path]:
    """
    Directory of the active scene bundle, or None without one.
    """
    name = scene_name()
    return paths.scenes_dir() / name if name else None


def load_scene() -> SceneBundle:
    """
    The active scene's scene/trajectory bundle, or an empty one without a scene.
    """
    directory = scene_dir()
    if not directory:
        return SceneBundle({}, {})
    scene = _read_json(directory / "scene.json")
    if not isinstance(scene, dict):
        return SceneBundle({}, {})
    trajectory = _read_json(directory / scene.get("trajectory", "trajectory.json"))
    return SceneBundle(scene, trajectory if isinstance(trajectory, dict) else {})


def load_urdf() -> ParsedUrdf:
    """
    Parse the active scene's robot URDF into its kinematic tree.

    Used by the kinematic-tree view; a regex parse suffices because the bundled URDFs
    are flat.
    """
    scene = load_scene().scene
    robot_model = next(
        (model for model in scene.get("models", []) if model.get("robot")), None
    )
    directory = scene_dir()
    if not robot_model or not directory:
        return ParsedUrdf([], [])
    urdf_path = directory / robot_model["urdf"]
    if not urdf_path.is_file():
        return ParsedUrdf([], [])
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
                    type=joint.group(2),
                    parent=parent.group(1),
                    child=child.group(1),
                )
            )
    return ParsedUrdf(links, joints)
