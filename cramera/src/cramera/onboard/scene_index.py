"""
The scenes index (``index.json``) the viewer's pickers read.

Kept free of the heavy onboarding imports (URDF/Gazebo/MJCF parsers, ``runpy``, the
monkey-patched :class:`~cramera.onboard.demo.Recorder`) so the always-on static file
server (:mod:`cramera.server`) and the live bridge's recording finalizer
(:mod:`cramera.live.recording_bundle`) can register a scene without pulling in the
offline onboarding pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from cramera import paths
from cramera.generated_json import GeneratedJson, write_json_atomically

RESERVED_SCENE_NAMES = (paths.LIVE_SCENE_NAME, paths.RECORDING_SCENE_NAME)
"""
Throwaway bundle names that are never something a user onboarded or saved, and must
never show up as a robot/environment choice in the real picker.
"""

SCENE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""
What a user-given scene name may look like: safe as a single path segment, without
resorting to escaping or length limits a filesystem might reject.
"""


class InvalidSceneName(Exception):
    """
    Raised by :func:`validate_scene_name` when a user-given name is not a safe, non-
    reserved scene name.
    """


def validate_scene_name(name: str) -> str:
    """
    Check that a user-given name is safe to use as a scene bundle's directory name.

    :param name: The name to validate.
    :return:``name`` unchanged, for chaining.
    :raises InvalidSceneName: If ``name`` is not exactly letters, digits, ``_`` or ``-``
        (1-64 characters), or is one of :data:`RESERVED_SCENE_NAMES`.
    """
    if not SCENE_NAME_PATTERN.match(name):
        raise InvalidSceneName(
            "a scene name must be 1-64 characters of letters, digits, '_' or '-'"
        )
    if name in RESERVED_SCENE_NAMES:
        raise InvalidSceneName("'%s' is a reserved scene name" % name)
    return name


@dataclass
class SceneIndexEntry:
    """
    One onboarded (or saved) scene bundle, as ``index.json`` advertises it to the
    viewer.

    The viewer's header offers a robot and an environment separately, but only ever
    resolves the pair back to a bundle that was actually recorded — these entries are
    what it looks that up in.
    """

    name: str
    """
    Directory name of the bundle, which is also its ``?scene=`` value.
    """

    robot: str
    """
    Name of the robot the scene was recorded with.
    """

    environment: Optional[str]
    """
    The scene's environment models joined by ``+``, or None for a bench-only scene.
    """

    @classmethod
    def of_directory(cls, scenes_directory: Path) -> List[SceneIndexEntry]:
        """
        Every onboarded bundle under a scenes directory, in name order.

        :param scenes_directory: Directory holding the scene bundles.
        """
        entries = []
        for bundle_directory in sorted(scenes_directory.iterdir()):
            if bundle_directory.name in RESERVED_SCENE_NAMES:
                continue  # a throwaway bundle, never something a user onboarded
            scene_path = bundle_directory / "scene.json"
            if not scene_path.is_file():
                continue
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            robot = scene.get("robot") or {}
            entries.append(
                cls(
                    name=bundle_directory.name,
                    robot=robot.get("name", ""),
                    environment=cls._environment_of(scene["models"]),
                )
            )
        return entries

    @staticmethod
    def _environment_of(models: List[Dict[str, Any]]) -> Optional[str]:
        """
        The name of a scene's environment, or None for a bench-only scene.

        :param models: The scene's ``models`` entries.
        """
        environments = [model["name"] for model in models if not model["robot"]]
        return "+".join(environments) if environments else None

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape ``index.json`` carries.
        """
        return {
            "name": self.name,
            "robot": self.robot,
            "environment": self.environment,
        }


def write_scene_index(path: Path, name: str) -> None:
    """
    Register a freshly written scene in the index the viewer reads.

    The ``scenes`` list is rebuilt from the bundles actually on disk, each carrying its
    robot/environment identity for the viewer's pickers, so a bundle that was removed or
    renamed since it was indexed cannot leave a stale entry behind. ``default`` is
    filled in on the first scene onboarded and left alone after that.

    :param path: Path of the scene index file.
    :param name: Name of the scene to register.
    """
    index: Dict[str, Any] = {}
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        index = {}
    index["scenes"] = [
        entry.to_payload() for entry in SceneIndexEntry.of_directory(path.parent)
    ]
    index.setdefault("default", name)
    write_json_atomically(path, index, indent=1)


def merged_scene_index() -> Dict[str, Any]:
    """
    The ``index.json`` the frontend fetches: the shared scenes plus local recordings
    saved under :func:`cramera.paths.local_scenes_directory`, with a local scene
    shadowing a shared one of the same name.

    Both roots are read straight from disk rather than from a persisted merged file, so
    a recording that was just saved (or discarded) shows up immediately.
    """
    shared_directory = paths.scenes_directory()
    local_directory = paths.local_scenes_directory()
    by_name: Dict[str, SceneIndexEntry] = {}
    if shared_directory.is_dir():
        by_name.update(
            {
                entry.name: entry
                for entry in SceneIndexEntry.of_directory(shared_directory)
            }
        )
    if local_directory != shared_directory and local_directory.is_dir():
        by_name.update(
            {
                entry.name: entry
                for entry in SceneIndexEntry.of_directory(local_directory)
            }
        )
    shared_index = GeneratedJson(shared_directory / "index.json").read()
    default = shared_index.get("default") if isinstance(shared_index, dict) else None
    return {
        "default": default,
        "scenes": [by_name[name].to_payload() for name in sorted(by_name)],
    }
