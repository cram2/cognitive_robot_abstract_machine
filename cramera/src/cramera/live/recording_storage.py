"""
Managing an already-finalized live-recording bundle on disk.

Deliberately free of :mod:`cramera.live.bridge`/``semantic_digital_twin`` — once a
recording has been written to disk (by :func:`cramera.live.recording_bundle.
finalize_recording`, whether from an explicit ``/recording/stop`` or its exit-time
safety net), discarding or saving it is a pure filesystem operation that works whether
or not the demo process that produced it is still running. This is what lets
:mod:`cramera.server` (the always-on viewer process, on a different port than the live
bridge) offer the same actions as a fallback once that process is gone.
"""

from __future__ import annotations

import json
import shutil

from cramera import paths
from cramera.generated_json import write_json_atomically
from cramera.onboard.scene_index import validate_scene_name, write_scene_index


class NoSavedRecording(Exception):
    """
    Raised by :func:`save_recording_bundle` when no finalized ``__recording__`` bundle
    exists on disk to save.
    """


class SceneNameTaken(Exception):
    """
    Raised by :func:`save_recording_bundle` when the requested name already names a
    scene in a shared or local scenes root.
    """


def has_saveable_recording() -> bool:
    """
    Whether a finalized ``__recording__`` bundle currently exists on disk.
    """
    return (
        paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME / "scene.json"
    ).is_file()


def discard_recording_bundle() -> None:
    """
    Delete the unsaved ``__recording__`` bundle from disk, if one exists.
    """
    shutil.rmtree(
        paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME, ignore_errors=True
    )


def save_recording_bundle(name: str) -> str:
    """
    Promote the finalized ``__recording__`` bundle to a permanent, locally saved scene.

    :param name: Name to save the recording under.
    :raises cramera.onboard.scene_index.InvalidSceneName: If ``name`` is unsafe or
        reserved.
    :raises NoSavedRecording: If no finalized ``__recording__`` bundle exists on disk.
    :raises SceneNameTaken: If ``name`` already names a scene in any scenes root.
    """
    validate_scene_name(name)
    source = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
    if not (source / "scene.json").is_file():
        raise NoSavedRecording("no finalized recording to save")
    if any((root / name).is_dir() for root in paths.scene_roots()):
        raise SceneNameTaken("a scene named '%s' already exists" % name)
    destination = paths.local_scenes_directory() / name
    shutil.move(str(source), str(destination))
    scene_path = destination / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["name"] = name
    write_json_atomically(scene_path, scene, indent=1)
    write_scene_index(paths.local_scenes_directory() / "index.json", name)
    return name
