"""
Write a live recording as a self-contained scene and trajectory bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

from cramera import paths
from cramera.recording_fields import SceneField
from cramera.body_geometry import POSE_PRECISION, rounded_scale
from cramera.generated_json import write_json_atomically
from cramera.knowledge.recorded_statecharts import (
    RecordedStatecharts,
    STATECHART_FILE,
)
from cramera.live.bridge import Bridge, ObjectCatalogEntry
from cramera.live.live_bundle import bundle_world_models
from cramera.live.recording import Recording, RecordedFrame, RecordingState
from cramera.live.recording_segments import derive_segments
from cramera.mesh_format import MeshFormat
from cramera.onboard.bundle_urdf import BundledAssets, companion_material_library

MESH_SUBDIRECTORY = "recording"
"""
Directory the recording's models' meshes nest under, inside the bundle.
"""


class NothingToBundle(Exception):
    """
    Raised by :func:`write_recording_bundle` when the recording has no frames.
    """


def write_recording_bundle(
    bridge: Bridge,
    frames: List[RecordedFrame],
    frames_per_second: float,
    output_directory: Path,
    scene_name: str,
) -> Dict[str, Any]:
    """
    Write a finalized recording's geometry and trajectory to disk.

    Clears ``output_directory`` first, mirroring
    :func:`cramera.live.live_bundle.build_live_scene`'s own throwaway-bundle behaviour.

    :param bridge: The live bridge whose world and object catalog are bundled.
    :param frames: The recording's buffered ticks, in order.
    :param frames_per_second: The recording's estimated frame rate.
    :param output_directory: Directory the bundle is written into.
    :param scene_name: Name the bundle's ``scene.json`` carries.
    :raises NothingToBundle: If no ticks were recorded.
    """
    if not frames:
        raise NothingToBundle("the recording has no frames")
    with bridge.bundle_lock:
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True)
        geometry = bundle_world_models(
            bridge.world,
            bridge.robot,
            output_directory,
            MESH_SUBDIRECTORY,
            overlay_bodies=bridge.overlay_bodies(),
        )
        objects = _loose_object_entries(bridge, frames[0], output_directory)
        scene = {
            "name": scene_name,
            "framesPerSecond": frames_per_second,
            "trajectory": "trajectory.json",
            "models": geometry.models,
            "robot": geometry.robot,
            "objects": objects,
            "segments": [segment.to_payload() for segment in derive_segments(frames)],
            "missingAssets": geometry.missing_assets,
            "worldBound": True,
            "bundleSignature": bridge.bundle_signature(),
            SceneField.PLAN_TREES: bridge.plan_state.recorded_trees(),
        }
        statecharts = RecordedStatecharts.of_snapshots(
            frame.statechart for frame in frames
        )
        if not statecharts.is_empty():
            scene["statecharts"] = STATECHART_FILE
            write_json_atomically(
                output_directory / STATECHART_FILE, statecharts.to_payload()
            )
        write_json_atomically(output_directory / "scene.json", scene, indent=1)
        write_json_atomically(
            output_directory / "trajectory.json",
            {
                "framesPerSecond": frames_per_second,
                "frames": [frame.frames for frame in frames],
                "base": [frame.base for frame in frames],
                "objects": [frame.objects for frame in frames],
            },
        )
        return scene


def finalize_recording(bridge: Bridge, recording: Recording) -> Optional[str]:
    """
    Ensure a recording's buffered frames have been written to disk, bundling them on
    first call.

    Idempotent and safe to call more than once — e.g. once from the viewer's explicit
    ``/recording/stop`` request, and again as a safety net if the demo process exits
    before that request ever arrives (see :mod:`cramera.live.visualization`): a
    recording already bundled, or one with nothing to bundle, is left untouched.

    :param bridge: The live bridge whose world and object catalog are bundled.
    :param recording: The recording to finalize.
    :return: The scene name the recording was bundled under, or None if there is nothing
        to bundle (capture never started, or no ticks were recorded).
    """
    with bridge.bundle_lock:
        if recording.state is RecordingState.IDLE:
            return None
        frames = recording.stop()
        if recording.scene_name is not None:
            return recording.scene_name
        if not frames:
            return None
        output_directory = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
        write_recording_bundle(
            bridge,
            frames,
            recording.frames_per_second(),
            output_directory,
            paths.RECORDING_SCENE_NAME,
        )
        recording.scene_name = paths.RECORDING_SCENE_NAME
        return recording.scene_name


def _loose_object_entries(
    bridge: Bridge, first_frame: RecordedFrame, output_directory: Path
) -> List[Dict[str, Any]]:
    """
    ``scene.json``'s ``objects`` entries for every catalog object present in the first
    recorded frame.

    An object the catalog knows about but that never appears in the first tick (spawned
    mid-recording) is skipped: an ``objects`` entry can only declare one static spawn
    pose, the same convention the offline onboarding pipeline uses.

    :param bridge: The live bridge whose object catalog and bodies are read.
    :param first_frame: The recording's first tick, whose ``objects`` poses double as
        each entry's spawn pose.
    :param output_directory: Directory a mesh-backed object's file is copied into.
    """
    entries = []
    for entry in bridge.object_metadata:
        spawn = first_frame.objects.get(entry.key)
        body = bridge.object_body(entry.key)
        if spawn is None or body is None:
            continue
        entries.append(_object_entry(entry, spawn, output_directory))
    return entries


def _object_entry(
    entry: ObjectCatalogEntry, spawn: List[float], output_directory: Path
) -> Dict[str, Any]:
    """
    One loose object's ``scene.json`` entry, including its geometry and material
    library.

    An inline box must be centered and aligned with its body. Other local transforms are
    retained in exported mesh geometry.

    :param entry: The object's publication identity, colour and native geometry.
    :param spawn: The object's pose in the recording's first frame.
    :param output_directory: Directory a mesh file is written into.
    """
    payload: Dict[str, Any] = {
        "id": entry.id,
        "key": entry.key,
        "spawn": spawn,
        "color": entry.color,
    }
    shapes = entry.shapes
    payload["height"] = round(float(shapes.combined_mesh.extents[2]), POSE_PRECISION)
    if (
        len(shapes) == 1
        and isinstance(shapes[0], Box)
        and shapes[0].origin.equivalent(HomogeneousTransformationMatrix())
    ):
        payload["box"] = rounded_scale(shapes[0].scale, POSE_PRECISION)
        return payload
    mesh_path = _write_object_mesh(entry.key, shapes, output_directory)
    payload[SceneField.MESH] = mesh_path
    material_library = companion_material_library(output_directory, mesh_path)
    if material_library is not None:
        payload[SceneField.MATERIAL_LIBRARY] = material_library
    return payload


def _write_object_mesh(
    key: str, shapes: ShapeCollection, output_directory: Path
) -> str:
    """
    Write a loose object's geometry into the bundle and answer the path it is served at.

    A single untransformed mesh with unit scale is copied with its side assets
    (materials, textures). Other geometry is exported from the collection's combined
    mesh, which retains each shape's local transform and scale. Each export has its own
    directory for generated materials and textures.

    :param key: The object's catalog key, used as the written file's basename.
    :param shapes: The native geometry selected for the object's catalog entry.
    :param output_directory: Directory the mesh is written into.
    """
    objects_directory = output_directory / "meshes" / "objects"
    objects_directory.mkdir(parents=True, exist_ok=True)
    if (
        len(shapes) == 1
        and isinstance(shapes[0], Mesh)
        and shapes[0].origin.equivalent(HomogeneousTransformationMatrix())
        and shapes[0].scale == Scale()
    ):
        source = shapes[0].filename
        if source and Path(source).is_file():
            destination = objects_directory / (key + Path(source).suffix)
            assets = BundledAssets(bundle_root=str(output_directory))
            if assets.copy(source, str(destination)):
                assets.copy_side_assets(source, str(destination))
                return "meshes/objects/" + destination.name
    destination = objects_directory / key / (Path(key).name + MeshFormat.OBJ.value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shapes.combined_mesh.export(str(destination))
    return destination.relative_to(output_directory).as_posix()
