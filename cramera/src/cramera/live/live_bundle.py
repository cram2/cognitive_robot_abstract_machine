"""
Bundles a live world's *current* model into a throwaway scene.

Serializes the world object the bridge is attached to — the robot subtree as one
model, everything else as the environment model — through the same
:class:`~cramera.onboard.world_to_urdf.UrdfDocument` serializer every recorded scene
goes through. Whatever a demo built its world from, and however it built it, is what
the viewer shows; no model source tracking or manual onboarding is involved.

A build is idempotent: while the world model is unchanged, the existing bundle is left
untouched — a viewer may be downloading its files at any moment, and deleting them
mid-flight breaks the page that requested the rebuild in the first place.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.generated_json import GeneratedJson
from cramera.live.bridge import Bridge
from cramera.mesh_format import MeshFormat
from cramera.onboard.bundle_urdf import BundleReport
from cramera.onboard.world_to_urdf import UrdfDocument
from cramera.robot_parts import RobotPartAnnotation

BUILD_LOCK = threading.Lock()
"""
Serializes bundling across this process.

The output directory is one fixed path, and a build clears it before writing it, so two
overlapping builds delete and overwrite each other's files. The bridge answers on a
threading HTTP server and the viewer keeps asking for the live scene while a demo starts
up, which makes that overlap routine rather than rare.
"""

ENVIRONMENT_MODEL_NAME = "environment"
"""
Name of the model holding every world body outside the robot and the object overlay.
"""

MESH_SUBDIRECTORY = "live"
"""
Directory bundled meshes nest under inside the scene's ``meshes/`` tree.
"""


def build_live_scene(bridge: Bridge) -> Optional[str]:
    """
    Bundle the live world's current model into a throwaway scene.

    :param bridge: The live bridge whose current world is bundled.
    :return: :data:`cramera.paths.LIVE_SCENE_NAME`, the scene name to navigate the
        viewer to, or None while no world is attached yet.
    """
    if bridge.world is None:
        return None
    output_directory = paths.scenes_directory() / paths.LIVE_SCENE_NAME
    with BUILD_LOCK:
        signature = bridge.bundle_signature()
        if _existing_signature(output_directory) == signature:
            return paths.LIVE_SCENE_NAME
        return _write_bundle(bridge, output_directory, signature)


def _existing_signature(output_directory: Path) -> Optional[str]:
    """
    The signature the existing bundle was built from, or None without a readable one.

    :param output_directory: Directory the previous bundle was written to.
    """
    scene = GeneratedJson(output_directory / "scene.json").read()
    if not isinstance(scene, dict):
        return None
    return scene.get("bundleSignature")


def _write_bundle(bridge: Bridge, output_directory: Path, signature: str) -> str:
    """
    Clear the throwaway scene's directory and serialize the world into it.

    Only ever called while :data:`BUILD_LOCK` is held.

    :param bridge: The live bridge whose current world is bundled.
    :param output_directory: Directory the bundle is written to, cleared first.
    :param signature: The digest of the world model this bundle is built from, recorded
        so the next build can tell whether anything changed.
    """
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    world, robot = bridge.world, bridge.robot
    robot_bodies = _robot_bodies(world, robot)
    models: List[Dict[str, Any]] = []
    environment_bodies = [
        body
        for body in world.bodies_topologically_sorted
        if body not in set(robot_bodies) and not _is_overlay_body(body)
    ]
    if environment_bodies:
        report = UrdfDocument.of_bodies(
            environment_bodies,
            ENVIRONMENT_MODEL_NAME,
            str(output_directory),
            MESH_SUBDIRECTORY,
        )
        models.append(_model_payload(report, is_robot=False))
    if robot_bodies:
        report = UrdfDocument.of_bodies(
            robot_bodies,
            type(robot).__name__.lower(),
            str(output_directory),
            MESH_SUBDIRECTORY,
            identity_root=robot.root,
        )
        models.append(_model_payload(report, is_robot=True))
    missing_assets = sorted(
        {missing for model in models for missing in model.pop("missing")}
    )
    scene = {
        "name": paths.LIVE_SCENE_NAME,
        "models": models,
        "robot": _robot_payload(robot),
        "objects": [],
        "segments": [],
        "missingAssets": missing_assets,
        # the world was attached before this bundle was built; kept for the viewer,
        # which rebuilds a bundle that reports False
        "worldBound": True,
        # what this bundle was built from; while it is unchanged, /live_scene leaves
        # the bundle untouched instead of deleting files a viewer may be downloading
        "bundleSignature": signature,
    }
    (output_directory / "scene.json").write_text(json.dumps(scene, indent=1))
    return paths.LIVE_SCENE_NAME


def _robot_bodies(world: World, robot: Optional[AbstractRobot]) -> List[Body]:
    """
    The robot's subtree in serialization order, or an empty list without a robot.

    :param world: The world the robot lives in.
    :param robot: The robot annotation, or None.
    """
    if robot is None:
        return []
    subtree = set(world.get_kinematic_structure_entities_of_branch(robot.root))
    return [body for body in world.bodies_topologically_sorted if body in subtree]


def _is_overlay_body(body: Body) -> bool:
    """
    Whether the object overlay renders this body instead of the scene bundle.

    Bodies named like mesh files are demo objects that spawn, move and disappear mid-
    run; the overlay streams their poses live, so baking them into the bundle would show
    them twice.

    :param body: The body to check.
    """
    return MeshFormat.of_path(str(body.name).split("/")[-1]) is not None


def _model_payload(report: BundleReport, is_robot: bool) -> Dict[str, Any]:
    """
    One entry of the scene's ``models`` list.

    :param report: What the serializer wrote for this model.
    :param is_robot: Whether the entry is the robot rather than the environment.
    """
    return {
        "name": report.name,
        "urdf": "%s.urdf" % report.name,
        "prefix": "",
        "robot": is_robot,
        "links": len(report.links),
        "movableJoints": report.movable_joints,
        "missing": report.missing,
    }


def _robot_payload(robot: Optional[AbstractRobot]) -> Optional[Dict[str, Any]]:
    """
    The scene's ``robot`` field, or None if no robot is bound.

    :param robot: The robot's semantic annotation, or None if no robot is bound.
    """
    if robot is None:
        return None
    root_name = str(robot.root.name)
    part_annotations = RobotPartAnnotation.of_robot(robot)
    return {
        "name": type(robot).__name__.lower(),
        "prefix": root_name.split("/", 1)[0] if "/" in root_name else "",
        "baseBody": root_name.split("/", 1)[-1],
        "parts": {annotation.name: annotation.links for annotation in part_annotations},
        "partAnnotations": [annotation.to_payload() for annotation in part_annotations],
    }
