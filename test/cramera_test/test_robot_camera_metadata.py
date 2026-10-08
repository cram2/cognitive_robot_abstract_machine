"""
Native camera annotations survive live and recorded scene publication.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree

import pytest
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import Camera

from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording import RecordedFrame
from cramera.live.recording_bundle import write_recording_bundle
from cramera.robot_camera import CameraField, RobotCamera

from .test_live_bundle import use_scratch_scenes_directory

if TYPE_CHECKING:
    from pathlib import Path

    from semantic_digital_twin.world import World


# %% native camera metadata
@pytest.fixture
def camera_robot(pr2_world_copy: World) -> PR2:
    """
    Return the native robot whose annotated cameras are being published.
    """
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    return robot


def test_camera_metadata_uses_native_values_and_name(camera_robot: PR2) -> None:
    """
    Preserve a named sensor's identity, frame, direction and native field of view.
    """
    camera = camera_robot.get_default_camera()
    with camera_robot._world.modify_world():
        camera.update_name(PrefixedName("inspection", prefix="sensors"))

    payload = RobotCamera(camera).to_payload()

    assert payload == {
        CameraField.NAME: str(camera.name),
        CameraField.LINK: str(camera.root.name),
        CameraField.FORWARD: camera.forward_facing_axis.to_np()[:3].tolist(),
        CameraField.HORIZONTAL_ANGLE: camera.field_of_view.horizontal_angle,
        CameraField.VERTICAL_ANGLE: camera.field_of_view.vertical_angle,
        CameraField.DEFAULT: camera.default_camera,
    }


def test_only_native_cameras_are_offered(camera_robot: PR2) -> None:
    """
    Exclude non-camera sensors even when they belong to the robot.
    """
    cameras = RobotCamera.of_robot(camera_robot)

    assert [entry.camera for entry in cameras] == [
        sensor for sensor in camera_robot.all_sensors if isinstance(sensor, Camera)
    ]
    assert any(not isinstance(sensor, Camera) for sensor in camera_robot.all_sensors)


def test_a_robot_without_cameras_has_no_viewpoints(camera_robot: PR2) -> None:
    """
    Camera-like link names cannot introduce viewpoints without native annotations.
    """
    camera_robot.mobile_base.torso.neck.sensors.clear()

    assert RobotCamera.of_robot(camera_robot) == []


# %% live and recorded publication
@pytest.fixture
def camera_bridge(pr2_world_copy: World) -> Bridge:
    """
    Attach a native robot world to the ordinary scene publisher.
    """
    bridge = Bridge()
    bridge.attach(pr2_world_copy)
    return bridge


def test_live_scene_publishes_camera_metadata(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Publish descriptors whose exact frame names resolve in the bundled URDF.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

    scene_name = build_live_scene(camera_bridge)
    scene_directory = scenes / scene_name
    scene = json.loads((scene_directory / "scene.json").read_text())

    assert scene["robot"][CameraField.CAMERAS] == [
        camera.to_payload() for camera in RobotCamera.of_robot(camera_bridge.robot)
    ]
    robot_model = next(model for model in scene["models"] if model["robot"])
    links = {
        link.attrib["name"]
        for link in ElementTree.parse(scene_directory / robot_model["urdf"]).findall(
            "link"
        )
    }
    assert all(
        camera[CameraField.LINK] in links
        for camera in scene["robot"][CameraField.CAMERAS]
    )


def test_recording_preserves_camera_metadata(
    camera_bridge: Bridge, tmp_path: Path
) -> None:
    """
    Persist the camera descriptors alongside the recorded robot model.
    """
    frame = RecordedFrame(frames={}, base=None, objects={})

    scene = write_recording_bundle(
        camera_bridge, [frame], 30.0, tmp_path / "recording", "camera-recording"
    )

    recorded_scene = json.loads((tmp_path / "recording" / "scene.json").read_text())
    expected = [
        camera.to_payload() for camera in RobotCamera.of_robot(camera_bridge.robot)
    ]
    assert scene["robot"][CameraField.CAMERAS] == expected
    assert recorded_scene["robot"][CameraField.CAMERAS] == expected


def test_cached_live_scene_without_camera_metadata_is_rebuilt(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Refresh an older bundle even when the robot's body topology is unchanged.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    scene_path = scenes / scene_name / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = scene["robot"].pop(CameraField.CAMERAS)
    scene_path.write_text(json.dumps(scene))

    build_live_scene(camera_bridge)

    assert json.loads(scene_path.read_text())["robot"][CameraField.CAMERAS] == cameras


def test_camera_projection_changes_refresh_the_live_bundle(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Camera settings remain current without requiring a change to the body tree.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    camera = camera_bridge.robot.get_default_camera()
    camera.field_of_view.vertical_angle /= 2
    camera.default_camera = False

    build_live_scene(camera_bridge)

    scene = json.loads((scenes / scene_name / "scene.json").read_text())
    [published] = scene["robot"][CameraField.CAMERAS]
    assert published[CameraField.VERTICAL_ANGLE] == camera.field_of_view.vertical_angle
    assert published[CameraField.DEFAULT] == camera.default_camera


def test_malformed_cached_robot_metadata_is_rebuilt(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A malformed cached robot entry cannot hide current camera descriptors.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    scene_path = scenes / scene_name / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = scene["robot"][CameraField.CAMERAS]
    scene["robot"] = []
    scene_path.write_text(json.dumps(scene))

    build_live_scene(camera_bridge)

    assert json.loads(scene_path.read_text())["robot"][CameraField.CAMERAS] == cameras
