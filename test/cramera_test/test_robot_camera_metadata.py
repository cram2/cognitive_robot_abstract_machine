"""
Native camera annotations survive live and recorded scene publication.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_state import WorldState

from cramera.live.bridge import Bridge
from cramera.live.live_bundle import (
    MESH_SUBDIRECTORY,
    build_live_scene,
    bundle_world_models,
)
from cramera.live.recording import RecordedFrame
from cramera.live.recording_bundle import write_recording_bundle
from cramera.recording_fields import SceneField
from cramera.robot_camera import CameraField, RobotCamera

from .test_live_bundle import use_scratch_scenes_directory

if TYPE_CHECKING:
    from pathlib import Path
    from threading import RLock

    from semantic_digital_twin.world import World


# %% native camera metadata
@pytest.fixture
def camera_robot(pr2_world_copy: World) -> PR2:
    """
    Return the native robot whose annotated cameras are being published.

    :param pr2_world_copy: Independent world containing the annotated PR2.
    :return: The world's native robot annotation.
    """
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    return robot


def test_camera_metadata_uses_native_values_and_name(camera_robot: PR2) -> None:
    """
    Preserve a named sensor's identity, frame, direction and native field of view.

    :param camera_robot: Native robot whose camera metadata is serialized.
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

    :param camera_robot: Native robot containing cameras and other sensor types.
    """
    cameras = RobotCamera.of_robot(camera_robot)

    assert [entry.camera for entry in cameras] == [
        sensor for sensor in camera_robot.all_sensors if isinstance(sensor, Camera)
    ]
    assert any(not isinstance(sensor, Camera) for sensor in camera_robot.all_sensors)


def test_a_robot_without_cameras_has_no_viewpoints(camera_robot: PR2) -> None:
    """
    Camera-like link names cannot introduce viewpoints without native annotations.

    :param camera_robot: Independent robot whose camera annotations can be removed.
    """
    camera_robot.mobile_base.torso.neck.sensors.clear()

    assert RobotCamera.of_robot(camera_robot) == []


# %% live and recorded publication
@pytest.fixture
def camera_bridge(pr2_world_copy: World) -> Bridge:
    """
    Attach a native robot world to the ordinary scene publisher.

    :param pr2_world_copy: Independent world containing the annotated PR2.
    :return: Bridge publishing that world.
    """
    bridge = Bridge()
    bridge.attach(pr2_world_copy)
    return bridge


def test_live_scene_publishes_camera_metadata(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Publish descriptors whose exact frame names resolve in the bundled URDF.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

    scene_name = build_live_scene(camera_bridge)
    scene_directory = scenes / scene_name
    scene = json.loads((scene_directory / "scene.json").read_text())

    assert scene[SceneField.ROBOT][CameraField.CAMERAS] == [
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
        for camera in scene[SceneField.ROBOT][CameraField.CAMERAS]
    )


def test_recording_preserves_camera_metadata(
    camera_bridge: Bridge, tmp_path: Path
) -> None:
    """
    Persist the camera descriptors alongside the recorded robot model.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for recording output.
    """
    frame = RecordedFrame(frames={}, base=None, objects={})

    scene = write_recording_bundle(
        camera_bridge, [frame], 30.0, tmp_path / "recording", "camera-recording"
    )

    recorded_scene = json.loads((tmp_path / "recording" / "scene.json").read_text())
    expected = [
        camera.to_payload() for camera in RobotCamera.of_robot(camera_bridge.robot)
    ]
    assert scene[SceneField.ROBOT][CameraField.CAMERAS] == expected
    assert recorded_scene[SceneField.ROBOT][CameraField.CAMERAS] == expected


def test_cached_live_scene_without_camera_metadata_is_rebuilt(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Refresh an older bundle even when the robot's body topology is unchanged.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for cached scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    scene_path = scenes / scene_name / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = scene[SceneField.ROBOT].pop(CameraField.CAMERAS)
    scene_path.write_text(json.dumps(scene))

    build_live_scene(camera_bridge)

    assert (
        json.loads(scene_path.read_text())[SceneField.ROBOT][CameraField.CAMERAS]
        == cameras
    )


def test_camera_projection_changes_refresh_the_live_bundle(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Camera projection remains current without requiring a change to the body tree.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for cached scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    camera = camera_bridge.robot.get_default_camera()
    camera.field_of_view.vertical_angle /= 2

    build_live_scene(camera_bridge)

    scene = json.loads((scenes / scene_name / "scene.json").read_text())
    [published] = scene[SceneField.ROBOT][CameraField.CAMERAS]
    assert published[CameraField.VERTICAL_ANGLE] == camera.field_of_view.vertical_angle


def test_default_camera_changes_refresh_the_live_bundle(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Refresh camera selection metadata independently of projection changes.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for cached scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    camera = camera_bridge.robot.get_default_camera()
    camera.default_camera = False

    build_live_scene(camera_bridge)

    scene = json.loads((scenes / scene_name / "scene.json").read_text())
    [published] = scene[SceneField.ROBOT][CameraField.CAMERAS]
    assert published[CameraField.DEFAULT] == camera.default_camera


def test_malformed_cached_robot_metadata_is_rebuilt(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A malformed cached robot entry cannot hide current camera descriptors.

    :param camera_bridge: Bridge attached to the native robot world.
    :param tmp_path: Isolated root for cached scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(camera_bridge)
    scene_path = scenes / scene_name / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = scene[SceneField.ROBOT][CameraField.CAMERAS]
    scene[SceneField.ROBOT] = []
    scene_path.write_text(json.dumps(scene))

    build_live_scene(camera_bridge)

    assert (
        json.loads(scene_path.read_text())[SceneField.ROBOT][CameraField.CAMERAS]
        == cameras
    )


def test_multiple_native_defaults_keep_the_robot_selection_order(
    _hsr_world_setup: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Retain the native preferred camera when several sensors carry the default flag.

    :param _hsr_world_setup: Shared native HSRB world copied before name changes.
    :param tmp_path: Isolated root for scene output.
    :param monkeypatch: Restores temporary scene configuration after the test.
    """
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    hsr_world_copy = deepcopy(_hsr_world_setup)
    [robot] = hsr_world_copy.get_semantic_annotations_by_type(HSRB)
    preferred = robot.get_default_camera()
    other_default = next(
        camera
        for camera in robot.all_sensors
        if isinstance(camera, Camera)
        and camera.default_camera
        and camera is not preferred
    )
    with hsr_world_copy.modify_world():
        preferred.update_name(PrefixedName("z-native-default"))
        other_default.update_name(PrefixedName("a-other-default"))
    bridge = Bridge()
    bridge.attach(hsr_world_copy)

    scene_name = build_live_scene(bridge)

    scene = json.loads((scenes / scene_name / "scene.json").read_text())
    defaults = [
        camera
        for camera in scene[SceneField.ROBOT][CameraField.CAMERAS]
        if camera[CameraField.DEFAULT]
    ]
    assert defaults[0][CameraField.LINK] == str(robot.get_default_camera().root.name)


# %% metadata changes advertised to attached viewers
def test_projection_changes_update_the_published_signature(
    camera_bridge: Bridge,
) -> None:
    """
    Request a viewer reload after a native camera's field of view changes.

    :param camera_bridge: Bridge attached to the native robot world.
    """
    before = camera_bridge.status()[SceneField.BUNDLE_SIGNATURE]
    camera = camera_bridge.robot.get_default_camera()
    with camera_bridge.world.state.world_lock:
        camera.field_of_view.vertical_angle /= 2

    assert camera_bridge.status()[SceneField.BUNDLE_SIGNATURE] != before


def test_default_changes_update_the_published_signature(camera_bridge: Bridge) -> None:
    """
    Request a viewer reload when the native default camera flag changes.

    :param camera_bridge: Bridge attached to the native robot world.
    """
    before = camera_bridge.status()[SceneField.BUNDLE_SIGNATURE]
    camera = camera_bridge.robot.get_default_camera()
    with camera_bridge.world.state.world_lock:
        camera.default_camera = False

    assert camera_bridge.status()[SceneField.BUNDLE_SIGNATURE] != before


def test_name_changes_update_the_published_signature(camera_bridge: Bridge) -> None:
    """
    Request a viewer reload when a camera's native name changes.

    :param camera_bridge: Bridge attached to the native robot world.
    """
    before = camera_bridge.status()[SceneField.BUNDLE_SIGNATURE]
    with camera_bridge.world.modify_world():
        camera_bridge.robot.get_default_camera().update_name(PrefixedName("inspection"))

    assert camera_bridge.status()[SceneField.BUNDLE_SIGNATURE] != before


def test_camera_motion_preserves_the_published_signature(camera_bridge: Bridge) -> None:
    """
    Keep ordinary articulated head motion on the pose stream without reloading.

    :param camera_bridge: Bridge attached to the native robot world.
    """
    before = camera_bridge.status()[SceneField.BUNDLE_SIGNATURE]
    head_pan = camera_bridge.world.get_connection_by_name(PR2Joint.HEAD_PAN)
    head_pan.position += 0.2

    assert camera_bridge.status()[SceneField.BUNDLE_SIGNATURE] == before


def test_signature_locks_the_replacement_world_after_attachment_changes(
    camera_bridge: Bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Retry under the new world's lock when attachment changes while waiting.

    :param camera_bridge: Bridge initially attached to the first native robot world.
    :param monkeypatch: Observes the signature reader choosing its first world lock.
    """
    initial_world = camera_bridge.world
    replacement_world = deepcopy(initial_world)
    selected = Event()
    world_lock = WorldState.world_lock

    def selected_world_lock(state: WorldState) -> RLock:
        """
        Signal selection while preserving the world's native lock.

        :param state: Native world state whose lock is being selected.
        :return: The unchanged native state lock.
        """
        if state is initial_world.state:
            selected.set()
        return world_lock.__get__(state, WorldState)

    initial_lock = initial_world.state.world_lock
    replacement_lock = replacement_world.state.world_lock
    monkeypatch.setattr(WorldState, "world_lock", property(selected_world_lock))

    with ThreadPoolExecutor(max_workers=1) as executor:
        with replacement_lock:
            with initial_lock:
                signature = executor.submit(camera_bridge.bundle_signature)
                assert selected.wait(timeout=10)
                camera_bridge.attach(replacement_world)
            with pytest.raises(TimeoutError):
                signature.result(timeout=0.1)
        assert signature.result(timeout=10) == camera_bridge.bundle_signature()


# %% independent camera snapshot locks
def can_lock_world(world: World) -> bool:
    """
    Check whether another operation currently holds the native state lock.

    :param world: Native world whose lock is tested from a competing thread.
    :return: Whether this thread could acquire and immediately release the lock.
    """
    lock = world.state.world_lock
    acquired = lock.acquire(blocking=False)
    if acquired:
        lock.release()
    return acquired


@pytest.mark.parametrize("recording", [False, True])
def test_export_releases_the_previous_world_before_reading_a_new_signature(
    camera_bridge: Bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording: bool,
) -> None:
    """
    An attachment change cannot make export retain one world while locking another.

    :param camera_bridge: Bridge initially attached to the first native robot world.
    :param tmp_path: Isolated root for live or recorded scene output.
    :param monkeypatch: Changes attachment at the signature-publication boundary.
    :param recording: Whether to exercise recording export instead of the live bundle.
    """
    use_scratch_scenes_directory(monkeypatch, tmp_path)
    initial_world = camera_bridge.world
    replacement_world = deepcopy(initial_world)
    bundle_signature = Bridge.bundle_signature

    def signature_after_attachment(bridge: Bridge) -> str:
        """
        Switch worlds and require the previous world to remain accessible.

        :param bridge: Bridge publishing the bundle signature.
        :return: Signature of the replacement native world.
        """
        bridge.attach(replacement_world)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(can_lock_world, initial_world).result(timeout=10)
        return bundle_signature(bridge)

    monkeypatch.setattr(Bridge, "bundle_signature", signature_after_attachment)

    if recording:
        write_recording_bundle(
            camera_bridge,
            [RecordedFrame(frames={}, base=None, objects={})],
            30.0,
            tmp_path / "recording",
            "camera-recording",
        )
    else:
        build_live_scene(camera_bridge)


def test_model_camera_metadata_holds_its_native_world_lock(
    camera_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Camera export serializes one native snapshot even without a bridge-level lock.

    :param camera_bridge: Bridge owning the robot and native world being exported.
    :param tmp_path: Isolated output directory for the bundled model.
    :param monkeypatch: Checks camera serialization against a competing native reader.
    """
    to_payload = RobotCamera.to_payload

    def locked_camera_payload(
        camera: RobotCamera,
    ) -> dict[CameraField, str | float | bool | list[float]]:
        """
        Require the camera's own world to stay locked during serialization.

        :param camera: Camera whose native fields are about to be serialized.
        :return: The camera's ordinary serialized descriptor.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert not executor.submit(
                can_lock_world, camera.camera.root._world
            ).result(timeout=10)
        return to_payload(camera)

    monkeypatch.setattr(RobotCamera, "to_payload", locked_camera_payload)

    bundle_world_models(
        camera_bridge.world,
        camera_bridge.robot,
        tmp_path,
        MESH_SUBDIRECTORY,
    )
