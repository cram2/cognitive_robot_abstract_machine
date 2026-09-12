import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.exceptions import (
    GraspClutter6DImageNotFoundError,
    GraspClutter6DObjectModelNotFoundError,
    GraspClutter6DSceneFilesMissingError,
)
from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.schema import (
    GraspClutter6DScene,
)
from semantic_digital_twin.adapters.mujoco_video_recording import MujocoVideoRecorder
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)

requires_mujoco_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "false",
    reason="Only run MuJoCo-backed tests in CI.",
)


def _write_scene(directory: Path):
    """
    Write a synthetic, minimal BOP-format scene (two frames, duplicate object ids in one
    frame, one frame with a world-to-camera transform and one without) to `directory`.
    """
    identity = np.eye(3).flatten().tolist()

    scene_camera = {
        # frame "1": no world frame, two instances of the same object id.
        "1": {
            "cam_K": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        },
        # frame "2": has a world-to-camera transform (identity rotation, offset along x).
        "2": {
            "cam_K": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
            "cam_R_w2c": identity,
            "cam_t_w2c": [500.0, 0.0, 0.0],
        },
    }
    scene_gt = {
        "1": [
            {"obj_id": 1, "cam_R_m2c": identity, "cam_t_m2c": [1000.0, 2000.0, 3000.0]},
            {"obj_id": 1, "cam_R_m2c": identity, "cam_t_m2c": [-1000.0, 0.0, 0.0]},
        ],
        "2": [
            {"obj_id": 1, "cam_R_m2c": identity, "cam_t_m2c": [0.0, 0.0, 500.0]},
        ],
    }

    (directory / "scene_camera.json").write_text(json.dumps(scene_camera))
    (directory / "scene_gt.json").write_text(json.dumps(scene_gt))


def _write_object_mesh(models_directory: Path, object_id: int = 1):
    """
    Write a synthetic object mesh in millimeter-scale units, like the real dataset's
    `.ply` files - `create_world`'s default `mesh_unit_scale` converts mm to meters, so a
    mesh already authored in meters would end up 1000x too small (a degenerate size that
    trips MuJoCo's "mass and inertia... must be larger than mjMINVAL" compile check).
    """
    models_directory.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    box.export(str(models_directory / f"obj_{object_id:06d}.ply"), file_type="ply")


def test_from_directory_missing_files_raises(tmp_path):
    with pytest.raises(GraspClutter6DSceneFilesMissingError):
        GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)


def test_from_directory_parses_frames_and_poses(tmp_path):
    _write_scene(tmp_path)
    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)

    assert {f.image_id for f in scene.frames} == {"1", "2"}
    frame_one = scene.frame("1")
    assert len(frame_one.object_poses) == 2
    assert frame_one.object_poses[0].object_id == 1
    # cam_t_m2c is in millimeters; camera_T_object's translation is already in meters.
    translation = frame_one.object_poses[0].camera_T_object.to_position()
    np.testing.assert_allclose(
        [float(translation.x), float(translation.y), float(translation.z)],
        [1.0, 2.0, 3.0],
    )
    frame_two = scene.frame("2")
    assert frame_two.camera.camera_T_world is not None
    world_translation = frame_two.camera.camera_T_world.to_position()
    np.testing.assert_allclose(
        [float(world_translation.x), float(world_translation.y), float(world_translation.z)],
        [0.5, 0.0, 0.0],
    )
    # frame "1" has no world-to-camera transform in the fixture.
    assert frame_one.camera.camera_T_world is None


def test_create_world_unknown_frame_raises(tmp_path):
    _write_scene(tmp_path)
    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    with pytest.raises(GraspClutter6DImageNotFoundError):
        scene.create_world(image_id="does-not-exist", models_directory=tmp_path)


def test_create_world_missing_model_raises(tmp_path):
    _write_scene(tmp_path)
    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    with pytest.raises(GraspClutter6DObjectModelNotFoundError):
        scene.create_world(image_id="1", models_directory=tmp_path)


def test_create_world_places_objects_relative_to_camera(tmp_path):
    _write_scene(tmp_path)
    models_directory = tmp_path / "models"
    _write_object_mesh(models_directory)

    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    world = scene.create_world(
        image_id="1", models_directory=models_directory, mesh_unit_scale=1e-3
    )

    # camera (world root, since with_world_frame defaults to False) + 2 object instances.
    assert len(world.bodies) == 3
    assert world.root.name.name == "camera"

    object_bodies = [b for b in world.bodies if b.name.name != "camera"]
    assert len(object_bodies) == 2
    # the two instances of obj_id 1 must be distinct bodies with distinct names.
    assert len({b.name.name for b in object_bodies}) == 2

    def position_of(body):
        p = body.global_pose.to_position()
        return round(float(p.x), 3), round(float(p.y), 3), round(float(p.z), 3)

    positions = {position_of(b) for b in object_bodies}
    assert (1.0, 2.0, 3.0) in positions
    assert (-1.0, 0.0, 0.0) in positions

    annotations = world.get_semantic_annotations_by_type(
        NaturalLanguageWithTypeDescription
    )
    assert len(annotations) == 2
    assert all(a.type_description == "object" for a in annotations)


def test_create_world_with_world_frame_positions_camera(tmp_path):
    _write_scene(tmp_path)
    models_directory = tmp_path / "models"
    _write_object_mesh(models_directory)

    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    world = scene.create_world(
        image_id="2", models_directory=models_directory, with_world_frame=True
    )

    assert world.root.name.name == "map"
    camera_body = next(b for b in world.bodies if b.name.name == "camera")

    # cam_R_w2c is identity and cam_t_w2c is [0.5, 0, 0] m, i.e. camera_T_map has that
    # translation; the camera's pose in the map frame is therefore its inverse: [-0.5,0,0].
    p = camera_body.global_pose.to_position()
    np.testing.assert_allclose(
        [float(p.x), float(p.y), float(p.z)], [-0.5, 0.0, 0.0], atol=1e-9
    )


def test_create_world_without_world_frame_flag_ignores_available_transform(tmp_path):
    """
    Frame "2" has a world-to-camera transform, but `with_world_frame` defaults to False,
    so the camera itself should remain the world's root.
    """
    _write_scene(tmp_path)
    models_directory = tmp_path / "models"
    _write_object_mesh(models_directory)

    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    world = scene.create_world(image_id="2", models_directory=models_directory)

    assert world.root.name.name == "camera"
    assert len(world.bodies) == 2


@requires_mujoco_ci
def test_create_world_with_world_frame_can_be_mirrored_into_mujoco(tmp_path):
    """
    Regression test: objects used to be attached to the camera body via
    `Connection6DoF` unconditionally, which is one level too deep once `with_world_frame`
    adds a separate `map` root above the camera - MuJoCo rejects a free joint (what
    Connection6DoF becomes when mirrored) that is not a direct child of the world root,
    with "free joint can only be used on top level". None of this module's other tests
    catch that, since they only build the World and never mirror it into an actual
    MuJoCo simulation.
    """
    _write_scene(tmp_path)
    models_directory = tmp_path / "models"
    _write_object_mesh(models_directory)

    scene = GraspClutter6DScene.from_directory(scene_id="000001", directory=tmp_path)
    world = scene.create_world(
        image_id="2", models_directory=models_directory, with_world_frame=True
    )
    assert world.root.name.name == "map"

    recorder = MujocoVideoRecorder(world=world)
    recorder.start()
    recorder.stop()
