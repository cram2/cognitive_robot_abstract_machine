"""
Verify camera replay from a full recorded semantic world.
"""

from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass, replace
from typing_extensions import Any
from unittest.mock import Mock

import numpy as np
import pytest

from robokudo import world as rk_world
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_mongodb_playback import MongoReplayMode
from robokudo.exceptions import InvalidCameraObservation, CameraObservationMissing
from robokudo.io.cas_view_codecs import CameraObservationCodec, ViewPayload
from robokudo.io.sensor_world import RecordedCameraWorld
from robokudo.io.storage import Storage, StorageDocumentField
from robokudo.io.exceptions import RecordedViewMissing
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    FieldOfViewCameraModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% Recorded scene


@dataclass
class RecordedScene:
    """
    Test scene containing a camera and unrelated inferred geometry.
    """

    world: World
    """
    Recorded semantic world.
    """

    observation: CameraObservation
    """
    Recorded sensor observation.
    """

    source_root: Body
    """
    Recorded world root outside the direct camera pose.
    """

    inferred_body: Body
    """
    Recorded entity that replay must exclude.
    """


def _recorded_scene(
    with_pose: bool = True, pose_frame_alias: bool = False
) -> RecordedScene:
    """
    Build a connected recorded world with a camera and inferred body.
    """
    source_world = World()
    source_root = Body(name=PrefixedName(name="source_root"))
    reference = Body(name=PrefixedName(name="map"))
    camera_root = Body(
        name=PrefixedName(name="/camera_frame" if pose_frame_alias else "camera_frame")
    )
    pose_child = (
        Body(name=PrefixedName(name="camera_frame"))
        if pose_frame_alias
        else camera_root
    )
    inferred_body = Body(name=PrefixedName(name="inferred_object"))
    effective_model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=525.0,
        focal_length_y=526.0,
        principal_point=Point2(x=319.5, y=239.5),
    )
    camera = Camera(
        name=PrefixedName(name="recorded_camera"),
        root=camera_root,
        camera_model=FieldOfViewCameraModel(
            angular_extent=FieldOfView(),
            image_resolution=effective_model.resolution,
        ),
        camera_range=CameraRange(minimum_distance=0.2, maximum_distance=12.0),
        modalities=(CameraModality.COLOR, CameraModality.DEPTH),
    )
    with source_world.modify_world():
        bodies = [source_root, reference, camera_root, inferred_body]
        if pose_frame_alias:
            bodies.append(pose_child)
        for body in bodies:
            source_world.add_body(body)
        connections = [
            (source_root, reference),
            (reference, camera_root),
            (source_root, inferred_body),
        ]
        if pose_frame_alias:
            connections.append((reference, pose_child))
        for parent, child in connections:
            source_world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=parent, child=child, world=source_world
                )
            )
        source_world.add_semantic_annotation(camera)

    pose = (
        HomogeneousTransformationMatrix(
            reference_frame=reference, child_frame=pose_child
        )
        if with_pose
        else None
    )
    return RecordedScene(
        world=source_world,
        observation=CameraObservation(
            camera=camera,
            effective_camera_model=effective_model,
            world_T_camera=pose,
            timestamp_nanoseconds=123,
        ),
        source_root=source_root,
        inferred_body=inferred_body,
    )


def _snapshot(world: World) -> dict[str, Any]:
    """
    Serialize a complete world as the storage writer does.
    """
    return WorldModelSnapshot(
        modifications=list(world.get_world_model_manager().model_modification_blocks),
        ids=list(world.state.keys()),
        states=list(world.state.positions),
    ).to_json()


# %% Sensor projection


def test_sensor_world_projection_excludes_inferred_entities() -> None:
    """
    Keep camera IDs and a direct pose while omitting unrelated world branches.
    """
    scene = _recorded_scene()

    sensor_world = RecordedCameraWorld(scene.observation).project()

    camera = sensor_world.get_semantic_annotations_by_type(Camera)[0]
    assert camera.id == scene.observation.camera.id
    assert camera.root.id == scene.observation.camera.root.id
    assert (
        camera.root.parent_connection.parent.id
        == scene.observation.world_T_camera.reference_frame.id
    )
    assert sensor_world.get_bodies_by_name(scene.source_root.name) == []
    assert sensor_world.get_bodies_by_name(scene.inferred_body.name) == []


def test_full_snapshot_replays_only_camera_context() -> None:
    """
    Restore a full snapshot temporarily and keep only its sensor context.
    """
    scene = _recorded_scene()
    document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, scene.observation
    )
    recorded_world = World()
    tracker = WorldEntityWithIDKwargsTracker.from_world(recorded_world)
    WorldModelSnapshot.apply_to_json_snapshot_to_world(
        recorded_world, _snapshot(scene.world), **tracker.create_kwargs()
    )
    restored_observation = CameraObservationCodec().decode_with_tracker(
        ViewPayload.from_document(document), tracker
    )
    rk_world.init_world_with_entity_tracker()
    reasoning_world = rk_world.world_instance()
    RecordedCameraWorld(restored_observation).install(reasoning_world)
    rk_world.init_world_entity_tracker_from_world(reasoning_world)
    _, replayed = Storage.decode_view_document(document)
    RecordedCameraWorld.apply_pose(reasoning_world, replayed)

    assert replayed.camera is not scene.observation.camera
    assert replayed.camera.id == scene.observation.camera.id
    assert replayed.camera.root.id == scene.observation.camera.root.id
    assert replayed.camera.camera_model == scene.observation.camera.camera_model
    assert replayed.camera.camera_range == scene.observation.camera.camera_range
    assert tuple(replayed.camera.modalities) == scene.observation.camera.modalities
    np.testing.assert_array_equal(
        replayed.camera.viewing_direction.to_np(),
        scene.observation.camera.viewing_direction.to_np(),
    )
    assert replayed.effective_camera_model == scene.observation.effective_camera_model
    assert replayed.world_T_camera.child_frame is replayed.camera.root
    assert (
        replayed.world_T_camera.reference_frame.id
        == scene.observation.world_T_camera.reference_frame.id
    )
    assert reasoning_world.get_bodies_by_name(scene.inferred_body.name) == []
    assert reasoning_world.get_bodies_by_name(scene.source_root.name) == []


def test_replay_resolves_leading_slash_pose_frame_alias() -> None:
    """
    Replay a TF pose whose frame omits the camera root's leading slash.
    """
    scene = _recorded_scene(pose_frame_alias=True)
    document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, scene.observation
    )
    frame = {StorageDocumentField.WORLD: json.dumps(_snapshot(scene.world))}
    reader = StorageReaderInterface.__new__(StorageReaderInterface)
    reader.storage = Mock(load_view_document=Mock(return_value=document))
    reader.rk_logger = Mock()
    rk_world.init_world_with_entity_tracker()

    reader._restore_sensor_world(
        {**frame, StorageDocumentField.VIEW_IDS: {CASViews.CAMERA_OBSERVATION: 1}}
    )
    _, replayed = Storage.decode_view_document(document)
    RecordedCameraWorld.apply_pose(rk_world.world_instance(), replayed)

    assert replayed.camera.root.id == scene.observation.camera.root.id
    assert replayed.world_T_camera.child_frame.id == (
        scene.observation.world_T_camera.child_frame.id
    )
    assert replayed.world_T_camera.child_frame is not replayed.camera.root
    assert (
        replayed.camera.root.parent_connection.parent
        is replayed.world_T_camera.child_frame
    )
    np.testing.assert_allclose(
        replayed.camera.root.global_transform.to_np(),
        replayed.world_T_camera.to_np(),
    )
    assert rk_world.world_instance().get_bodies_by_name(scene.inferred_body.name) == []


@pytest.mark.parametrize("pose_frame_alias", [False, True])
def test_later_frame_updates_same_camera_connection(pose_frame_alias: bool) -> None:
    """
    Apply a new sampled pose without replacing the projected camera.
    """
    scene = _recorded_scene(pose_frame_alias=pose_frame_alias)
    rk_world.init_world_with_entity_tracker()
    reasoning_world = rk_world.world_instance()
    RecordedCameraWorld(scene.observation).install(reasoning_world)
    rk_world.init_world_entity_tracker_from_world(reasoning_world)
    first_document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, scene.observation
    )
    _, first = Storage.decode_view_document(first_document)

    moved_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.5,
        pos_y=0.0,
        pos_z=0.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        reference_frame=scene.observation.world_T_camera.reference_frame,
        child_frame=scene.observation.world_T_camera.child_frame,
    )
    second_observation = replace(
        scene.observation, world_T_camera=moved_pose, timestamp_nanoseconds=124
    )
    second_document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, second_observation
    )
    _, second = Storage.decode_view_document(second_document)
    RecordedCameraWorld.apply_pose(reasoning_world, second)

    assert second.camera is first.camera
    assert second.timestamp_nanoseconds == second_observation.timestamp_nanoseconds
    np.testing.assert_array_equal(
        second.world_T_camera.child_frame.parent_connection.origin.to_np(),
        moved_pose.to_np(),
    )


def test_missing_pose_merges_camera_into_existing_world() -> None:
    """
    Attach an unposed recorded camera without restoring recorded objects.
    """
    scene = _recorded_scene(with_pose=False)
    rk_world.init_world_with_entity_tracker()
    reasoning_world = rk_world.world_instance()
    unrelated = Body(name=PrefixedName(name="runtime_entity"))
    with reasoning_world.modify_world():
        reasoning_world.add_body(unrelated)
    RecordedCameraWorld(scene.observation).install(reasoning_world)
    rk_world.init_world_entity_tracker_from_world(reasoning_world)

    _, replayed = Storage.decode_view_document(
        Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, scene.observation)
    )

    assert replayed.world_T_camera is None
    assert replayed.camera.root.parent_connection.parent is unrelated
    assert reasoning_world.get_bodies_by_name(scene.inferred_body.name) == []


def test_camera_pose_child_must_match_camera_root() -> None:
    """
    Reject a sampled pose for a different sensor frame.
    """
    scene = _recorded_scene()
    wrong_child = Body(name=PrefixedName(name="different_camera_frame"))
    wrong_pose = HomogeneousTransformationMatrix(
        reference_frame=scene.observation.world_T_camera.reference_frame,
        child_frame=wrong_child,
    )

    with pytest.raises(InvalidCameraObservation):
        RecordedCameraWorld(
            replace(scene.observation, world_T_camera=wrong_pose)
        ).project()


def test_camera_pose_requires_a_reference_frame() -> None:
    """
    Reject a pose whose target frame cannot be represented in the sensor world.
    """
    scene = _recorded_scene()
    pose = HomogeneousTransformationMatrix(child_frame=scene.observation.camera.root)

    with pytest.raises(InvalidCameraObservation):
        RecordedCameraWorld(replace(scene.observation, world_T_camera=pose)).project()


# %% Reader behavior


def test_storage_reader_keeps_camera_identity_in_cas() -> None:
    """
    Keep the world camera instance when transferring decoded views into a CAS.
    """
    scene = _recorded_scene()
    rk_world.set_world(scene.world)
    rk_world.init_world_entity_tracker_from_world(scene.world)
    frame = {StorageDocumentField.VIEW_IDS: {CASViews.CAMERA_OBSERVATION: 1}}
    reader = StorageReaderInterface.__new__(StorageReaderInterface)
    reader._recorded_camera_id = scene.observation.camera.id
    reader.reader = Mock(get_next_frame=Mock(return_value=frame))

    def load_views(cas_frame, included_view_names):
        cas_frame[StorageDocumentField.VIEWS] = {
            CASViews.CAMERA_OBSERVATION: scene.observation,
            CASViews.DEPTH_IMAGE: np.zeros((1, 1)),
        }

    reader.storage = Mock(load_views_from_mongo_in_cas=Mock(side_effect=load_views))
    reader.camera_config = Mock(
        restore_annotations=False, replay_mode=MongoReplayMode.SENSOR_CONTEXT
    )
    cas = CAS()

    reader.set_data(cas)

    assert cas.require_camera_observation().camera is scene.observation.camera


def test_sensor_replay_skips_recorded_nonsensor_views() -> None:
    """
    Fetch only selected sensor views from a recorded frame.
    """
    storage = Storage.__new__(Storage)
    collection = Mock()
    collection.find_one.return_value = Storage.encode_view_document(
        CASViews.COLOR_IMAGE, 7
    )
    storage.db = {Storage.VIEW_COLLECTION_NAME: collection}
    frame = {
        StorageDocumentField.VIEW_IDS: {
            CASViews.COLOR_IMAGE: 1,
            CASViews.QUERY: 2,
            CASViews.CLOUD: 3,
        },
        StorageDocumentField.VIEWS: {},
    }

    storage.load_views_from_mongo_in_cas(
        frame, included_view_names=set(StorageReaderInterface.SENSOR_VIEW_NAMES)
    )

    assert frame[StorageDocumentField.VIEWS] == {CASViews.COLOR_IMAGE: 7}
    collection.find_one.assert_called_once_with({"_id": 1})


def test_explicit_full_world_replay_restores_recorded_objects() -> None:
    """
    Restore inferred entities only when full-world playback is selected.
    """
    scene = _recorded_scene()
    rk_world.init_world_with_entity_tracker()
    reader = StorageReaderInterface.__new__(StorageReaderInterface)
    reader.rk_logger = Mock()
    reader._recorded_camera_id = None
    reader.camera_config = Mock(
        restore_annotations=False, replay_mode=MongoReplayMode.FULL_WORLD
    )
    frame = {
        StorageDocumentField.WORLD: json.dumps(_snapshot(scene.world)),
        StorageDocumentField.VIEW_IDS: {CASViews.CAMERA_OBSERVATION: 1},
    }
    reader.reader = Mock(get_next_frame=Mock(return_value=frame))
    camera_document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, scene.observation
    )

    def load_views(cas_frame, included_view_names):
        assert included_view_names is None
        cas_frame[StorageDocumentField.VIEWS] = {
            CASViews.CAMERA_OBSERVATION: Storage.decode_view_document(camera_document)[
                1
            ],
            CASViews.DEPTH_IMAGE: np.zeros((1, 1)),
            CASViews.QUERY: {"recorded": True},
        }

    reader.storage = Mock(load_views_from_mongo_in_cas=Mock(side_effect=load_views))
    cas = CAS()

    reader.set_data(cas)

    assert (
        rk_world.world_instance().find_world_entity_with_id(scene.inferred_body.id)
        is not None
    )
    assert (
        cas.require_camera_observation().camera
        is rk_world.world_instance().get_semantic_annotation_by_id(
            scene.observation.camera.id
        )
    )
    assert cas.contains(CASViews.QUERY)


# %% Projection isolation


@pytest.mark.parametrize("pose_frame_alias", [False, True])
def test_sensor_projection_copies_camera_data_and_sampled_pose(
    pose_frame_alias: bool,
) -> None:
    """
    Projection preserves the sampled optical pose without sharing camera calibration.
    """
    scene = _recorded_scene(pose_frame_alias=pose_frame_alias)
    projected_world = RecordedCameraWorld(scene.observation).project()
    projected_camera = projected_world.get_semantic_annotation_by_id(
        scene.observation.camera.id
    )
    assert projected_camera is not scene.observation.camera
    assert projected_camera.root is not scene.observation.camera.root
    assert projected_camera.camera_model == scene.observation.camera.camera_model
    assert projected_camera.camera_model is not scene.observation.camera.camera_model
    np.testing.assert_allclose(
        projected_camera.root.global_transform.to_np(),
        scene.observation.world_T_camera.to_np(),
        atol=1e-12,
    )


# %% Sensor replay input validation


def test_sensor_replay_requires_recorded_camera_observation() -> None:
    """
    Sensor replay rejects a frame without a camera observation before reading storage.
    """
    reader = StorageReaderInterface.__new__(StorageReaderInterface)
    reader.storage = Mock()
    with pytest.raises(CameraObservationMissing):
        reader._restore_sensor_world({StorageDocumentField.VIEW_IDS: {}})
    reader.storage.load_view_document.assert_not_called()


@pytest.mark.parametrize("conflicting_identity", [False, True])
def test_sensor_replay_rejects_conflicting_camera_root_without_changing_world(
    conflicting_identity: bool,
) -> None:
    """
    A recorded camera cannot overwrite a runtime body sharing its identity or name.
    """
    scene = _recorded_scene()
    root = scene.observation.camera.root
    conflict = (
        Body(name=PrefixedName(name="runtime_camera"), id=root.id)
        if conflicting_identity
        else Body(name=root.name)
    )
    rk_world.init_world_with_entity_tracker()
    reasoning_world = rk_world.world_instance()
    with reasoning_world.modify_world():
        reasoning_world.add_body(conflict)
    document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, scene.observation
    )
    reader = StorageReaderInterface.__new__(StorageReaderInterface)
    reader.storage = Mock(load_view_document=Mock(return_value=document))
    reader.rk_logger = Mock()
    frame = {
        StorageDocumentField.WORLD: json.dumps(_snapshot(scene.world)),
        StorageDocumentField.VIEW_IDS: {CASViews.CAMERA_OBSERVATION: 1},
    }
    with pytest.raises(InvalidCameraObservation):
        reader._restore_sensor_world(frame)
    assert reasoning_world.bodies == [conflict]
    assert reasoning_world.get_semantic_annotations_by_type(Camera) == []


# %% Missing stored views


def test_storage_reports_missing_view_document() -> None:
    """
    A broken recorded view reference raises a structured storage error.
    """
    storage = Storage.__new__(Storage)
    collection = Mock()
    collection.find_one.return_value = None
    storage.db = {Storage.VIEW_COLLECTION_NAME: collection}
    with pytest.raises(RecordedViewMissing):
        storage.load_view_document(
            {StorageDocumentField.VIEW_IDS: {CASViews.COLOR_IMAGE: 42}},
            CASViews.COLOR_IMAGE,
        )
