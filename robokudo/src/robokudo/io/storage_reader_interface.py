"""
MongoDB storage reader interface for RoboKudo.

This module provides an interface for reading stored sensor data and annotations
from a MongoDB database. It supports:

* Reading stored RGB-D camera data
* Loading camera calibration information
* Restoring annotations and views
* Automatic cursor management
* Optional looping through stored data

The module is primarily used for:

* Replaying recorded data
* Testing and debugging pipelines
* Offline data analysis
* Visualization of stored data
"""

from __future__ import annotations

from krrood.ormatic.utils import classproperty

# %% Imports
import json
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing_extensions import Any, TYPE_CHECKING
from uuid import UUID

from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World

from robokudo import world as rk_world
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
    MongoReplayMode,
)

from robokudo.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.exceptions import (
    CameraObservationMissing,
    InvalidCameraObservation,
    UnknownMode,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.cas_view_codecs import CameraObservationCodec, ViewPayload
from robokudo.io.sensor_world import RecordedCameraWorld
from robokudo.io.storage import Storage, StorageDocumentField

if TYPE_CHECKING:
    from robokudo.types.camera import CameraObservation

# %% Snapshot restoration


@dataclass(frozen=True)
class RestoredWorld:
    """
    Capture the lookup context and duration of one snapshot restoration.
    """

    tracker: WorldEntityWithIDKwargsTracker
    """
    Entity lookup context for the restored world.
    """

    duration: timedelta
    """
    Elapsed time spent decoding and applying the snapshot.
    """


# %% Recorded camera input


class StorageReaderInterface(CameraInterface):
    """
    A camera interface for reading data from MongoDB storage.

    This interface reads sensor data and annotations that were previously stored using
    the StorageWriter annotator. It handles data deserialization and restoration of the
    Common Analysis Structure (CAS) views.
    """

    @classproperty
    def SENSOR_VIEW_NAMES(cls) -> frozenset[CASViews]:
        """
        CAS views that represent recorded sensor inputs.
        """
        return frozenset(
            {
                CASViews.COLOR_IMAGE,
                CASViews.DEPTH_IMAGE,
                CASViews.COLOR_TO_DEPTH_RATIO,
                CASViews.CAMERA_OBSERVATION,
            }
        )

    def __init__(self, camera_config: MongoCameraConfig) -> None:
        """
        Initialize the storage reader interface.

        Sets up MongoDB connection and creates a list reader for the specified database.

        :param camera_config: Configuration containing database settings
        """
        super().__init__(camera_config)

        self.storage: Storage = Storage(camera_config.db_name)
        """
        MongoDB storage interface.
        """
        self.reader: Storage.ListReader = self.storage.ListReader(camera_config.db_name)
        """
        List-based reader for MongoDB data.
        """
        self._recorded_camera_id: UUID | None = None
        """
        Camera identity selected from the first recorded world snapshot.
        """

    def _restore_sensor_world(self, cas_frame: dict[str, Any]) -> None:
        """
        Extract the recorded camera into the active reasoning world.

        :param cas_frame: Stored frame containing the world snapshot and camera view.
        :raises CameraObservationMissing: If the frame has no camera observation.
        :raises InvalidCameraObservation: If recorded sensor identities or names
            conflict.
        """
        camera_payload = self._load_camera_payload(cas_frame)
        recorded_world = World()
        restored = self._restore_world(
            cas_frame[StorageDocumentField.WORLD], recorded_world
        )
        observation = CameraObservationCodec().decode_with_tracker(
            camera_payload, restored.tracker
        )
        projection_started = perf_counter()
        reasoning_world = rk_world.world_instance()
        self._check_sensor_conflicts(observation, reasoning_world)
        RecordedCameraWorld(observation).install(reasoning_world)
        rk_world.init_world_entity_tracker_from_world(reasoning_world)
        self._recorded_camera_id = observation.camera.id
        self.rk_logger.debug(
            "Mongo sensor replay restored full world in %.6f s and projected camera in %.6f s",
            restored.duration.total_seconds(),
            perf_counter() - projection_started,
        )

    def _load_camera_payload(self, cas_frame: dict[str, Any]) -> ViewPayload:
        """
        Load the camera observation document referenced by a recorded frame.

        :param cas_frame: Stored frame containing camera view identifiers.
        :return: Encoded camera observation ready for snapshot-bound decoding.
        :raises CameraObservationMissing: If the frame has no camera observation.
        """
        if CASViews.CAMERA_OBSERVATION not in cas_frame[StorageDocumentField.VIEW_IDS]:
            raise CameraObservationMissing()
        camera_document = self.storage.load_view_document(
            cas_frame, CASViews.CAMERA_OBSERVATION
        )
        return ViewPayload.from_document(camera_document)

    @staticmethod
    def _check_sensor_conflicts(
        observation: CameraObservation, reasoning_world: World
    ) -> None:
        """
        Reject recorded sensor identities or names already present in the target world.

        :param observation: Recorded camera and direct pose frames to install.
        :param reasoning_world: Destination world whose entities must be preserved.
        :raises InvalidCameraObservation: If an existing entity would conflict.
        """
        recorded_entities = [observation.camera, observation.camera.root]
        if observation.world_T_camera is not None:
            recorded_entities.append(observation.world_T_camera.reference_frame)
            if observation.pose_child is not observation.camera.root:
                recorded_entities.append(observation.pose_child)
        if any(
            reasoning_world.find_world_entity_with_id(entity.id) is not None
            for entity in recorded_entities
        ):
            raise InvalidCameraObservation(
                reason="recorded sensor entity already exists in the reasoning world"
            )
        if any(
            reasoning_world.get_bodies_by_name(entity.name)
            for entity in recorded_entities
            if entity is not observation.camera
        ) or any(
            camera.name == observation.camera.name
            for camera in reasoning_world.get_semantic_annotations_by_type(Camera)
        ):
            raise InvalidCameraObservation(
                reason="recorded sensor name already exists in the reasoning world"
            )

    @staticmethod
    def _restore_world(snapshot: str, target_world: World) -> RestoredWorld:
        """
        Apply a recorded snapshot into the supplied world.

        :param snapshot: JSON snapshot of the recorded world.
        :param target_world: World receiving the recorded state.
        :return: Entity tracker and elapsed duration of the restoration.
        """
        started = perf_counter()
        tracker = WorldEntityWithIDKwargsTracker.from_world(target_world)
        WorldModelSnapshot.apply_to_json_snapshot_to_world(
            target_world, json.loads(snapshot), **tracker.create_kwargs()
        )
        return RestoredWorld(
            tracker=tracker, duration=timedelta(seconds=perf_counter() - started)
        )

    def _restore_full_world(self, cas_frame: dict[str, Any]) -> None:
        """
        Replace reasoning state with the complete recorded frame world.

        :param cas_frame: Recorded frame containing the complete world snapshot.
        """
        rk_world.init_world_with_entity_tracker()
        reasoning_world = rk_world.world_instance()
        restored = self._restore_world(
            cas_frame[StorageDocumentField.WORLD], reasoning_world
        )
        rk_world.init_world_entity_tracker_from_world(reasoning_world)
        self.rk_logger.debug(
            "Mongo full replay restored world in %.3f ms",
            restored.duration.total_seconds() * 1000.0,
        )

    def has_new_data(self) -> bool:
        """
        Check if more data is available to read.

        Handles looping behavior based on camera configuration and maintains cursor
        position in the data sequence.

        :return: True if more data is available, False otherwise
        """
        # Check if we have to reinitialize the cursor after we hit the end of the recorded data
        if self.camera_config.loop and not self.reader.cursor_has_frames():
            self.reader.reset_cursor()

        return self.reader.cursor_has_frames()

    def set_data(self, cas: CAS) -> None:
        """
        Update the Common Analysis Structure with data from storage.

        This method:
        * Retrieves the next frame from storage
        * Restores views and annotations
        * Updates camera intrinsics
        * Sets depth availability flag

        :param cas: Common Analysis Structure to update
        """
        cas_frame = self.reader.get_next_frame()
        if cas_frame is None:
            self.rk_logger.debug(f"Reader has no next frame cas_frame:={cas_frame}")
            return

        if self.camera_config.replay_mode == MongoReplayMode.SENSOR_CONTEXT:
            if self._recorded_camera_id is None:
                self._restore_sensor_world(cas_frame)
            included_views = set(self.SENSOR_VIEW_NAMES)
        elif self.camera_config.replay_mode == MongoReplayMode.FULL_WORLD:
            self._restore_full_world(cas_frame)
            included_views = None
        else:
            raise UnknownMode(
                mode=self.camera_config.replay_mode, context="Mongo replay"
            )
        cas_frame[StorageDocumentField.VIEWS] = {}
        self.storage.load_views_from_mongo_in_cas(
            cas_frame, included_view_names=included_views
        )

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame[StorageDocumentField.VIEWS].items():
            if view_name == CASViews.CAMERA_OBSERVATION:
                cas.camera_observation = view_content
            else:
                cas.set(view_name, view_content)
        observation = cas.require_camera_observation()
        if self.camera_config.replay_mode == MongoReplayMode.SENSOR_CONTEXT:
            if observation.camera.id != self._recorded_camera_id:
                raise InvalidCameraObservation(
                    reason="recorded camera identity changed during playback"
                )
            RecordedCameraWorld.apply_pose(rk_world.world_instance(), observation)

        # Restore annotations
        if self.camera_config.restore_annotations:
            self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas)

        if cas.depth_image is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False
