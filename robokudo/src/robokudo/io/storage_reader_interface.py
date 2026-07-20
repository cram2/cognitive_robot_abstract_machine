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

from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)

import open3d as o3d
from robokudo.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import StoredCameraTransformFrameMetadataMissing
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.storage import Storage
import robokudo.world as world
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


class StorageReaderInterface(CameraInterface):
    """
    A camera interface for reading data from MongoDB storage.

    This interface reads sensor data and annotations that were previously stored using
    the StorageWriter annotator. It handles data deserialization and restoration of the
    Common Analysis Structure (CAS) views.
    """

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

        # Restore the views from the individual documents. The camera transform is
        # rebound explicitly below because its stored frame references belong to
        # the recorded world, not the currently running world.
        cas_frame["views"] = {}
        self.storage.load_views_from_mongo_in_cas(
            cas_frame,
            excluded_view_names={CASViews.CAMERA_TO_WORLD_TRANSFORM},
        )
        self._restore_camera_to_world_transform(cas_frame)

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame["views"].items():
            cas.set(view_name, view_content)

        # Restore annotations
        if self.camera_config.restore_annotations:
            self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas)

        if cas.depth_image is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False

        # Compute the camera intrinsic from the camera info.
        # this is harder to serialize and put transparently into mongo as of today, so we'll do it here
        # Construct o3d camera intrinsics from camera info in CAS

        # TODO this can be unified with the camera intrinsic creation in the camera interface. Check
        #  ROSCameraInterface.set_o3d_camera_intrinsics_from_ros_camera_info
        #  => type_conversion.o3d_camera_intrinsics_from_ros_camera_info(camera_info):
        if "cam_info" in cas_frame["views"]:
            camera_info = cas_frame["views"]["cam_info"]

            if camera_info is None:
                # nothing to do
                return

            width = camera_info.width
            height = 960  # we assume that the kinect right now only outputs 1280x... images with the 4:3 crop
            fx = camera_info.k[0]
            cx = camera_info.k[2]
            fy = camera_info.k[4]
            cy = camera_info.k[5]
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, fx, fy, cx, cy
            )
            cas.set(CASViews.CAMERA_INTRINSIC, camera_intrinsic)

    def _restore_camera_to_world_transform(self, cas_frame: dict) -> None:
        """
        Restore the stored camera pose as a transform bound to the running world.

        The serialized ``CAMERA_TO_WORLD_TRANSFORM`` may reference bodies from the
        recorded world. This method decodes the numeric transform, resolves the intended
        frame names, creates/updates the matching bodies in the current RoboKudo world,
        and stores the rebound transform in ``cas_frame["views"]``.
        """
        view_ids = cas_frame.get("view_ids", {})
        view_id = view_ids.get(CASViews.CAMERA_TO_WORLD_TRANSFORM)
        if view_id is None:
            return

        view_document = self.storage.db[Storage.VIEW_COLLECTION_NAME].find_one(
            {"_id": view_id}
        )
        if not view_document:
            raise RuntimeError(
                f"Couldn't find view '{CASViews.CAMERA_TO_WORLD_TRANSFORM}' with id={view_id}."
            )

        # Non-SemDT transform payloads are left to the regular codec path.
        if view_document["serializer_id"] != "semdt_homogeneous_transform_v1":
            decoded_view_name, decoded_view_value = Storage.decode_view_document(
                view_document
            )
            if decoded_view_name == CASViews.CAMERA_TO_WORLD_TRANSFORM:
                cas_frame["views"][
                    CASViews.CAMERA_TO_WORLD_TRANSFORM
                ] = decoded_view_value
            return

        stored_transform, world_frame, camera_frame = (
            self._decode_stored_camera_to_world_transform(view_document, cas_frame)
        )
        rebound_transform = self._rebind_camera_to_world_transform(
            stored_transform=stored_transform,
            world_frame=world_frame,
            camera_frame=camera_frame,
        )
        cas_frame["views"][CASViews.CAMERA_TO_WORLD_TRANSFORM] = rebound_transform

    def _decode_stored_camera_to_world_transform(
        self, view_document: dict, cas_frame: dict
    ) -> tuple[HomogeneousTransformationMatrix, str, str]:
        """
        Decode a stored camera transform and determine its frame names.

        Recordings must carry frame names as view metadata. The returned transform
        contains only the numeric pose and must be rebound before use.
        """
        payload = view_document["payload"]
        metadata = view_document.get("metadata", {})
        world_frame = metadata.get("reference_frame_name")
        camera_frame = metadata.get("child_frame_name")

        if world_frame is None or camera_frame is None:
            raise StoredCameraTransformFrameMetadataMissing()

        return (
            self._decode_transform_payload_without_frames(payload),
            world_frame,
            camera_frame,
        )

    @staticmethod
    def _decode_transform_payload_without_frames(
        payload: dict,
    ) -> HomogeneousTransformationMatrix:
        """
        Decode only the numeric transform, ignoring serialized frame UUIDs.
        """
        payload_without_frames = dict(payload)
        payload_without_frames.pop("reference_frame_id", None)
        payload_without_frames.pop("child_frame_id", None)
        return HomogeneousTransformationMatrix.from_json(payload_without_frames)

    @staticmethod
    def _rebind_camera_to_world_transform(
        stored_transform: HomogeneousTransformationMatrix,
        world_frame: str,
        camera_frame: str,
    ) -> HomogeneousTransformationMatrix:
        """
        Create a camera transform whose frames belong to the running world.

        The numeric pose is copied from ``stored_transform``. The reference and child
        frames are looked up or created in the current global RoboKudo world, and the
        corresponding world connection origin is updated.
        """
        world.setup_world_for_camera_frame(
            world_frame=world_frame,
            camera_frame=camera_frame,
        )
        runtime_world = world.world_instance()
        world_body = runtime_world.get_body_by_name(world_frame)
        camera_body = runtime_world.get_body_by_name(camera_frame)

        rebound_transform = HomogeneousTransformationMatrix(
            data=stored_transform.to_np(),
            reference_frame=world_body,
            child_frame=camera_body,
        )
        world.update_connection_transform(
            to_name=world_body.name,
            from_name=camera_body.name,
            transform=rebound_transform,
        )
        return rebound_transform
