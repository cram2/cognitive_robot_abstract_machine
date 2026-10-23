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
from robokudo.exceptions import (
    StoredCameraTransformFrameMetadataMissing,
    StoredViewMissing,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.storage import Storage
from robokudo.io.camera_placement import CameraPlacement, MeasuredCameraPose
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


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

    def set_data(self, cas: CAS, world: World) -> None:
        """
        :meth:`CameraInterface.set_data` for the next stored frame, whose camera is
        placed at the pose recorded with it.
        """
        cas_frame = self.reader.get_next_frame()
        if cas_frame is None:
            self.rk_logger.debug(f"Reader has no next frame cas_frame:={cas_frame}")
            return

        # The camera transform is left out, since its stored frame references belong
        # to the recorded world, not the running one.
        cas_frame["views"] = {}
        self.storage.load_views_from_mongo_in_cas(
            cas_frame,
            world,
            excluded_view_names={CASViews.CAMERA_TO_WORLD_TRANSFORM},
        )
        recorded_pose = self._recorded_camera_pose(cas_frame, world)

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame["views"].items():
            cas.set(view_name, view_content)

        # Restore annotations
        if self.camera_config.restore_annotations:
            self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas, world)

        if cas.depth_image is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False

        # Compute the camera intrinsic from the camera info.
        # this is harder to serialize and put transparently into mongo as of today, so we'll do it here
        # Construct o3d camera intrinsics from camera info in CAS

        # TODO this can be unified with the camera intrinsic creation in the camera interface. Check
        #  ROSCameraInterface.set_o3d_camera_intrinsics_from_ros_camera_info
        #  => type_conversion.o3d_camera_intrinsics_from_ros_camera_info(camera_info):
        camera_info = cas_frame["views"].get("cam_info")
        if camera_info is not None:
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
        if recorded_pose is not None:
            CameraPlacement(world).place(cas, recorded_pose)

    def _recorded_camera_pose(
        self, cas_frame: dict, world: World
    ) -> MeasuredCameraPose | None:
        """
        The camera pose recorded with ``cas_frame``.

        A camera transform stored in a format without frame names is decoded into the
        views of ``cas_frame`` instead.

        :param cas_frame: The stored CAS frame.
        :param world: The world perception perceives into.
        :return: The recorded camera pose, or None if ``cas_frame`` has none with frame
            names.
        :raises StoredViewMissing: If the storage does not hold the recorded camera
            transform.
        """
        view_ids = cas_frame.get("view_ids", {})
        view_id = view_ids.get(CASViews.CAMERA_TO_WORLD_TRANSFORM)
        if view_id is None:
            return None

        view_document = self.storage.db[Storage.VIEW_COLLECTION_NAME].find_one(
            {"_id": view_id}
        )
        if not view_document:
            raise StoredViewMissing(
                view_name=CASViews.CAMERA_TO_WORLD_TRANSFORM, view_id=view_id
            )

        if view_document["serializer_id"] == "semdt_homogeneous_transform_v1":
            return self._decode_recorded_camera_pose(view_document)

        decoded_view_name, decoded_view_value = Storage.decode_view_document(
            view_document, world
        )
        if decoded_view_name == CASViews.CAMERA_TO_WORLD_TRANSFORM:
            cas_frame["views"][CASViews.CAMERA_TO_WORLD_TRANSFORM] = decoded_view_value
        return None

    def _decode_recorded_camera_pose(self, view_document: dict) -> MeasuredCameraPose:
        """
        :param view_document: The stored document of the camera transform.
        :return: The camera pose the document recorded, named by the frame names in its
            metadata.
        :raises StoredCameraTransformFrameMetadataMissing: If the metadata names no
            frames.
        """
        metadata = view_document.get("metadata", {})
        world_frame = metadata.get("reference_frame_name")
        camera_frame = metadata.get("child_frame_name")

        if world_frame is None or camera_frame is None:
            raise StoredCameraTransformFrameMetadataMissing()

        return MeasuredCameraPose(
            world_frame=world_frame,
            camera_frame=camera_frame,
            world_T_camera=self._decode_transform_payload_without_frames(
                view_document["payload"]
            ),
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
