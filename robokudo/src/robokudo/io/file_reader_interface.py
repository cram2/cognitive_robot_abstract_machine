"""
File-based data reading interface for RoboKudo.

This module provides interfaces for reading sensor data from the local filesystem.
It supports reading:

* RGB-D camera data (color and depth images)
* Camera calibration information
* Timestamped data sequences

The module is primarily used for:

* Unit testing with recorded data
* Simple demos and examples
* Development and debugging
"""

from __future__ import annotations

# %% Imports
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import ament_index_python.packages
import cv2
from typing_extensions import Any

from krrood.adapters.json_serializer import from_json
from robokudo.cas import CAS, CASViews
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.exceptions import InvalidFileSource
from semantic_digital_twin.datastructures.camera_model import CameraModality

# %% Recording timestamps

type RecordedTimestamp = str
"""
Timestamp text extracted from a recorded frame's filename.
"""

# %% File playback


@dataclass(init=False, eq=False)
class FileReaderInterface(CameraInterface):
    """
    A camera interface for reading stored data from the local filesystem.

    This interface reads sensor data from files in a specified directory.
    Files must follow the naming convention:
    * rk_TIMESTAMP_color_image.jpg - Color image data
    * rk_TIMESTAMP_depth_image.png - Depth image data
    * rk_TIMESTAMP_camera_observation.json - Camera observation metadata

    The 'rk_' prefix is configurable. Data suffixes use the corresponding
    :class:`robokudo.cas.CASViews` names.

    .. note::
        This interface is primarily for testing and demos. For production,
        use StorageReaderInterface instead.
    """

    frame_timestamps: list[RecordedTimestamp] = field(init=False, default_factory=list)
    """
    Timestamps in playback order.
    """

    next_frame_index: int = field(init=False, default=0)
    """
    Position of the next frame to read.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the file reader interface.

        Sets up the directory structure and loads all matching data files. Raises
        exceptions for invalid configurations or missing files.

        :param camera_config: Camera configuration object
        :raises Exception: If target directory or ROS package is invalid
        """
        super().__init__(camera_config)

        if camera_config.lookup_viewpoint:
            warnings.warn(
                "FileReaderInterface does not support live TF transform lookup. "
                "lookup_viewpoint=True will be ignored; configure "
                "static_camera_transform_enabled=True to write a transform into the CAS.",
                UserWarning,
                stacklevel=2,
            )

        self.initialized: bool = False
        """
        Whether the interface was successfully initialized.
        """
        # Compute the targeted directory by considering the target_ros_package variable
        if camera_config.target_ros_package:
            # rospack = rospkg.RosPack()
            package_path = Path(
                ament_index_python.packages.get_package_share_directory(
                    camera_config.target_ros_package
                )
            )
            target_dir_path = package_path.joinpath(camera_config.target_dir)
            if not target_dir_path.exists() or not target_dir_path.is_dir():
                raise InvalidFileSource(
                    reason=f"{str(target_dir_path)} is not existing or not a directory."
                )
            self.target_dir: str = str(target_dir_path)
            """
            Target directory path as string.
            """
            self.target_dir_path: Path = target_dir_path
            """
            Target directory as Path object.
            """
        else:
            if not camera_config.target_dir:
                raise InvalidFileSource(
                    reason="FileReaderInterface: target_ros_package AND target_dir not properly set. Check CameraConfig"
                )

            self.target_dir: str = camera_config.target_dir
            self.target_dir_path: Path = Path(self.target_dir)

        if hasattr(camera_config, "filename_prefix"):
            self.filename_prefix = camera_config.filename_prefix
        else:
            self.filename_prefix = "rk_"
            """
            Prefix for data files.

            Make sure that this matches the filename_prefix in the FileReader.
            """
        self.loaded_paths: dict[RecordedTimestamp, dict[CASViews, Path]] = dict()
        """
        Dictionary mapping timestamps to file paths.
        """
        self.loaded_data: dict[RecordedTimestamp, dict[CASViews, Any]] = dict()
        """
        Dictionary mapping timestamps to loaded data.
        """
        # Matching all files with the set prefix and the timestamp
        file_paths_found = [
            x for x in sorted(self.target_dir_path.glob(f"{self.filename_prefix}*"))
        ]

        # Insert files into self.loaded_data and self.loaded_paths, hashed by the timestamp in the filename
        file_path: Path
        for file_path in file_paths_found:
            # Lookup the timestamp from the filename. Match also by the prefix, but ignore it by having a capture
            # group for the actual timestamp with '()'
            # Additionally, capture the stored data suffix.
            regexp_result = re.search(
                rf"{self.filename_prefix}([0-9\.]+)_(.*)\.", file_path.name
            )
            if not regexp_result:
                continue
            matched_timestamp = regexp_result.groups()[0]
            matched_suffix = regexp_result.groups()[1]
            if matched_suffix not in CASViews:
                raise InvalidFileSource(
                    reason=f"Unknown data suffix {matched_suffix!r} in {file_path.name}"
                )
            matched_data_type = CASViews(matched_suffix)
            if matched_timestamp not in self.loaded_paths:
                self.loaded_paths[matched_timestamp] = dict()
                self.loaded_data[matched_timestamp] = dict()

            self.loaded_paths[matched_timestamp][matched_data_type] = file_path

            match matched_data_type:
                case CASViews.COLOR_IMAGE:
                    data = cv2.imread(str(file_path))
                    if data is None:
                        raise InvalidFileSource(
                            reason=f"OpenCV couldn't read {str(file_path)}"
                        )
                    self.loaded_data[matched_timestamp][matched_data_type] = data
                case CASViews.DEPTH_IMAGE:
                    data = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
                    if data is None:
                        raise InvalidFileSource(
                            reason=f"OpenCV couldn't read {str(file_path)}"
                        )
                    self.loaded_data[matched_timestamp][matched_data_type] = data
                case CASViews.CAMERA_OBSERVATION:
                    with open(str(file_path)) as fp:
                        camera_observation_data = json.load(fp)
                        camera_observation_data["effective_camera_model"] = from_json(
                            camera_observation_data["effective_camera_model"]
                        )
                        self.loaded_data[matched_timestamp][matched_data_type] = (
                            camera_observation_data
                        )

        self.frame_timestamps = list(self.loaded_data)
        self.next_frame_index = 0
        self.initialized = True

    def has_new_data(self) -> bool:
        if not self.initialized:
            return False
        if self.camera_config.loop and self.next_frame_index == len(
            self.frame_timestamps
        ):
            self.next_frame_index = 0
        return self.next_frame_index < len(self.frame_timestamps)

    def get_next_data(self) -> dict[CASViews, Any] | None:
        """
        Return the next recorded frame, or ``None`` after the final frame.
        """
        if self.next_frame_index >= len(self.frame_timestamps):
            return None
        timestamp = self.frame_timestamps[self.next_frame_index]
        self.next_frame_index += 1
        return self.loaded_data[timestamp]


# %% RGB-D playback


@dataclass(init=False, eq=False)
class RGBDFileReaderInterface(FileReaderInterface):
    """
    Specialized file reader interface for RGB-D camera data.

    This class extends FileReaderInterface to handle the specific case of reading RGB-D
    camera data, including color images, depth images, and camera calibration
    information.

    Inherits all instance variables from FileReaderInterface.
    """

    def set_data(self, cas: CAS) -> None:
        """
        Set the next RGB-D data frame into the CAS.

        This method:
        * Reads the next color and depth images
        * Reads the recorded camera frame, model, and timestamp
        * Applies the configured static camera transform, if enabled
        * Updates the CAS with all loaded data

        :param cas: Common Analysis Structure to update
        """
        data = self.get_next_data()
        if data is None:
            return

        cas.set(CASViews.COLOR_IMAGE, data[CASViews.COLOR_IMAGE])
        cas.set(CASViews.DEPTH_IMAGE, data[CASViews.DEPTH_IMAGE])

        recorded_observation = data[CASViews.CAMERA_OBSERVATION]
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, self.camera_config.color_to_depth_ratio)
        camera_frame = recorded_observation["camera_frame"]
        world_T_camera = self.static_world_T_camera_if_configured(camera_frame)
        self.store_camera_observation(
            cas=cas,
            camera_model=recorded_observation["effective_camera_model"],
            camera_frame=camera_frame,
            timestamp_nanoseconds=recorded_observation["timestamp_nanoseconds"],
            modalities=(CameraModality.COLOR, CameraModality.DEPTH),
            world_T_camera=world_T_camera,
        )
