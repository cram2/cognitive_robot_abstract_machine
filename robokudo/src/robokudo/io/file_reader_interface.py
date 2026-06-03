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

import pathlib
import re
import json

import cv2
import ament_index_python.packages

from typing_extensions import Any, Dict, List, Optional, TypeVar

from pathlib import Path

from robokudo.cas import CASViews, CAS
from robokudo.io.camera_interface import CameraInterface
from robokudo.utils.type_conversion import (
    ros_cam_info_from_dict,
    o3d_cam_intrinsics_from_ros_cam_info,
)

T = TypeVar("T")


class FileReaderInterface(CameraInterface):
    """
    A camera interface for reading stored data from the local filesystem.

    This interface reads sensor data from files in a specified directory.
    Files must follow the naming convention:
    * rk_TIMESTAMP_color_image.jpg - Color image data
    * rk_TIMESTAMP_depth_image.png - Depth image data
    * rk_TIMESTAMP_cam_info.json - Camera calibration data

    The 'rk_' prefix is configurable. Data types in filenames must match
    CASViews definitions.

    .. note::
        This interface is primarily for testing and demos. For production,
        use StorageReaderInterface instead.
    """

    class DictIteratorReader:
        """Helper class for iterating over dictionary data in sequence.

        This class provides cursor-based iteration over dictionary data,
        maintaining the sequence order and supporting reset operations.
        """

        def __init__(
            self,
            data: Optional[Dict[str, Dict[str, Any]]] = None,
            data_sequence: Optional[List[str]] = None,
        ):
            """Initialize the dictionary iterator.

            :param data: Dictionary containing the data, defaults to empty dict
            :param data_sequence: List defining iteration order
            """

            self.index: Optional[int] = None
            """Current position in the sequence"""

            self.data: Dict[str, Dict[str, Any]] = data if data else dict()
            """Dictionary containing the data"""

            self.data_sequence: List[str] = data_sequence if data_sequence else list()
            """List defining iteration order"""

            assert len(self.data_sequence) == len(self.data)

            self.reset_cursor()

        def reset_cursor(self) -> None:
            """Reset the iterator to the beginning of the sequence.

            Sets index to 0 if there is data, None otherwise.
            """
            if len(self.data_sequence) > 0:
                self.index = 0
            else:
                self.index = None

        def cursor_has_data(self) -> bool:
            """Check if there is more data to read.

            :return: True if more data is available, False otherwise
            """
            if self.index is None:
                return False

            return self.index < len(self.data_sequence)

        def get_next_data(self) -> Optional[Dict]:
            """Get the next data item in the sequence.

            :return: Next data item if available, None otherwise
            """
            if not self.cursor_has_data():
                return None

            data = self.data[self.data_sequence[self.index]]
            self.index += 1

            return data

    def __init__(self, camera_config: Any) -> None:
        """Initialize the file reader interface.

        Sets up the directory structure and loads all matching data files.
        Raises exceptions for invalid configurations or missing files.

        :param camera_config: Camera configuration object
        :raises Exception: If target directory or ROS package is invalid
        """
        super().__init__(camera_config)

        self.initialized: bool = False
        """Whether the interface was successfully initialized"""

        # Compute the targeted directory by considering the target_ros_package variable
        if camera_config.target_ros_package:
            # rospack = rospkg.RosPack()
            try:
                package_path = Path(
                    ament_index_python.packages.get_package_share_directory(
                        camera_config.target_ros_package
                    )
                )
            except ament_index_python.packages.PackageNotFoundError:
                raise Exception(
                    f"FileReaderInterface: ROS package with name '{camera_config.target_ros_package}' does not exist. "
                    f"Please adjust your camera config."
                )
            else:
                target_dir_path = package_path.joinpath(camera_config.target_dir)
                if not target_dir_path.exists() or not target_dir_path.is_dir():
                    raise Exception(
                        f"{str(target_dir_path)} is not existing or not a directory."
                    )

                self.target_dir: str = str(target_dir_path)
                """Target directory path as string"""

                self.target_dir_path: pathlib.PurePath = target_dir_path
                """Target directory as Path object"""
        else:
            if not camera_config.target_dir:
                raise Exception(
                    f"FileReaderInterface: target_ros_package AND target_dir not properly set. Check CameraConfig"
                )

            self.target_dir: str = camera_config.target_dir
            self.target_dir_path: Path = Path(self.target_dir)

        if hasattr(camera_config, "filename_prefix"):
            self.filename_prefix = camera_config.filename_prefix
        else:
            self.filename_prefix = "rk_"
            """Prefix for data files. Make sure that this matches the filename_prefix in the FileReader."""

        self.loaded_paths: Dict[str, Dict[str, Path]] = dict()
        """Dictionary mapping timestamps to file paths"""

        self.loaded_data: Dict[str, Dict[str, Any]] = dict()
        """Dictionary mapping timestamps to loaded data"""

        # Matching all files with the set prefix and the timestamp
        file_paths_found = [
            x for x in sorted(self.target_dir_path.glob(f"{self.filename_prefix}*"))
        ]

        # Insert files into self.loaded_data and self.loaded_paths, hashed by the timestamp in the filename
        file_path: Path
        for file_path in file_paths_found:
            # Lookup the timestamp from the filename. Match also by the prefix, but ignore it by having a capture
            # group for the actual timestamp with '()'
            # Additionally, capture the 'type' of the data (e.g. color, depth, cam_info, etc.)
            regexp_result = re.search(
                rf"{self.filename_prefix}([0-9\.]+)_(.*)\.", file_path.name
            )
            if not regexp_result:
                continue
            matched_timestamp = regexp_result.groups()[0]
            matched_data_type = regexp_result.groups()[1]
            if matched_timestamp not in self.loaded_paths:
                self.loaded_paths[matched_timestamp] = dict()
                self.loaded_data[matched_timestamp] = dict()

            self.loaded_paths[matched_timestamp][matched_data_type] = file_path

            # Load the actual content
            # data is named by the CASViews during recording time. So mathc against these
            if matched_data_type == CASViews.COLOR_IMAGE:
                data = cv2.imread(str(file_path))
                if data is None:
                    raise Exception(f"OpenCV couldn't read {str(file_path)}")
                self.loaded_data[matched_timestamp][matched_data_type] = data
            elif matched_data_type == CASViews.DEPTH_IMAGE:
                data = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
                if data is None:
                    raise Exception(f"OpenCV couldn't read {str(file_path)}")
                self.loaded_data[matched_timestamp][matched_data_type] = data
            elif matched_data_type == CASViews.CAM_INFO:
                with open(str(file_path)) as fp:
                    cam_info_json = json.load(fp)
                    self.loaded_data[matched_timestamp][matched_data_type] = (
                        ros_cam_info_from_dict(cam_info_json)
                    )

        # Initialize the main datastructure that we use to access the data
        # iteratively and to be able to peek into it for checking if data is available
        self.data_reader = FileReaderInterface.DictIteratorReader(
            data=self.loaded_data, data_sequence=list(iter(self.loaded_data))
        )
        """Iterator for accessing loaded data"""

        self.initialized = True

    def has_new_data(self) -> bool:
        """Check if new data is available to read.

        Handles looping behavior based on camera configuration and
        maintains cursor position in the data sequence.

        :return: True if new data is available, False otherwise
        """
        if not self.initialized:
            return False

        # Check if we have to reinitialize the cursor after we hit the end of the recorded data
        if self.camera_config.loop and not self.data_reader.cursor_has_data():
            self.data_reader.reset_cursor()

        return self.data_reader.cursor_has_data()


class RGBDFileReaderInterface(FileReaderInterface):
    """Specialized file reader interface for RGB-D camera data.

    This class extends FileReaderInterface to handle the specific case of
    reading RGB-D camera data, including color images, depth images, and
    camera calibration information.

    Inherits all instance variables from FileReaderInterface.
    """

    def set_data(self, cas: CAS) -> None:
        """Set the next RGB-D data frame into the CAS.

        This method:
        * Reads the next color and depth images
        * Applies any necessary fixes (e.g., Kinect height fix)
        * Sets camera calibration and transformation data
        * Updates the CAS with all loaded data

        :param cas: Common Analysis Structure to update
        """
        data = self.data_reader.get_next_data()

        cas.set(CASViews.COLOR_IMAGE, data[CASViews.COLOR_IMAGE])
        cas.set(CASViews.DEPTH_IMAGE, data[CASViews.DEPTH_IMAGE])

        cam_info = data[CASViews.CAM_INFO]
        if self.camera_config.kinect_height_fix_mode:
            cam_info.height = 960  # Kinect hack ...
        cas.set(CASViews.CAM_INFO, cam_info)

        cas.set(
            CASViews.CAM_INTRINSIC,
            o3d_cam_intrinsics_from_ros_cam_info(data[CASViews.CAM_INFO]),
        )
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.camera_config.color2depth_ratio)
