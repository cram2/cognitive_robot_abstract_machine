"""Sensor data storage to filesystem.

This module provides an annotator for:

* Storing sensor data to local filesystem
* Supporting color and depth images
* Saving camera calibration info
* Using consistent naming schemes

The module uses:

* OpenCV image writing
* JSON serialization
* Timestamp-based filenames
* Configurable storage paths

.. note::
   Designed to work with robokudo.io.file_reader_interface.
"""

import json
from pathlib import Path
from timeit import default_timer

import cv2
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.type_conversion import ros_cam_info_to_dict


class FileWriter(BaseAnnotator):
    """Sensor data storage to filesystem.

    This module provides methods to store sensor data into the local filesystem.
    This way you can work on recorded data without having to use bag files.
    It is intended to be used with robokudo.io.file_reader_interface

    .. note::
       Target directory must exist and be writable.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for file writer."""

        class Parameters:
            """Parameters for configuring file storage.

            .. note::
               filename_prefix should match FileReader configuration.
            """

            def __init__(self) -> None:
                self.target_dir: str = "/tmp/"
                """Directory to store files in"""

                self.filename_prefix: str = "rk_"
                """Prefix for generated filenames, make sure that this matches the FileReader filename_prefix"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "FileWriter",
        descriptor: "FileWriter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the file writer.

        :param name: Name of this annotator instance, defaults to "FileWriter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.initialized = False

        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.rk_logger.debug(
            f"Writing files to {self.descriptor.parameters.target_dir}"
        )
        self.target_dir_path: Path = Path(self.descriptor.parameters.target_dir)

        if self.target_dir_path.exists() and self.target_dir_path.is_dir():
            self.initialized = True
        else:
            self.rk_logger.error(
                f"{self.descriptor.parameters.target_dir} is not existing or not a directory"
            )

    def generate_full_file_path_(self, cas_view: str, file_extension: str) -> str:
        """Generate the full filename and path where data should be stored.

        Creates a filepath using:

        * Configured prefix
        * Current timestamp
        * CAS view name
        * File extension

        :param cas_view: One of the definitions from CASView.X
        :param file_extension: File extension without dot (e.g. "jpg")
        :return: A string with the full path according to our naming scheme.
        """
        fn_prefix = self.descriptor.parameters.filename_prefix
        timestamp = self.get_cas().timestamp
        return str(
            self.target_dir_path.joinpath(
                f"{fn_prefix}{str(timestamp)}_{cas_view}.{file_extension}"
            )
        )

    def update(self) -> Status:
        """Process and store sensor data.

        The method:

        * Checks initialization status
        * Loads color image, depth image and camera info
        * Writes color image as JPG
        * Writes depth image as PNG
        * Saves camera info as JSON

        :return: SUCCESS after storing, FAILURE if not initialized
        """
        start_timer = default_timer()

        if not self.initialized:
            print(
                f"FileWriter has not been properly instantiated. Check error log for __init__ errors."
            )
            return Status.FAILURE

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        cam_info = self.get_cas().get(CASViews.CAM_INFO)

        cv2.imwrite(
            self.generate_full_file_path_(
                cas_view=CASViews.COLOR_IMAGE, file_extension="jpg"
            ),
            color,
        )
        cv2.imwrite(
            self.generate_full_file_path_(
                cas_view=CASViews.DEPTH_IMAGE, file_extension="png"
            ),
            depth,
        )

        cam_info_dict = ros_cam_info_to_dict(cam_info)
        with open(
            str(
                self.generate_full_file_path_(
                    cas_view=CASViews.CAM_INFO, file_extension="json"
                )
            ),
            "w",
        ) as fp:
            json.dump(cam_info_dict, fp)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
