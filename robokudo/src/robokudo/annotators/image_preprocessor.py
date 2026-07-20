"""Image preprocessing and point cloud generation.

This module provides an annotator for:

* Converting RGB-D camera data to point clouds
* Handling color and depth image synchronization
* Managing camera intrinsic parameters
* Providing visualization modes for debugging

.. note::
   Depth values are expected to be in millimeters.
"""

from __future__ import annotations

import copy
from enum import StrEnum
from timeit import default_timer

import cv2
import numpy
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import ColorToDepthRatioMissing
from robokudo.utils.annotator_helper import scale_camera_intrinsics
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

if TYPE_CHECKING:
    import numpy.typing as npt


class ImagePreprocessorAnnotator(BaseAnnotator):
    """RGB-D image preprocessor and point cloud generator.

    This annotator:

    * Converts RGB-D camera data to point clouds
    * Handles color/depth image synchronization
    * Manages camera intrinsic parameters
    * Provides visualization modes for debugging

    .. warning::
       Requires properly configured camera intrinsics and color-to-depth ratio.
    """

    class ViewMode(StrEnum):
        """Visualization mode enumeration."""

        COLOR = "color"
        DEPTH = "depth"

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for image preprocessing."""

        class Parameters:
            """Parameters for configuring preprocessing."""

            def __init__(self):
                self.depth_trunc: float = 9.0
                """Maximum depth distance in meters, points beyond are discarded from cloud creation"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ImagePreprocessor",
        descriptor: ImagePreprocessorAnnotator.Descriptor | None = None,
    ) -> None:
        """Initialize the image preprocessor.

        :param name: Name of this annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.color: Optional[npt.NDArray] = None
        """Optional color image currently being worked with"""

        self.depth: Optional[npt.NDArray] = None
        """Optional depth image currently being worked with"""

        self.display_mode: ImagePreprocessorAnnotator.ViewMode = self.ViewMode.COLOR
        """The display mode to use (color or depth)"""

    def update(self) -> Status:
        """Process RGB-D images and generate point cloud.

        The method:

        * Loads color and depth images from CAS
        * Synchronizes and scales images
        * Converts to Open3D format
        * Generates point cloud
        * Updates visualization based on view mode

        :return: SUCCESS after processing
        :raises ColorToDepthRatioMissing: If color-to-depth ratio is not set
        """
        start_timer = default_timer()

        self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.camera_intrinsics = copy.deepcopy(
            self.get_cas().get(CASViews.CAMERA_INTRINSIC)
        )

        if self.display_mode is self.ViewMode.DEPTH:
            self.get_annotator_output_struct().set_image(self.depth)
        else:
            self.get_annotator_output_struct().set_image(self.color)

        scale_camera_intrinsics(self)

        resized_color = None
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                self.get_cas(), self.color
            )
        except ColorToDepthRatioMissing:
            self.rk_logger.error(
                "No color to depth ratio set by your camera driver! Can't preprocess."
            )
            raise

        # o3d expects color images in RGB order
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = None
        try:
            o3d_depth = o3d.geometry.Image(
                self.depth
            )  # Please note that depth values should be in mm
        except RuntimeError:
            # Even though you might have a uint16 already, it might be required to explicitly cast to uint16
            # (Note: the byteorder field might be different after the cast to make o3d happy)
            o3d_depth = o3d.geometry.Image(self.depth.astype(numpy.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            convert_rgb_to_intensity=False,
            depth_trunc=self.descriptor.parameters.depth_trunc,
        )

        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics
        )

        self.get_cas().set(CASViews.POINTCLOUD_CAMERA_INTRINSIC, self.camera_intrinsics)

        self.get_cas().set_ref(CASViews.CLOUD, cloud)
        self.get_annotator_output_struct().set_geometries(
            cloud
        )  # the visualizer can only show non-NaN pointclouds

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def key_callback(self, key: int) -> None:
        """Handle keyboard input for view mode switching.

        :param key: ASCII value of pressed key

        Available keys:

        * '1': Switch to color view mode
        * '2': Switch to depth view mode
        """
        if key == ord("1"):
            self.display_mode = self.ViewMode.COLOR
        if key == ord("2"):
            self.display_mode = self.ViewMode.DEPTH
