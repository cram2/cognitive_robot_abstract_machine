"""
Image preprocessing and point cloud generation.

This module provides an annotator for:

* Converting RGB-D camera data to point clouds
* Handling color and depth image synchronization
* Managing camera intrinsic parameters
* Providing visualization modes for debugging

.. note::
   Depth values are expected to be in millimeters.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum
from timeit import default_timer

import cv2
import numpy
import open3d
from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

if TYPE_CHECKING:
    import numpy.typing as npt


# %% ImagePreprocessor parameters


@dataclass(kw_only=True)
class ImagePreprocessorParameters(AnnotatorParameters):
    """Parameters for configuring preprocessing."""

    depth_truncation: float = 9.0
    """Maximum depth distance in meters, points beyond are discarded from cloud creation"""


# %% ImagePreprocessor descriptor


@dataclass(kw_only=True)
class ImagePreprocessorDescriptor(AnnotatorDescriptor[ImagePreprocessorParameters]):
    """Declarative configuration of :class:`ImagePreprocessorAnnotator`."""

    parameters: ImagePreprocessorParameters = field(
        default_factory=ImagePreprocessorParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% ImagePreprocessor annotator


@dataclass(eq=False)
class ImagePreprocessorAnnotator(BaseAnnotator[ImagePreprocessorDescriptor]):
    """
    RGB-D image preprocessor and point cloud generator.

    This annotator:

    * Converts RGB-D camera data to point clouds
    * Handles color/depth image synchronization
    * Manages camera intrinsic parameters
    * Provides visualization modes for debugging

    .. warning::
       Requires properly configured camera intrinsics and color-to-depth ratio.
    """

    name: str = "ImagePreprocessor"
    """Name of this node in the behaviour tree."""

    descriptor: ImagePreprocessorDescriptor = field(
        default_factory=ImagePreprocessorDescriptor
    )
    """Configuration supplied to this annotator instance."""

    color: npt.NDArray | None = field(default=None, init=False, repr=False)
    """Optional color image currently being worked with"""

    depth: npt.NDArray | None = field(default=None, init=False, repr=False)
    """Optional depth image currently being worked with"""

    display_mode: ImagePreprocessorAnnotator.ViewMode = field(init=False, repr=False)
    """The display mode to use (color or depth)"""

    class ViewMode(StrEnum):
        """
        Visualization mode enumeration.
        """

        COLOR = "color"
        DEPTH = "depth"

    @classproperty
    def Descriptor(cls) -> type[ImagePreprocessorDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return ImagePreprocessorDescriptor

    def __post_init__(self) -> None:
        """Initialize annotator resources after :meth:`robokudo.annotators.core.BaseAnnotator.__post_init__`."""
        super().__post_init__()
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.display_mode: ImagePreprocessorAnnotator.ViewMode = self.ViewMode.COLOR

    def update(self) -> Status:
        """
        Process RGB-D images and generate point cloud.

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
        cas = self.get_cas()
        color_to_depth_ratio = cas.color_to_depth_ratio
        pointcloud_camera_model = (
            cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            )
        )
        self.camera_intrinsics = Open3DCameraModelAdapter.to_intrinsic(
            pointcloud_camera_model
        )

        if self.display_mode is self.ViewMode.DEPTH:
            self.get_annotator_output_struct().set_image(self.depth)
        else:
            self.get_annotator_output_struct().set_image(self.color)

        resized_color = None
        resized_color = get_scaled_color_image_for_depth_image(
            self.get_cas(), self.color
        )

        # open3d expects color images in RGB order
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        open3d_color_image = open3d.geometry.Image(color_rgb)
        native_depth = numpy.ascontiguousarray(
            self.depth, dtype=self.depth.dtype.newbyteorder("=")
        )
        open3d_depth_image = open3d.geometry.Image(native_depth)
        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d_color_image,
            open3d_depth_image,
            convert_rgb_to_intensity=False,
            depth_trunc=self.descriptor.parameters.depth_truncation,
        )

        cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics
        )

        cas.pointcloud_camera_model = pointcloud_camera_model

        self.get_cas().set_ref(CASViews.CLOUD, cloud)
        self.get_annotator_output_struct().set_geometries(
            cloud
        )  # the visualizer can only show non-NaN pointclouds

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def key_callback(self, key: int) -> None:
        """
        Handle keyboard input for view mode switching.

        :param key: ASCII value of pressed key

        Available keys:

        * '1': Switch to color view mode
        * '2': Switch to depth view mode
        """
        if key == ord("1"):
            self.display_mode = self.ViewMode.COLOR
        if key == ord("2"):
            self.display_mode = self.ViewMode.DEPTH
