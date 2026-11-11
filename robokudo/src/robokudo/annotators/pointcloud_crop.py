"""
Point cloud cropping and visualization.

This module provides an annotator for:

* Cropping point clouds using axis-aligned bounding boxes
* Supporting both sensor and world coordinate frames
* Generating and visualizing point cloud masks
* Combining masks with color images

.. note::
   Cropping can be done in either sensor coordinates (default) or world coordinates.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from timeit import default_timer

import cv2
import open3d
from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.utils.annotator_helper import (
    transform_cloud_from_camera_to_world,
    transform_cloud_from_world_to_camera,
)
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import get_mask_from_pointcloud

if TYPE_CHECKING:
    import numpy.typing as npt


# %% PointcloudCrop parameters


@dataclass(kw_only=True)
class PointcloudCropParameters(AnnotatorParameters):
    """Parameters for configuring point cloud cropping."""

    minimum_x: float = -2.0
    """Minimum X coordinate"""

    minimum_y: float = -2.0
    """Minimum Y coordinate"""

    minimum_z: float = -9.0
    """Minimum Z coordinate"""

    maximum_x: float = 2.0
    """Maximum X coordinate"""

    maximum_y: float = 2.0
    """Maximum Y coordinate"""

    maximum_z: float = 3.0
    """Maximum Z coordinate"""

    relative_to_world: bool = False
    """Whether to crop the PC in camera/sensor coordinates or whether it shoudl be transformed to world coordinates beforehand."""


# %% PointcloudCrop descriptor


@dataclass(kw_only=True)
class PointcloudCropDescriptor(AnnotatorDescriptor[PointcloudCropParameters]):
    """Declarative configuration of :class:`PointcloudCropAnnotator`."""

    parameters: PointcloudCropParameters = field(
        default_factory=PointcloudCropParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% PointcloudCrop annotator


class PointcloudCropAnnotator(BaseAnnotator[PointcloudCropDescriptor]):
    """
    Point cloud cropping using axis-aligned bounding boxes.

    Crop a subset of points from a pointcloud data based on min/max X,Y,Z values.
    The crop is either done in sensor coordinates (default) or relative to the world frame.

    .. warning::
       When using world coordinates, requires valid camera-to-world transform in CAS.
    """

    @classproperty
    def Descriptor(cls) -> type[PointcloudCropDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return PointcloudCropDescriptor

    def __init__(
        self,
        name: str = "PointcloudCropAnnotator",
        descriptor: PointcloudCropDescriptor | None = None,
    ) -> None:
        """
        Initialize the point cloud cropper.

        :param name: Name of this annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.color: npt.NDArray | None = None
        """
        A copy of the color image currently being worked with.
        """

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Crop the cloud in the selected frame and publish its visualization.

        :return: Processing status, with failures recorded on the blackboard.
        """
        start_timer = default_timer()
        self.rk_logger.warning(f"{self.__class__.__name__} called for update()")

        cloud = self.get_cas().get(CASViews.CLOUD)
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        pointcloud_camera_model = self.get_cas().pointcloud_camera_model
        color_to_depth_ratio = self.get_cas().color_to_depth_ratio

        if self.descriptor.parameters.relative_to_world:
            cloud = transform_cloud_from_camera_to_world(self.get_cas(), cloud)

        bounding_box = open3d.geometry.AxisAlignedBoundingBox(
            [
                self.descriptor.parameters.minimum_x,
                self.descriptor.parameters.minimum_y,
                self.descriptor.parameters.minimum_z,
            ],
            [
                self.descriptor.parameters.maximum_x,
                self.descriptor.parameters.maximum_y,
                self.descriptor.parameters.maximum_z,
            ],
        )

        cropped_cloud = cloud.crop(bounding_box)

        if self.descriptor.parameters.relative_to_world:
            cropped_cloud = transform_cloud_from_world_to_camera(
                self.get_cas(), cropped_cloud
            )

        self.get_cas().set_ref(CASViews.CLOUD, cropped_cloud)
        self.get_annotator_output_struct().set_geometries(cropped_cloud)

        mask_scale = 1.0 / color_to_depth_ratio.horizontal
        mask = get_mask_from_pointcloud(
            cropped_cloud,
            self.color,
            pointcloud_camera_model,
            mask_scale_factor=mask_scale,
            crop_to_ref=True,
        )
        combined_mask_color = cv2.addWeighted(self.color, 0.5, mask, 0.5, 0)
        self.get_annotator_output_struct().set_image(combined_mask_color)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
