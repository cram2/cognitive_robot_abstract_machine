"""Point cloud cropping and visualization.

This module provides an annotator for:

* Cropping point clouds using axis-aligned bounding boxes
* Supporting both sensor and world coordinate frames
* Generating and visualizing point cloud masks
* Combining masks with color images

.. note::
   Cropping can be done in either sensor coordinates (default) or world coordinates.
"""

from __future__ import annotations
from timeit import default_timer

import cv2
import open3d as o3d
from py_trees.common import Status
from typing_extensions import Optional, TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.annotator_helper import (
    transform_cloud_from_cam_to_world,
    transform_cloud_from_world_to_cam,
)
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import get_mask_from_pointcloud

if TYPE_CHECKING:
    import numpy.typing as npt


class PointcloudCropAnnotator(BaseAnnotator):
    """Point cloud cropping using axis-aligned bounding boxes.

    Crop a subset of points from a pointcloud data based on min/max X,Y,Z values.
    The crop is either done in sensor coordinates (default) or relative to the world frame.

    .. warning::
       When using world coordinates, requires valid camera-to-world transform in CAS.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud cropping."""

        class Parameters:
            """Parameters for configuring point cloud cropping."""

            def __init__(self) -> None:
                self.min_x: float = -2.0
                """Minimum X coordinate"""

                self.min_y: float = -2.0
                """Minimum Y coordinate"""

                self.min_z: float = -9.0
                """Minimum Z coordinate"""

                self.max_x: float = 2.0
                """Maximum X coordinate"""

                self.max_y: float = 2.0
                """Maximum Y coordinate"""

                self.max_z: float = 3.0
                """Maximum Z coordinate"""

                self.relative_to_world: bool = False
                """Whether to crop the PC in camera/sensor coordinates or whether it shoudl be transformed to world coordinates beforehand."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "PointcloudCropAnnotator",
        descriptor: "PointcloudCropAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the point cloud cropper.

        :param name: Name of this annotator instance, defaults to "PointcloudCropAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.color: Optional[npt.NDArray] = None
        """A copy of the color image currently being worked with."""

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Process and crop point cloud data.

        The method:

        * Loads point cloud and camera data from CAS
        * Optionally transforms to world coordinates
        * Crops using axis-aligned bounding box
        * Generates visualization mask
        * Updates CAS with cropped cloud
        * Creates combined visualization

        :return: SUCCESS after processing
        :raises Exception: If world transform not found when needed
        """
        start_timer = default_timer()
        self.rk_logger.warning(f"{self.__class__.__name__} called for update()")

        cloud = self.get_cas().get(CASViews.CLOUD)
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        pc_cam_intrinsics = self.get_cas().get(CASViews.PC_CAM_INTRINSIC)
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)

        #
        # Lookup camera information if the crop should be done relative to the world frame
        #

        if self.descriptor.parameters.relative_to_world:
            try:
                cloud = transform_cloud_from_cam_to_world(self.get_cas(), cloud)
            except Exception as e:
                self.rk_logger.warning(
                    f"Couldn't find camera viewpoint in the CAS and relative_to_world is true. "
                    f"Fail. Error: {e}"
                )
            return Status.FAILURE

        #
        # Crop the point cloud
        #
        assert isinstance(cloud, o3d.geometry.PointCloud)

        abb = o3d.geometry.AxisAlignedBoundingBox(
            [
                self.descriptor.parameters.min_x,
                self.descriptor.parameters.min_y,
                self.descriptor.parameters.min_z,
            ],
            [
                self.descriptor.parameters.max_x,
                self.descriptor.parameters.max_y,
                self.descriptor.parameters.max_z,
            ],
        )

        cropped_cloud = cloud.crop(abb)

        # Transform cloud back to camera coordinates if it has been transformed to world before
        if self.descriptor.parameters.relative_to_world:
            cropped_cloud_transformed = transform_cloud_from_world_to_cam(
                self.get_cas(), cropped_cloud
            )
            cropped_cloud = cropped_cloud_transformed

        assert isinstance(cropped_cloud, o3d.geometry.PointCloud)

        self.get_cas().set_ref(CASViews.CLOUD, cropped_cloud)
        self.get_annotator_output_struct().set_geometries(cropped_cloud)

        mask_scale = 1.0 / color2depth_ratio[0]
        mask = get_mask_from_pointcloud(
            cropped_cloud,
            self.color,
            pc_cam_intrinsics,
            mask_scale_factor=mask_scale,
            crop_to_ref=True,
        )
        combined_mask_color = cv2.addWeighted(self.color, 0.5, mask, 0.5, 0)
        self.get_annotator_output_struct().set_image(combined_mask_color)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
