"""3D pose estimation using oriented bounding boxes.

This module provides an annotator for:

* Calculating 3D poses for object hypotheses
* Using z-axis aligned oriented bounding boxes
* Supporting world frame alignment
* Generating visualization markers

The module uses:

* 2D projection for rotation estimation
* Minimum area rectangles for orientation
* Coordinate frame transformations
* Open3D visualization tools

.. note::
   By default, z-axis is aligned with world frame up direction.
"""

from __future__ import annotations
import copy
import math
from timeit import default_timer

import cv2
import numpy.linalg
import open3d as o3d
from py_trees.common import Status
from typing_extensions import Tuple, List, TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation, BoundingBox3DAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    transform_cloud_from_cam_to_world,
    get_world_to_cam_transform_matrix,
)
from robokudo.utils.transform import (
    construct_rotation_matrix,
    get_transform_matrix,
    get_quaternion_from_rotation_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class ClusterPoseBBAnnotator(BaseAnnotator):
    """3D pose estimation using oriented bounding boxes.

    This annotator:

    * Calculates 3D poses for object point clusters
    * Uses z-axis aligned oriented bounding boxes
    * Projects points to 2D for rotation estimation
    * Creates pose and bounding box annotations
    * Supports axis alignment by box extents

    .. note::
       Default behavior aligns z-axis with world frame up direction.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for pose estimation."""

        class Parameters:
            """Parameters for configuring pose estimation."""

            def __init__(self) -> None:
                self.align_x_axis_by_max_bbox_extent: bool = False
                """
                Whether to align x-axis with longest box side.
                
                The default behaviour of this Annotator orients the z-axis such that it always points up w.r.t the
                world frame. By setting this parameter to True, this behaviour will be overwritten.
                In that case, x will always point in the direction of the largest side (extent) of the bounding box
                """

                self.bounding_box_visualization_color: Tuple[int, int, int] = (0, 0, 0)
                """RGB color for box visualization, defaults to [0,0,0]"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ClusterPoseBBAnnotator",
        descriptor: "ClusterPoseBBAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the pose estimator.

        :param name: Name of this annotator instance, defaults to "ClusterPoseBBAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def adjust_bb_orientation_by_bb_size(
        self, pose_orientation: npt.NDArray, bounding_box_extents: List
    ) -> Tuple[npt.NDArray, List]:
        """Align bounding box axes with extent dimensions.

        Reorients the bounding box so that:

        * X-axis aligns with longest dimension
        * Y-axis aligns with second longest
        * Z-axis aligns with shortest dimension

        :param pose_orientation: Current orientation as 3x3 rotation matrix
        :param bounding_box_extents: Current box dimensions [x,y,z]
        :return: Tuple of (aligned orientation matrix, reordered extents)
        """

        # We want to rearrange the axes of the BB by its extents.
        # Let's create a list that contains indices of the bounding box extents, but in sorted order (descending)
        bounding_box_extents_sorted_with_idx = [
            index
            for index, number in sorted(
                enumerate(bounding_box_extents), key=lambda x: x[1], reverse=True
            )
        ]

        # Adjust BB extents such that they correspond to the new order
        new_bounding_box_extents = [
            bounding_box_extents[idx] for idx in bounding_box_extents_sorted_with_idx
        ]
        return (
            construct_rotation_matrix(
                pose_orientation, bounding_box_extents_sorted_with_idx
            ),
            new_bounding_box_extents,
        )

    def update(self) -> Status:
        """Process object hypotheses and estimate poses.

        The method:

        * Loads point cloud from CAS
        * For each object hypothesis:
          * Transforms points to world frame
          * Projects to 2D for orientation estimation
          * Calculates minimum area rectangle
          * Creates oriented bounding box
          * Generates pose and box annotations
          * Creates visualization markers

        :return: SUCCESS after processing, FAILURE if transform not found
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        geometries_to_visualize = []

        # Iterate over everything that is an Object hypothesis and calculate the centroid
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
        for object_hypothesis in object_hypotheses:
            if (
                object_hypothesis.points is None
                or len(object_hypothesis.points.points) <= 10
            ):
                continue

            # Since we need to do some transformations on the data, we'll deepcopy them first to
            # avoid altering the points in the CAS
            cluster_cloud = copy.deepcopy(object_hypothesis.points)

            try:
                cluster_cloud_in_world = transform_cloud_from_cam_to_world(
                    self.get_cas(), cluster_cloud, transform_inplace=True
                )
            except Exception as e:
                self.rk_logger.warning(
                    f"Couldn't find camera viewpoint in the CAS." f"Fail. Error: {e}"
                )
                return Status.FAILURE

            # Project cluster points to 2d (onto the world plane with z as the normal)
            min_x, min_y, min_z = numpy.asarray(cluster_cloud_in_world.points).min(
                axis=0
            )
            max_x, max_y, max_z = numpy.asarray(cluster_cloud_in_world.points).max(
                axis=0
            )

            # Calculate 2d point set by ignoring z-axis for every row in the cluster points
            # Multiply by 1000 to create image coordinates from meters
            cluster_cloud_mat_2d = (
                numpy.asarray(cluster_cloud_in_world.points)[:, 0:2] * 1000
            ).astype(numpy.int32)
            rect_center2d, rect_size, rect_angle = cv2.minAreaRect(cluster_cloud_mat_2d)

            # rect size idx 0 = width
            if rect_size[0] < rect_size[1]:
                rect_angle += 90
                new_rect_size = (rect_size[1], rect_size[0])
                rect_size = new_rect_size

            bb_size = numpy.asarray(rect_size) / 1000.0

            translation_in_world = (
                (max_x + min_x) / 2,
                (max_y + min_y) / 2,
                (max_z + min_z) / 2,
            )

            sinA = math.sin(rect_angle / 180.0 * math.pi)
            cosA = math.cos(rect_angle / 180.0 * math.pi)

            # Rotation around z (up-axis in world coordinates)
            rotation_matrix_in_world = [[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]

            # We've now got the cluster translation and rotation in world coordinates.
            # Transform back to sensor coordinates
            cluster_transform_in_world = get_transform_matrix(
                rotation_matrix_in_world, translation_in_world
            )

            world_to_cam_transform = get_world_to_cam_transform_matrix(self.get_cas())

            cluster_transform_in_cam = numpy.matmul(
                world_to_cam_transform, cluster_transform_in_world
            )
            cluster_translation_in_cam = cluster_transform_in_cam[:3, 3]
            cluster_rotation_in_cam = cluster_transform_in_cam[:3, :3]
            bounding_box_extents = [bb_size[0], bb_size[1], max_z - min_z]

            # We've finished calculating the transformation and size of the bounding box.
            # If the pose shall be aligned w.r.t the bbox extents, call the corresponding method
            if self.descriptor.parameters.align_x_axis_by_max_bbox_extent:
                aligned_orientation, aligned_extents = (
                    self.adjust_bb_orientation_by_bb_size(
                        pose_orientation=cluster_rotation_in_cam,
                        bounding_box_extents=bounding_box_extents,
                    )
                )

                aligned_transform_in_cam = get_transform_matrix(
                    rotation=aligned_orientation,
                    translation=cluster_translation_in_cam,
                )

                cluster_transform_in_cam = aligned_transform_in_cam
                cluster_translation_in_cam = cluster_transform_in_cam[:3, 3]
                cluster_rotation_in_cam = cluster_transform_in_cam[:3, :3]
                bounding_box_extents = aligned_extents

            # Draw a frame in the visualization
            cluster_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            cluster_frame.transform(cluster_transform_in_cam)
            geometries_to_visualize.append(cluster_frame)

            # Annotate the pose information
            pose_annotation = PoseAnnotation()
            pose_annotation.translation = list(cluster_translation_in_cam)
            pose_annotation.rotation = list(
                get_quaternion_from_rotation_matrix(cluster_rotation_in_cam)
            )
            pose_annotation.source = type(self).__name__
            object_hypothesis.annotations.append(pose_annotation)

            # Generate a BB from the cluster info for visualization
            cluster_obb = o3d.geometry.OrientedBoundingBox(
                center=cluster_translation_in_cam,
                R=cluster_rotation_in_cam,
                extent=numpy.array(bounding_box_extents),
            )
            cluster_obb.color = (
                self.descriptor.parameters.bounding_box_visualization_color
            )

            rk_bb = BoundingBox3DAnnotation()
            rk_bb.x_length = bounding_box_extents[0]
            rk_bb.y_length = bounding_box_extents[1]
            rk_bb.z_length = bounding_box_extents[2]
            rk_bb.pose.translation = cluster_translation_in_cam
            rk_bb.pose.rotation = pose_annotation.rotation
            rk_bb.source = type(self).__name__

            object_hypothesis.annotations.append(rk_bb)

            geometries_to_visualize.append(cluster_obb)

        # Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(geometries_to_visualize)

        self.get_annotator_output_struct().set_geometries(vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
