import copy
from timeit import default_timer

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.cv import BoundingBox3D
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    transform_cloud_from_cam_to_world,
    get_world_to_cam_transform_matrix,
)
from robokudo.utils.transform import get_transform_matrix

"""
Robokudo Size Annotator on Bounding Boxes 3D

:class:`SizeBBAnnotator` performs an analysis of the 3D point cloud from any object hypothesis to 
derive the size of the objects from 3D oriented bounding boxes and annotates
the information found for later processing steps.

Processing steps:
----------------

* The object's point cluster is copied and transformed into world coordinates.
* The minimum enclosing box (bounding box) is created on the 2D projections of 
  the points to determine the object's size and rotation.
* The transformations of the minimum enclosing box are converted to camera coordinates.
* A 3D oriented bounding box is generated based on the transformations.
* Compute sizes along minimum and maximum bounds of newly constructed oriented 3D bounding box.
* The size of the bounding box is stored in the object's annotations.
"""


class SizeBBAnnotator(BaseAnnotator):
    """A class to analyze 3D point clouds and compute oriented bounding box sizes.

    This annotator processes object hypotheses by computing their oriented 3D bounding boxes
    and storing size information as annotations. It performs coordinate transformations and
    uses minimum area rectangles to determine object dimensions.
    """

    def __init__(self, name: str = "SizeBBAnnotator") -> None:
        """Default construction. Minimal one-time init!"""
        super().__init__(name)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """Process object hypotheses to compute and annotate their 3D bounding box sizes.

        For each object hypothesis with sufficient points:

        * Transforms point cloud to world coordinates
        * Projects points to 2D and finds minimum area rectangle
        * Computes 3D oriented bounding box
        * Annotates size information
        * Updates visualization

        :return: SUCCESS if processing completed, FAILURE if camera viewpoint not found
        :raises Exception: If camera viewpoint transform cannot be found in CAS
        """
        start_timer = default_timer()
        vis_geometries = [self.get_cas().get_copy(CASViews.CLOUD)]
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        for oh in object_hypotheses:
            if not isinstance(oh, ObjectHypothesis):
                continue
            if oh.points is None or len(oh.points.points) <= 10:
                continue

            pcd = oh.points
            cluster_cloud = copy.deepcopy(pcd)
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
            min_x, min_y, min_z = np.asarray(cluster_cloud_in_world.points).min(axis=0)
            max_x, max_y, max_z = np.asarray(cluster_cloud_in_world.points).max(axis=0)

            # Calculate 2d point set by ignoring z-axis for every row in the cluster points
            # Multiply by 1000 to create image coordinates from meters
            cluster_cloud_mat_2d = (
                np.asarray(cluster_cloud_in_world.points)[:, 0:2] * 1000
            ).astype(np.int32)
            rect_center2d, rect_size, rect_angle = cv2.minAreaRect(cluster_cloud_mat_2d)

            # rect size idx 0 = width
            if rect_size[0] < rect_size[1]:
                rect_angle += 90
                new_rect_size = (rect_size[1], rect_size[0])
                rect_size = new_rect_size

            bb_size = np.asarray(rect_size) / 1000.0

            translation_in_world = (
                (max_x + min_x) / 2,
                (max_y + min_y) / 2,
                (max_z + min_z) / 2,
            )

            sin_a = np.sin(rect_angle / 180.0 * np.pi)
            cos_a = np.cos(rect_angle / 180.0 * np.pi)

            # Rotation around z (up-axis in world coordinates)
            rotation_matrix_in_world = [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ]

            cluster_transform_in_world = get_transform_matrix(
                rotation_matrix_in_world, translation_in_world
            )

            world_to_cam_transform = get_world_to_cam_transform_matrix(self.get_cas())

            cluster_transform_in_cam = np.matmul(
                world_to_cam_transform, cluster_transform_in_world
            )

            # Annotate the pose information

            cluster_translation_in_cam = cluster_transform_in_cam[:3, 3]
            cluster_rotation_in_cam = cluster_transform_in_cam[:3, :3]

            # Generate a BB from the cluster info
            cluster_obb = o3d.geometry.OrientedBoundingBox(
                center=cluster_translation_in_cam,
                R=cluster_rotation_in_cam,
                extent=np.array([bb_size[0], bb_size[1], max_z - min_z]),
            )

            # Compute euclidean distance for x,y,z lengths
            min_bound = cluster_obb.get_min_bound().reshape(3, 1)
            max_bound = cluster_obb.get_max_bound().reshape(3, 1)
            x, y, z = np.linalg.norm(max_bound - min_bound, axis=1)
            size_annotation = BoundingBox3D()
            size_annotation.x_length = x
            size_annotation.y_length = y
            size_annotation.z_length = z

            # self.rk_logger.info(
            #     f"x length: {x}, y_length {y}, z_length {z}"
            #     f" of type object: {oh.classification.classname}")
            oh.annotations.append(size_annotation)
            vis_geometries.extend([cluster_obb, pcd])

            x1 = oh.roi.roi.pos.x
            y1 = oh.roi.roi.pos.y
            x2 = oh.roi.roi.pos.x + oh.roi.roi.width
            y2 = oh.roi.roi.pos.y + oh.roi.roi.height
            text = f"x: {x:0,.2f}, y: {y:0,.2f}, z: {z:0,.2f}"
            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(
                visualization_img,
                text,
                (int(x1), (int(y1) - 5)),
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )
            visualization_img = cv2.rectangle(
                visualization_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(vis_geometries)
        return Status.SUCCESS
