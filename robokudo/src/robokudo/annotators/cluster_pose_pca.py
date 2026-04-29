"""3D pose estimation using principal component analysis.

This module provides an annotator for:

* Calculating 3D poses for object hypotheses
* Using PCA for orientation estimation
* Computing oriented bounding boxes
* Generating visualization markers

The module uses:

* Principal component analysis for axes
* Eigenvalue decomposition
* Coordinate frame alignment
* Open3D visualization tools

.. note::
   Requires at least 10 points per object hypothesis.
"""

import copy
from timeit import default_timer

import numpy as np
import numpy.linalg
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import (
    get_transform_matrix,
    get_quaternion_from_rotation_matrix,
)


class ClusterPosePCAAnnotator(BaseAnnotator):
    """3D pose estimation using principal component analysis.

    This annotator:

    * Calculates 3D poses for object point clusters
    * Uses PCA for orientation estimation
    * Aligns coordinate frames with principal axes
    * Creates pose annotations
    * Generates visualization markers

    .. note::
       Requires minimum 10 points per object hypothesis.
    """

    def __init__(self, name: str = "ClusterPosePCAAnnotator") -> None:
        """Initialize the PCA pose estimator.

        :param name: Name of this annotator instance, defaults to "ClusterPosePCAAnnotator"
        """
        super().__init__(name)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """Process object hypotheses and estimate poses.

        The method:

        * Loads point cloud from CAS
        * For each object hypothesis:
          * Computes centroid and covariance
          * Performs eigenvalue decomposition
          * Aligns coordinate frame with principal axes
          * Creates pose annotations
          * Creates visualization markers

        :return: SUCCESS after processing
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        annotations = self.get_cas().annotations
        geometries_to_visualize = []

        # Iterate over everything that is a Object hypothesis and calculate the centroid
        for annotation in annotations:
            if not isinstance(annotation, ObjectHypothesis):
                continue
            # Each hypothesis must have valid points before we can proceed
            if annotation.points is None:
                continue
            if not isinstance(annotation.points, o3d.geometry.PointCloud):
                continue
            if len(annotation.points.points) <= 10:
                continue

            object_hypothesis = annotation

            cluster_cloud = object_hypothesis.points
            centroid, covariance = cluster_cloud.compute_mean_and_covariance()
            eigenvalues, eigenvectors = numpy.linalg.eig(covariance)

            # from scipy.linalg import eig
            # w,vl,vr = scipy.linalg.eig(covariance)

            eigenvalues_copy = copy.deepcopy(eigenvalues)
            first_eigen_vec_idx = eigenvalues_copy.argmax()  # z
            z = eigenvectors[:, first_eigen_vec_idx]
            eigenvalues_copy[first_eigen_vec_idx] = -1

            second_eigen_vec_idx = eigenvalues_copy.argmax()  # x
            x = eigenvectors[:, second_eigen_vec_idx]
            eigenvalues_copy[second_eigen_vec_idx] = -1

            # third_eigen_vec_idx = eigenvalues_copy.argmax()  # y
            y = -np.cross(x, z)
            # y = eigenvectors[:,third_eigen_vec_idx]
            # eigenvalues_copy[third_eigen_vec_idx] = -1

            new_rotation = numpy.zeros_like(eigenvectors)
            new_rotation[:, 0] = x
            new_rotation[:, 1] = y
            new_rotation[:, 2] = z

            cluster_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

            t = get_transform_matrix(new_rotation, centroid)
            cluster_frame.transform(t)

            pose_annotation = PoseAnnotation()
            pose_annotation.translation = list(centroid)
            pose_annotation.rotation = list(
                get_quaternion_from_rotation_matrix(new_rotation)
            )
            pose_annotation.source = type(self).__name__
            object_hypothesis.annotations.append(pose_annotation)

            cluster_obb = cluster_cloud.get_oriented_bounding_box()
            geometries_to_visualize.append(cluster_obb)

            geometries_to_visualize.append(cluster_frame)

        # Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(geometries_to_visualize)

        self.get_annotator_output_struct().set_geometries(vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
