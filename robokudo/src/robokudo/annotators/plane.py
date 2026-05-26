"""Plane detection and visualization in point clouds.

This module provides an annotator for:

* Detecting dominant planes using RANSAC
* Creating plane model annotations
* Visualizing detected planes and inliers
* Supporting threaded execution

The plane detection uses:

* RANSAC for robust model fitting
* Distance threshold for inlier selection
* Plane equation in ax + by + cz + d = 0 form

.. note::
   Plane visualization includes both inlier points and a mesh model.
"""

from timeit import default_timer

import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import ThreadedAnnotator, BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Plane
from robokudo.utils.transform import get_transform_from_plane_equation


class PlaneAnnotator(ThreadedAnnotator):
    """Plane detector and visualizer for point clouds.

    This annotator:

    * Detects largest plane using RANSAC
    * Creates plane model annotations
    * Visualizes plane inliers and model
    * Runs in a separate thread

    .. note::
       Uses Open3D's plane segmentation with configurable parameters.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for plane detection."""

        class Parameters:
            """Parameters for configuring plane detection."""

            def __init__(self) -> None:
                self.visualize_plane_model: bool = True
                """Show plane model mesh"""

                self.distance_threshold: float = 0.02
                """"""

                self.num_iterations: int = 50
                """"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "PlaneAnnotator",
        descriptor: "PlaneAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the plane detector.

        :param name: Name of this annotator instance, defaults to "PlaneAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

    def compute(self) -> Status:
        """Detect and annotate dominant plane in point cloud.

        The method:

        * Loads point cloud from CAS
        * Detects plane using RANSAC:
          * Distance threshold: 0.02
          * Sample size: 3 points
          * Iterations: 50
        * Creates plane annotation with:
          * Model parameters (a, b, c, d)
          * Inlier point indices
        * Visualizes results:
          * Inlier points in original cloud
          * Optional plane model mesh

        :return: SUCCESS after processing
        """
        self.rk_logger.info("Plane compute start")
        start_timer = default_timer()
        cloud = self.get_cas().get(CASViews.CLOUD)
        color_image = self.get_cas().get(
            CASViews.COLOR_IMAGE
        )  # shape [H, W, 3], usually RGB
        depth_image = self.get_cas().get(
            CASViews.DEPTH_IMAGE
        )  # shape [H, W], float or uint16
        cam_intrinsics = self.get_cas().get(CASViews.CAM_INTRINSIC)
        # print(f"Loaded cloud with {len(cloud.points)} points")

        plane_model, inliers = cloud.segment_plane(
            distance_threshold=self.descriptor.parameters.distance_threshold,
            ransac_n=3,
            num_iterations=self.descriptor.parameters.num_iterations,
        )

        [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        plane_annotation = Plane()
        plane_annotation.model = plane_model
        plane_annotation.inliers = inliers
        self.get_cas().annotations.append(plane_annotation)

        # Visualization

        # a) Show the points on the found plane
        plane_cloud = cloud.select_by_index(inliers)
        visualized_geometries = []
        visualized_geometries.append({"name": "plane_cloud", "geometry": plane_cloud})

        if self.descriptor.parameters.visualize_plane_model:
            # b) Show the plane model itself by a flat mesh
            plane_model_mesh = o3d.geometry.TriangleMesh.create_box(
                width=1.0, height=1.0, depth=0.01
            )
            plane_model_mesh.translate(
                np.array([-0.5, -0.5, 0])
            )  # Shift origin to center instead of corner
            plane_model_mesh.paint_uniform_color(np.array([1, 0, 0]))

            transform = get_transform_from_plane_equation(plane_model)
            plane_model_mesh.transform(transform)

            visualized_geometries.append(
                {"name": "plane_model", "geometry": plane_model_mesh}
            )
            self.get_annotator_output_struct().set_geometries(visualized_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.rk_logger.info("Plane compute end")
        return Status.SUCCESS
