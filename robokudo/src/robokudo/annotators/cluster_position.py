"""3D position estimation for object hypotheses.

This module provides an annotator for:

* Calculating 3D positions for object hypotheses
* Supporting different analysis scopes
* Computing centroids from point clouds
* Generating visualization markers

The module uses:

* Point cloud centroid computation
* Covariance analysis
* Open3D visualization tools
* Flexible annotation types

.. note::
   Can analyze either ObjectHypothesis or CloudAnnotation data.
"""

from __future__ import annotations

import copy
from timeit import default_timer
from typing_extensions import List, TYPE_CHECKING, Type

import open3d as o3d
import py_trees

from . import core, outputs
from ..types import annotation, scene
from ..cas import CAS, CASViews

if TYPE_CHECKING:
    import numpy.typing as npt


class ClusterPositionAnnotator(core.BaseAnnotator):
    """3D position estimation for object hypotheses.

    This annotator:

    * Calculates 3D positions from point clouds
    * Supports multiple analysis scopes
    * Computes centroids and covariance
    * Creates position annotations
    * Generates visualization markers

    .. note::
       Can process either ObjectHypothesis or CloudAnnotation data.
    """

    class Descriptor(core.BaseAnnotator.Descriptor):
        """Configuration descriptor for position estimation."""

        class Parameters:
            """Parameters for configuring position estimation.

            Analysis parameters:

            :ivar analysis_scope: Type of data to analyze (ObjectHypothesis or CloudAnnotation), defaults to ObjectHypothesis
            :type analysis_scope: type

            Visualization:

            :ivar visualizer_point_radius: Radius of centroid sphere markers in meters, defaults to 0.04
            :type visualizer_point_radius: float
            """

            def __init__(self):
                # On which data shall we perform the position estimation?
                # Object Hypothesis or specific annotations like CloudAnnotation?
                self.analysis_scope = scene.ObjectHypothesis
                self.visualizer_point_radius = 0.04  # in meters

    def __init__(self, name="ClusterPositionAnnotator", descriptor=Descriptor()):
        """Initialize the position estimator.

        :param name: Name of this annotator instance, defaults to "ClusterPositionAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: ClusterPositionAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        """Process object hypotheses and estimate positions.

        The method:

        * Loads point cloud from CAS
        * For each object hypothesis:
          * Gets appropriate point cloud data
          * Computes centroid and covariance
          * Creates position annotation
          * Creates visualization marker

        :return: SUCCESS after processing
        :rtype: py_trees.Status
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        centroids_to_visualize = []

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            scene.ObjectHypothesis
        )

        for object_hypothesis in object_hypotheses:
            if object_hypothesis.points is None:
                continue

            if (
                self.descriptor.parameters.analysis_scope
                == annotation.CloudAnnotation
            ):
                o_clouds: List[annotation.CloudAnnotation] = (
                    CAS.filter_by_type(
                        annotation.CloudAnnotation,
                        object_hypothesis.annotations,
                    )
                )
                if len(o_clouds) == 0:
                    self.rk_logger.warning(
                        "CloudAnnotation mode, but no CloudAnnotation found on object"
                    )
                    continue

                cluster_cloud = o_clouds[0].points
            else:
                cluster_cloud = object_hypothesis.points
            centroid, covariance = cluster_cloud.compute_mean_and_covariance()

            position = self.position_annotation_from_centroid(centroid)
            object_hypothesis.annotations.append(position)

            self.add_centroid_to_vis(centroid, centroids_to_visualize)

        # Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(centroids_to_visualize)

        self.get_annotator_output_struct().set_geometries(vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def position_annotation_from_centroid(self, centroid):
        """Create position annotation from centroid.

        :param centroid: 3D centroid coordinates
        :type centroid: numpy.ndarray
        :return: Position annotation with centroid as translation
        :rtype: robokudo.types.annotation.PositionAnnotation
        """
        position = annotation.PositionAnnotation()
        position.source = type(self).__name__
        position.translation = centroid
        return position

    def add_centroid_to_vis(self, centroid, centroids_to_visualize):
        """Add centroid visualization marker.

        Creates a colored sphere at the centroid position.

        :param centroid: 3D centroid coordinates
        :type centroid: numpy.ndarray
        :param centroids_to_visualize: List to append visualization marker to
        :type centroids_to_visualize: list
        """
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.descriptor.parameters.visualizer_point_radius
        )  # in meters
        centroid_sphere.paint_uniform_color([255, 0, 0])
        centroid_sphere.translate(centroid)
        centroids_to_visualize.append(centroid_sphere)
