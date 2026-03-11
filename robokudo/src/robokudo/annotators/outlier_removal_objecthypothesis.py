"""Statistical outlier removal and clustering for object hypotheses.

This module implements statistical outlier removal based on standard deviation and
number of neighbors in point clouds, followed by clustering to refine object hypotheses.
The implementation works in-place to minimize memory usage.

Key features:

* Statistical outlier removal using point neighborhoods
* DBSCAN clustering for object refinement
* In-place point cloud modification
* Configurable parameters per object class
* Visualization of processed clusters

Authors:
* Sorin Arion
* Naser Azizi
"""

import threading
from timeit import default_timer

import numpy as np
import open3d as o3d
import py_trees
import rclpy.parameter
import rcl_interfaces.msg
from typing_extensions import List, Dict, Type

import robokudo.annotators.core
import robokudo.io.ros
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.error_handling
import robokudo.cas

"""
This module implements a statistical outlierremoval based on the standard deviation and
number of neighbors in a point cloud. Afterwards a clustering is applied, to get for each 
object-hypothesis the actual object. This implementation works in place, such that there is no 
remaining copy of the old state for some point cloud. 

Authors: Sorin Arion, Naser Azizi 
"""

_PYTYPE_TO_ROS_FIELD: Dict[Type, str] = {
    int: "integer_value",
    float: "double_value",
    str: "string_value",
    bool: "bool_value",
    list: "string_array_value",  # rclpy supports string arrays only
}


class OutlierRemovalOnObjectHypothesisAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Annotator for statistical outlier removal and clustering refinement.

    This annotator processes object hypotheses by:

    * Removing statistical outliers from point clouds
    * Clustering remaining points using DBSCAN
    * Selecting largest cluster as refined object
    * Updating object hypothesis point clouds in-place
    * Providing visualization of processed clusters

    The processing can be selectively disabled for specific object classes.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for outlier removal and clustering."""

        class Parameters:
            """Parameter container for outlier removal configuration."""

            def __init__(self) -> None:
                # TODO Rename the dbscan_neighbors parameter. It is not about neighbors, but it is the MINIMUM
                self.dbscan_neighbors: int = 90
                """Minimum points to form a cluster (DBSCAN min_samples)"""

                self.dbscan_epsilon: float = 0.02
                """DBSCAN neighborhood size"""

                self.stat_neighbors: int = 200
                """Number of neighbors for statistical analysis"""

                self.stat_std: float = 0.5
                """Standard deviation threshold for outlier removal"""

                self.skip_removal_on_classes: List[str] = []
                """List of class names to skip processing
                
                If you want to skip the complete outlier removal process on certain classes, you can add a list of
                strings here. The values are case-sensitive!
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "OutlierRemovalOnObjectHypothesis",
        descriptor: "OutlierRemovalOnObjectHypothesisAnnotator.Descriptor" = Descriptor(),
    ):
        """Initialize the outlier removal annotator.

        :param name: Name of this annotator instance, defaults to "OutlierRemovalOnObjectHypothesis"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.node = robokudo.io.ros.get_node()

        for param_name, default_value in vars(self.descriptor.parameters).items():
            self.node.declare_parameter(f"{self.name}/{param_name}", default_value)

        self.node.add_on_set_parameters_callback(self.parameters_callback)

        self.rk_logger.info("OutlierRemovalNode initialized with current parameters")

    def parameters_callback(
        self, params: List[rclpy.parameter.Parameter]
    ) -> rcl_interfaces.msg.SetParametersResult:
        for param in params:
            if hasattr(self.descriptor.parameters, param.name):
                setattr(self.descriptor.parameters, param.name, param.value)

        self.rk_logger.info("Received reconf call: " + str(params))
        return rcl_interfaces.msg.SetParametersResult(successful=True)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """Process object hypotheses to remove outliers and refine clusters.

        :return: SUCCESS if processing completed, raises Exception if no clusters found
        :raises Exception: If no clusters are found after processing
        """

        start_timer = default_timer()
        pcd_cluster = self.cluster_statistical_outlierremoval_pcd()

        if not pcd_cluster:
            # TODO This should not be decided by this Annotator, but by the Result Generator - Refactor and test AEs
            self.rk_logger.warning(f"No Clusters have been found.")
            self.feedback_message = f"No clusters have been found"
            raise Exception("No Clusters have been found.")
        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def cluster_statistical_outlierremoval_pcd(self) -> bool:
        """Perform outlier removal and clustering on each object hypothesis.

        :return: True, if at least one of the object hypotheses could be optimized. False otherwise.
        """
        annotations = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )
        vis_geometries = []

        optimized_one_cluster = False

        for annotation in annotations:
            if annotation.points is None:
                continue

            if len(self.descriptor.parameters.skip_removal_on_classes) > 0:
                classes: List[robokudo.types.annotation.Classification] = (
                    robokudo.cas.CAS.filter_by_type(
                        robokudo.types.annotation.Classification, annotation.annotations
                    )
                )

                found_class = False
                for c in classes:
                    if (
                        c.classname
                        in self.descriptor.parameters.skip_removal_on_classes
                    ):
                        found_class = True
                        optimized_one_cluster = True  # this cluster could probably been optimized, but is ignored by config
                        continue
                if found_class:
                    continue

            pcd = annotation.points
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.descriptor.parameters.stat_neighbors,
                std_ratio=self.descriptor.parameters.stat_std,
            )

            pcd = pcd.select_by_index(ind)

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                labels = np.array(
                    pcd.cluster_dbscan(
                        eps=self.descriptor.parameters.dbscan_epsilon,
                        min_points=self.descriptor.parameters.dbscan_neighbors,
                        print_progress=True,
                    )
                )
            """
            We pick the biggest cluster, assuming that its point cloud represents
            an actual object and not noise
            """
            try:
                cluster_sizes = []
                max_label = labels.max()
                for val in range(0, max_label + 1):
                    cluster_sizes.append(np.where(labels == val)[0].shape[0])
                best_cluster = np.argmax(np.asarray(cluster_sizes))

            except ValueError as e:
                # We couldn't optimize THIS object hypothesis, but maybe there are other ones that already have been
                # optimized in previous iterations or we will find optimizables ones next
                continue
                # return False

            optimized_one_cluster = True

            cluster_indices = np.where(labels == best_cluster)[0]
            clustered_pcd = pcd.select_by_index(cluster_indices)
            # Replace old state with clustered point cloud
            annotation.points = clustered_pcd
            vis_geometries.append(clustered_pcd)

        if not optimized_one_cluster:
            return False

        visualization_img = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(vis_geometries)

        return True
