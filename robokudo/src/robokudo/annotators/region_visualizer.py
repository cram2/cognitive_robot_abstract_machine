"""Region visualization for RoboKudo.

This module provides an annotator for visualizing semantic map regions in both
3D space and ROS visualization markers. It handles coordinate transformations
between different reference frames to properly display regions.
"""

import sys
from timeit import default_timer

import numpy
import numpy as np
import open3d as o3d
import py_trees
import rclpy
from scipy.spatial.transform import Rotation as R

import robokudo.annotators
import robokudo.annotators.core
import robokudo.io.tf_listener_proxy
import robokudo.semantic_map
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.transform
from robokudo.cas import CASViews
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.semantic_map import (
    get_obb_from_semantic_map_region_in_cam_coordinates,
    get_obb_from_semantic_map_region_with_transform_matrix,
)


class RegionVisualizer(robokudo.annotators.core.ThreadedAnnotator):
    """Annotator for visualizing semantic map regions.

    This annotator retrieves a semantic map and visualizes its regions in:

    1. The 3D Annotator output using Open3D
    2. ROS Visualization Markers

    The visualizer handles coordinate transformations between world frame,
    camera frame, and region-specific frames to properly display regions
    in the correct positions.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for region visualization."""

        class Parameters:
            """Parameters for configuring region visualization."""

            def __init__(self) -> None:
                self.world_frame_name: str = "map"
                """Name of the world coordinate frame"""

                self.semantic_map_ros_package: str = "robokudo"
                """ROS package containing semantic map."""

                self.semantic_map_name: str = "semantic_map_iai_kitchen"
                """Name of semantic map module. Should be in descriptors/semantic_maps/."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "RegionVisualizer",
        descriptor: "RegionVisualizer.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the region visualizer.

        :param name: Name of the annotator instance, defaults to "RegionVisualizer"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name=name, descriptor=descriptor)

        self.world_frame_name = self.descriptor.parameters.world_frame_name
        """Name of the world coordinate frame"""
        self.semantic_map = None
        """Loaded semantic map instance."""

        self.load_semantic_map()

    def load_semantic_map(self) -> None:
        """Load the semantic map from the configured ROS package.

        Uses ModuleLoader to dynamically load the semantic map module.
        """
        self.semantic_map = ModuleLoader.load_semantic_map(
            self.descriptor.parameters.semantic_map_ros_package,
            self.descriptor.parameters.semantic_map_name,
        )

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """Compute the visualization of semantic map regions.

        Creates visualizations containing:

        * 3D oriented bounding boxes for each region
        * Point cloud data
        * World coordinate frame
        * ROS visualization markers (via semantic map)

        Handles coordinate transformations between:

        * World frame to camera frame
        * Region-specific frames to camera frame

        :return: SUCCESS after creating visualizations
        :raises AssertionError: If region is not a valid SemanticMapEntry
        """
        start_timer = default_timer()
        cloud = self.get_cas().get(CASViews.CLOUD)

        self.load_semantic_map()
        self.semantic_map.publish_visualization_markers()

        active_regions = self.semantic_map.entries

        visualized_geometries = []

        try:
            world_to_cam_transform = (
                robokudo.utils.annotator_helper.get_world_to_cam_transform_matrix(
                    self.get_cas()
                )
            )
        except KeyError as err:
            self.rk_logger.warning(f"Couldn't find viewpoint in the CAS: {err}")
            return py_trees.common.Status.FAILURE

        for key, region in active_regions.items():
            assert isinstance(region, robokudo.semantic_map.SemanticMapEntry)
            # Will be used for saving the indices of the cloud for this specific region

            # if region is defined in the world frame, the region can be transformed with the transformation matrix from world to cam
            if region.frame_id == self.world_frame_name:
                transform_matrix = (
                    robokudo.utils.annotator_helper.get_world_to_cam_transform_matrix(
                        self.get_cas()
                    )
                )
                obb = get_obb_from_semantic_map_region_in_cam_coordinates(
                    region,
                    self.descriptor.parameters.world_frame_name,
                    transform_matrix,
                )

            # if the region is not defined in the world frame
            else:
                # raise "Not supported yet - Migrate to ROS2 first"
                # get translation and rotation of region
                transform_listener = robokudo.io.tf_listener_proxy.instance()
                newest = rospy.Time(0)
                try:
                    transform_listener.waitForTransform(
                        self.world_frame_name,
                        region.frame_id,
                        newest,
                        rospy.Duration(2.0),
                    )

                    translation_region_frame, rotation_region_frame = (
                        transform_listener.lookupTransform(
                            self.world_frame_name, region.frame_id, newest
                        )
                    )

                except Exception as err:
                    print(f"Camera Interface lookup_transform: Exception caught: {err}")
                    return False

                # calculate the transformation matrix from region frame to cam
                translation_region_frame = np.array(translation_region_frame)
                rotation_region_frame = np.array(rotation_region_frame)

                # calculate transformation matrix
                matrix_region_frame_to_world_frame = R.from_quat(
                    rotation_region_frame
                ).as_matrix()
                matrix_region_frame_to_world_frame = np.hstack(
                    (
                        matrix_region_frame_to_world_frame,
                        translation_region_frame.reshape(-1, 1),
                    )
                )
                matrix_region_frame_to_world_frame = np.vstack(
                    (matrix_region_frame_to_world_frame, np.array([0, 0, 0, 1]))
                )

                # calculate transformation from region to cam
                matrix_region_frame_to_camera = (
                    world_to_cam_transform @ matrix_region_frame_to_world_frame
                )

                transform_matrix = matrix_region_frame_to_camera
                obb = get_obb_from_semantic_map_region_with_transform_matrix(
                    region, transform_matrix
                )

            visualized_geometries.append({"name": region.name, "geometry": obb})

        # Place the filtered PointCloud into the CAS, overwriting the previous one

        visualized_geometries.append({"name": "cloud", "geometry": cloud})

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        visualized_geometries.append(
            {
                "name": "world_frame",
                "geometry": world_frame.transform(world_to_cam_transform),
            }
        )
        self.get_annotator_output_struct().set_geometries(visualized_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS
