"""Point cloud filtering based on semantic regions.

This module provides an annotator for:

* Filtering point clouds using semantic map regions
* Supporting regions in world and local frames
* Creating region hypotheses with poses
* Visualizing filtered regions and clouds

The module uses:

* Semantic maps to define regions of interest
* Frame transformations for region alignment
* Oriented bounding boxes for point filtering

.. note::
   Regions can be defined in either world frame or local frames.
"""

import sys
from timeit import default_timer

import numpy as np
import open3d as o3d
import py_trees
from rclpy.duration import Duration
from rclpy.time import Time
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


class RegionFilter(robokudo.annotators.core.ThreadedAnnotator):
    """Point cloud filtering using semantic map regions.

    The RegionFilter can be used to filter point clouds based on a environment model based on different
    regions. These regions are collected in a 'SemanticMap' which has one 'SemanticMapEntry' per region of interest.
    Semantics to these regions are linked by referencing well-known names from your URDF and/or knowledge base.

    .. note::
       Regions are defined in a SemanticMap with SemanticMapEntry objects.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for region filtering."""

        class Parameters:
            """Parameters for configuring region filtering.
            , defaults to "map"
            :ivar semantic_map_ros_package: , defaults to "robokudo"
            :ivar semantic_map_name: , defaults to "semantic_map_iai_kitchen"
            :ivar active_region: , empty for all regions
            """

            def __init__(self) -> None:
                self.world_frame_name: str = "map"
                """Name of the world coordinate frame"""

                self.semantic_map_ros_package: str = "robokudo"
                """ROS package containing semantic map"""

                self.semantic_map_name: str = "semantic_map_iai_kitchen"
                """Name of semantic map module. Should be in descriptors/semantic_maps/."""

                self.active_region: str = ""
                """Name of active region to filter. Does not define a specific region but can be used to check the active regions."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "RegionFilter",
        descriptor: "RegionFilter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the region filter.

        :param name: Name of this annotator instance, defaults to "RegionFilter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name=name, descriptor=descriptor)

        self.world_frame_name: str = self.descriptor.parameters.world_frame_name
        """Name of the world coordinate frame read from parameters"""

        self.semantic_map = None
        """The semantic map instance"""

        self.load_semantic_map()

        self.active_region = self.descriptor.parameters.active_region
        """Name of the active region to filter read from parameters"""

    def load_semantic_map(self) -> None:
        """Load semantic map from configured package and module.

        Uses ModuleLoader to dynamically load the semantic map module.
        """
        self.semantic_map = ModuleLoader.load_semantic_map(
            self.descriptor.parameters.semantic_map_ros_package,
            self.descriptor.parameters.semantic_map_name,
        )

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """Filter point cloud using semantic map regions.

        The method:

        * Loads point cloud and query from CAS
        * Updates semantic map and active regions
        * For each active region:
          * Transforms region to camera frame
          * Creates oriented bounding box
          * Filters points within box
          * Creates region hypothesis with pose
        * Updates CAS with filtered cloud
        * Creates visualization markers

        :return: SUCCESS after processing
        :raises Exception: If queried region not found in map
        """
        start_timer = default_timer()
        cloud = self.get_cas().get(CASViews.CLOUD)

        query = None
        if self.get_cas().contains(CASViews.QUERY):
            query = self.get_cas().get(CASViews.QUERY)

        self.load_semantic_map()
        self.semantic_map.publish_visualization_markers()

        active_regions = self.semantic_map.entries

        # Overwrite active_regions, if location is explicitly mentioned
        if query is not None:
            queried_location = query.obj.location
            if queried_location != "" and queried_location != self.active_region:
                self.rk_logger.info(
                    f"Setting region filter to check for location '{queried_location}'"
                )
                active_regions = dict()
                try:
                    active_regions[queried_location] = self.semantic_map.entries[
                        queried_location
                    ]
                except KeyError as ke:
                    raise Exception(
                        f"Couldn't find requested location {queried_location} in semantic map"
                    )

        visualized_geometries = []
        filtered_indices = set()

        self.rk_logger.debug(f"Analyzing {len(active_regions.keys())}")

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
            filtered_indices_for_this_region = set()
            # RegionHypothetis for this specific region
            region_hypothesis = robokudo.types.scene.RegionHypothesis()

            # if region is defined in the world frame, the region can be transformed with the transformation matrix from world to cam
            if region.frame_id == self.world_frame_name:
                obb = get_obb_from_semantic_map_region_in_cam_coordinates(
                    region,
                    self.descriptor.parameters.world_frame_name,
                    world_to_cam_transform,
                )
                # creates a PoseAnnotation
                pose = robokudo.types.annotation.PoseAnnotation()
                pose.translation = [
                    region.position_x,
                    region.position_y,
                    region.position_z,
                ]
                pose.translation = [
                    region.orientation_x,
                    region.orientation_y,
                    region.orientation_z,
                    region.orientation_w,
                ]

                region_hypothesis.annotations.append(pose)

            # if the region is not defined in the world frame
            else:
                # get translation and rotation of region
                transform_listener = robokudo.io.tf_listener_proxy.instance()
                newest = Time()
                try:
                    target_frame = self.world_frame_name  # avoid leading '/'
                    source_frame = region.frame_id

                    ok = transform_listener.can_transform(
                        target_frame,
                        source_frame,
                        newest,
                        timeout=Duration(seconds=2.0),
                    )
                    if not ok:
                        self.rk_logger.error(
                            f"lookup_transform: TF not available: {target_frame} <- {source_frame}"
                        )
                        return py_trees.common.Status.FAILURE

                    # ROS 2 equivalent of lookupTransform(...)
                    tf_stamped = transform_listener.lookup_transform(
                        target_frame,
                        source_frame,
                        newest,
                        timeout=Duration(seconds=2.0),
                    )

                    translation_region_frame = (
                        tf_stamped.transform.translation.x,
                        tf_stamped.transform.translation.y,
                        tf_stamped.transform.translation.z,
                    )
                    rotation_region_frame = (
                        tf_stamped.transform.rotation.x,
                        tf_stamped.transform.rotation.y,
                        tf_stamped.transform.rotation.z,
                        tf_stamped.transform.rotation.w,
                    )

                except Exception as err:
                    print(f"Camera Interface lookup_transform: Exception caught: {err}")
                    return py_trees.common.Status.FAILURE

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

                # Create a PoseAnnotation
                region_position_in_region_frame = np.array(
                    [region.position_x, region.position_y, region.position_z, 1]
                )
                # calculate region position in world frame
                region_position_in_world_frame = np.dot(
                    matrix_region_frame_to_world_frame, region_position_in_region_frame
                )
                x_transformed, y_transformed, z_transformed, _ = (
                    region_position_in_world_frame
                )

                rotation_region_frame_quat = R.from_quat(rotation_region_frame)
                rot_matrix_region_to_world = matrix_region_frame_to_world_frame[:3, :3]
                resulting_rot_matrix_world = np.dot(
                    rot_matrix_region_to_world, rotation_region_frame_quat.as_matrix()
                )
                rotation_world_frame_quat = R.from_matrix(
                    resulting_rot_matrix_world
                ).as_quat()
                rotation_world_frame_list = rotation_world_frame_quat.tolist()

                pose = robokudo.types.annotation.PoseAnnotation()
                pose.translation = [x_transformed, y_transformed, z_transformed]
                pose.rotation = rotation_world_frame_list

                region_hypothesis.annotations.append(pose)

            # To avoid copying the whole points per region into a new cloud, we'll only collect and save
            # the indices of matching points
            region_indices = obb.get_point_indices_within_bounding_box(cloud.points)
            filtered_indices.update(region_indices)

            # Update the indices which belong to the region and save the cloud and name of region in RegionHypothesis
            filtered_indices_for_this_region.update(region_indices)
            filtered_cloud_for_this_region = cloud.select_by_index(
                list(filtered_indices_for_this_region)
            )
            region_hypothesis.points = filtered_cloud_for_this_region
            region_hypothesis.name = region.name
            self.get_cas().annotations.append(region_hypothesis)

            visualized_geometries.append({"name": region.name, "geometry": obb})

        # Place the filtered PointCloud into the CAS, overwriting the previous one
        filtered_cloud = cloud.select_by_index(list(filtered_indices))
        self.get_cas().set(CASViews.CLOUD, filtered_cloud)

        visualized_geometries.append(
            {"name": "filtered cloud", "geometry": filtered_cloud}
        )

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
