"""
Location annotator for RoboKudo.

This module provides an annotator for determining object locations relative to semantic regions.
It supports:

* Semantic map integration
* Region-based location analysis
* Percentage-based containment checks
* World frame transformations
* Multi-region handling

The module is used for:

* Object localization
* Spatial reasoning
* Scene understanding
* Region-based filtering
"""

from __future__ import annotations
from timeit import default_timer

import numpy
import py_trees
from typing_extensions import List, TYPE_CHECKING

import robokudo.annotators.core
import robokudo.semantic_map
import robokudo.types
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.transform
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.semantic_map import (
    get_obb_from_semantic_map_region_in_cam_coordinates,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from collections.abc import Iterable


class LocationAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    """Determine object locations within semantic regions.

    The purpose of this location annotator is to receive a list of intriguing regions and incorporate the
    corresponding region names into the location annotations for objects that reside within those specific regions.

    The annotator:

    * Loads and manages semantic map data
    * Transforms regions between world and camera frames
    * Checks object containment in regions
    * Creates location annotations for contained objects
    * Supports filtering by desired regions
    * Handles coordinate frame transformations
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for location annotator."""

        class Parameters:
            """Parameter container for location configuration."""

            def __init__(self) -> None:
                self.percentage: int = 50
                """Threshold percentage for an object to be considered in a region"""

                self.world_frame_name: str = "map"
                """Name of the world coordinate frame"""

                self.semantic_map_ros_package: str = "robokudo"
                """ROS package containing semantic map"""

                self.semantic_map_name: str = "semantic_map_iai_kitchen"
                """Name of semantic map descriptor. Should be in descriptors/semantic_maps/"""

                self.desired_regions: List[str] = ["kitchen_island"]
                """List of regions from semantic_maps list to consider. Leave empty to include all regions."""

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion

    def __init__(
        self,
        name: str = "LocationAnnotator",
        descriptor: "LocationAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the location annotator.

        :param name: Annotator name, defaults to "LocationAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name=name, descriptor=descriptor)
        self.load_semantic_map()

        self.semantic_map = None
        """The currently loaded semantic map instance."""

    def load_semantic_map(self) -> None:
        """Load a semantic map from the configured package and name.

        Uses ModuleLoader to dynamically load the semantic map descriptor
        from the configured ROS package.

        :return: None
        """
        module_loader = ModuleLoader()
        self.semantic_map = module_loader.load_semantic_map(
            self.descriptor.parameters.semantic_map_ros_package,
            self.descriptor.parameters.semantic_map_name,
        )

    def add_location_in_object_hypotheses(
        self,
        region_name: str,
        region: robokudo.semantic_map.SemanticMapEntry,
        world_to_cam_transform_matrix: npt.NDArray,
        object_hypotheses: Iterable[robokudo.types.scene.ObjectHypothesis],
    ) -> None:
        """Add location annotations to objects within a region.

        For each object hypothesis:

        * Transforms region to camera frame
        * Checks point containment in region
        * Calculates containment percentage
        * Creates location annotation if above threshold

        :param region_name: Name of the region to check
        :param region: Semantic map region entry
        :param world_to_cam_transform_matrix: Transform from world to camera frame
        :param object_hypotheses: List of object hypotheses to check
        :return: None
        """
        obb = get_obb_from_semantic_map_region_in_cam_coordinates(
            region,
            self.descriptor.parameters.world_frame_name,
            world_to_cam_transform_matrix,
        )
        for object_hypothesis in object_hypotheses:
            # Extract the indices of an object that lies inside the region
            object_indices = obb.get_point_indices_within_bounding_box(
                object_hypothesis.points.points
            )
            # Total number of object indices
            total_indices = len(object_hypothesis.points.points)
            if total_indices == 0:
                continue
            # Calculate the percentage
            percentage_indices_inside = (len(object_indices) / total_indices) * 100
            if percentage_indices_inside >= self.descriptor.parameters.percentage:
                # Create a location annotation object
                location_annotation = robokudo.types.annotation.LocationAnnotation()
                # Insert the region name to the location annotation
                location_annotation.name = region_name
                object_hypothesis.annotations.append(location_annotation)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """Process object hypotheses to determine their locations.

        The method:

        * Loads and updates semantic map
        * Gets camera to world transform
        * For each active region:
            * Checks if region is in desired list
            * Processes objects in region
            * Adds location annotations

        :return: SUCCESS after processing
        """
        start_timer = default_timer()
        self.load_semantic_map()
        self.semantic_map.publish_visualization_markers()

        active_regions = self.semantic_map.entries
        # TODO Filter active regions by FRUSTUM CULLING

        world_to_cam_transform_matrix = (
            robokudo.utils.annotator_helper.get_world_to_cam_transform_matrix(
                self.get_cas()
            )
        )
        object_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )

        for region_name, region in active_regions.items():

            if len(self.descriptor.parameters.desired_regions) == 0:
                self.add_location_in_object_hypotheses(
                    region_name,
                    region,
                    world_to_cam_transform_matrix,
                    object_hypotheses,
                )

            elif region_name in self.descriptor.parameters.desired_regions:
                assert isinstance(region, robokudo.semantic_map.SemanticMapEntry)
                self.add_location_in_object_hypotheses(
                    region_name,
                    region,
                    world_to_cam_transform_matrix,
                    object_hypotheses,
                )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS
