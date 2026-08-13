"""
Location annotator for RoboKudo.

This module provides an annotator for determining object locations relative to world regions.
It supports:

* World descriptor integration
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

from py_trees.common import Status
from typing_extensions import List, TYPE_CHECKING

from robokudo.annotators.core import ThreadedAnnotator, BaseAnnotator
from robokudo.types.annotation import LocationAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import get_world_to_cam_transform_matrix
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
import robokudo.world as rk_world
from robokudo.utils.region import region_obb_in_cam_coordinates
from semantic_digital_twin.world_description.world_entity import Region

if TYPE_CHECKING:
    import numpy.typing as npt
    from collections.abc import Iterable


class LocationAnnotator(ThreadedAnnotator):
    """Determine object locations within semantic regions.

    The purpose of this location annotator is to receive a list of intriguing regions and incorporate the
    corresponding region names into the location annotations for objects that reside within those specific regions.

    The annotator:

    * Loads and manages world descriptor data
    * Transforms regions between world and camera frames
    * Checks object containment in regions
    * Creates location annotations for contained objects
    * Supports filtering by desired regions
    * Handles coordinate frame transformations
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for location annotator."""

        class Parameters:
            """Parameter container for location configuration."""

            def __init__(self) -> None:
                self.percentage: int = 50
                """Threshold percentage for an object to be considered in a region"""

                self.world_frame_name: str = "map"
                """Name of the world coordinate frame"""

                self.desired_regions: List[str] = ["kitchen_island"]
                """List of region names to consider. Leave empty to include all regions."""

        # overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

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

    def add_location_in_object_hypotheses(
        self,
        region: Region,
        world_to_cam_transform_matrix: npt.NDArray,
        object_hypotheses: Iterable[ObjectHypothesis],
    ) -> None:
        """Add location annotations to objects within a region.

        For each object hypothesis:

        * Transforms region to camera frame
        * Checks point containment in region
        * Calculates containment percentage
        * Creates location annotation if above threshold

        :param region: Semantic map region entry
        :param world_to_cam_transform_matrix: Transform from world to camera frame
        :param object_hypotheses: List of object hypotheses to check
        """
        runtime_world = rk_world.world_instance()
        obb = region_obb_in_cam_coordinates(
            runtime_world, region, world_to_cam_transform_matrix
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
                location_annotation = LocationAnnotation()
                location_annotation.region = region
                location_annotation.name = str(region.name)
                object_hypothesis.annotations.append(location_annotation)

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """Process object hypotheses to determine their locations.

        The method:

        * Loads and updates world descriptor
        * Gets camera to world transform
        * For each active region:
            * Checks if region is in desired list
            * Processes objects in region
            * Adds location annotations

        :return: SUCCESS after processing
        """
        start_timer = default_timer()
        runtime_world = rk_world.world_instance()
        regions = runtime_world.get_kinematic_structure_entity_by_type(Region)
        active_regions = {str(region.name): region for region in regions}
        # TODO Filter active regions by FRUSTUM CULLING

        world_to_cam_transform_matrix = get_world_to_cam_transform_matrix(
            self.get_cas()
        )
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        for region_name, region in active_regions.items():

            if len(self.descriptor.parameters.desired_regions) == 0:
                self.add_location_in_object_hypotheses(
                    region,
                    world_to_cam_transform_matrix,
                    object_hypotheses,
                )

            elif region_name in self.descriptor.parameters.desired_regions:
                self.add_location_in_object_hypotheses(
                    region,
                    world_to_cam_transform_matrix,
                    object_hypotheses,
                )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
