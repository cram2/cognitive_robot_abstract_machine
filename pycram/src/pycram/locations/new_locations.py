from dataclasses import dataclass
from typing import Iterator


from krrood.entity_query_language.factories import variable
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.base import Location
from pycram.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from pycram.locations.pose_validator import (
    ReachabilitySequenceValidator,
    VisibilityValidator,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.spatial_types.spatial_types import Pose


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    return Location(
        context,
        target_pose,
        OccupancyCostmap(
            resolution=0.02,
            width=200,
            height=200,
            orientation_generator=target_pose,
            world=context.world,
            distance_to_obstacle=0.2,
            robot_view=context.robot,
        ),
    )


def reachability_location(
    target_pose: Pose,
    context: Context,
    arm: Arms,
    grasp_description: GraspDescription = None,
) -> Location:
    grasp_description = grasp_description or GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(arm, context.robot),
    )
    costmap = OccupancyCostmap() & RingCostmap()
    return Location(context, target_pose, costmap, ReachabilitySequenceValidator())


def visibility_location(target_pose: Pose, context: Context) -> Location:
    costmap = OccupancyCostmap() & VisibilityCostmap()
    return Location(context, target_pose, costmap, VisibilityValidator())
