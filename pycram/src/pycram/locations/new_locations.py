from copy import deepcopy

from typing_extensions import List

from krrood.adapters.json_serializer import list_like_classes
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
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def make_default_occupancy_costmap(context: Context, target: Pose) -> OccupancyCostmap:
    ground_pose = deepcopy(target)
    ground_pose.z = 0

    base_bb = context.robot.base.bounding_box

    return OccupancyCostmap(
        resolution=0.02,
        width=200,
        height=200,
        world=context.world,
        distance_to_obstacle=(base_bb.depth / 2 + base_bb.width / 2) / 2,
        robot_view=context.robot,
        origin=ground_pose,
    )


def get_object_in_hand(
    test_robot: AbstractRobot, test_world: World, arm: Arms = Arms.BOTH
) -> List[Body]:

    manipulator = ViewManager.get_end_effector_view(
        arm,
        test_robot,
    )
    manipulators = (
        [manipulator] if not isinstance(manipulator, list_like_classes) else manipulator
    )
    objs = set()
    for man in manipulators:
        objs.update(
            test_world.get_kinematic_structure_entities_of_branch(man.tool_frame)
        )
        objs.remove(man.tool_frame)
    return list(objs)


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    return Location(
        context,
        target_pose,
        make_default_occupancy_costmap(context, target_pose),
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
        man := ViewManager.get_end_effector_view(arm, context.robot),
    )
    costmap = make_default_occupancy_costmap(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=0.4,  # That needs to be replaced with an estimate of the reachability space of the robot arms
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            ReachabilitySequenceValidator(
                pose_sequence=grasp_description._pose_sequence(
                    target_pose,
                    get_object_in_hand(context.robot, context.world, arm)[0],
                ),
                tip_link=man.tool_frame,
                world=context.world,
                robot=context.robot,
            )
        ],
    )


def visibility_location(target_pose: Pose, context: Context) -> Location:
    camera = context.robot.get_default_camera()
    costmap = make_default_occupancy_costmap(context, target_pose) & VisibilityCostmap(
        min_height=camera.minimal_height,
        max_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            VisibilityValidator(
                world=context.world, robot=context.robot, target_pose=target_pose
            )
        ],
    )
