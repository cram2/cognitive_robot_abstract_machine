"""
A Unitree G1 moves a parcel between two pallet stacks of the AWS RoboMaker small
warehouse.

Needs the ``aws_robomaker_small_warehouse_world`` package built in the workspace, since
the world and its meshes are read from its share directory.
"""

from __future__ import annotations

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveJointsMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import start_visualization
from coraplex.view_manager import ViewManager
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% where everything stands in the warehouse

WORLD_URI = (
    "package://aws_robomaker_small_warehouse_world/worlds/no_roof_small_warehouse/"
    "no_roof_small_warehouse.world"
)
"""
The roofless variant of the warehouse, which can be looked into from above in RViz.
"""

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all of its leg joints at zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much for the
robot's feet to rest on the floor rather than sink through it.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(1.9, 9.1, PELVIS_HEIGHT_ABOVE_FLOOR)
"""
Where the robot starts, in the aisle south of the two pallet stacks.
"""

PARCEL_SCALE = Scale(0.08, 0.08, 0.14)
"""
The extents of the transported parcel.
"""

PICK_POSE = Pose.from_xyz_rpy(2.75, 9.3, 0.793)
"""
Where the parcel starts. Rotated 90 degrees so a FRONT grasp approach can reach it.
"""

PLACE_POSE = Pose.from_xyz_rpy(2.6, 7.7, 0.8)
"""
Where the parcel ends up.
"""

STANDING_DISTANCE = 0.6
"""
How far the robot stands from a pose, in meters, opposite its FRONT-facing side.

Within the G1's reach, and far enough from a pallet stack to leave its footprint free.
"""

# %% building the world and the plan


def build_world() -> World:
    """
    :return: The warehouse with the G1 and the parcel in it.
    """
    return WorldSpecification.from_gazebo(
        WORLD_URI,
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.box(
                "parcel",
                PARCEL_SCALE,
                color=Color(0.85, 0.45, 0.1),
                parent_T_self=PICK_POSE.to_homogeneous_matrix(),
            )
        ],
    ).to_domain_object()


def standing_pose_in_front_of(pose: Pose, world: World) -> Pose:
    """
    :param pose: The pose the robot should approach from its FRONT-facing side.
    :param world: The world the pose is expressed in.
    :return: The pose the robot stands in to reach that pose with a FRONT grasp.
    """
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - STANDING_DISTANCE * np.cos(yaw),
        pose.y - STANDING_DISTANCE * np.sin(yaw),
        PELVIS_HEIGHT_ABOVE_FLOOR,
        yaw=yaw,
        reference_frame=world.root,
    )


def build_plan(world: World, robot: UnitreeG1) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :return: The plan transporting the parcel from one pallet stack to the other.
    """
    parcel = world.get_body_by_name("parcel")
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    place_pose = Pose(
        PLACE_POSE.to_position(), PLACE_POSE.to_quaternion(), reference_frame=world.root
    )
    pick_pose = Pose(
        PICK_POSE.to_position(), PICK_POSE.to_quaternion(), reference_frame=world.root
    )

    return sequential(
        [
            # %% bring to place pose
            ParkArmsAction(Arms.BOTH),
            NavigateAction(standing_pose_in_front_of(PICK_POSE, world)),
            PickUpAction(parcel, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=[
                    connection.name for connection in robot.torso.active_connections
                ],
                positions=[0.0] * len(robot.torso.active_connections),
            ),
            NavigateAction(Pose.from_xyz_rpy(yaw=-1.57, reference_frame=robot.root)),
            NavigateAction(standing_pose_in_front_of(PLACE_POSE, world)),
            PlaceAction(parcel, place_pose, Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=[
                    connection.name for connection in robot.torso.active_connections
                ],
                positions=[0.0] * len(robot.torso.active_connections),
            ),
        ],
        context=context,
    ).plan


def build_plan2(world: World, robot: UnitreeG1) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :return: The plan transporting the parcel from one pallet stack to the other.
    """
    parcel = world.get_body_by_name("parcel")
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    place_pose = Pose(
        PLACE_POSE.to_position(), PLACE_POSE.to_quaternion(), reference_frame=world.root
    )
    pick_pose = Pose(
        PICK_POSE.to_position(), PICK_POSE.to_quaternion(), reference_frame=world.root
    )

    return sequential(
        [
            # %% bring to place pose
            ParkArmsAction(Arms.BOTH),
            NavigateAction(standing_pose_in_front_of(PLACE_POSE, world)),
            PickUpAction(parcel, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=[
                    connection.name for connection in robot.torso.active_connections
                ],
                positions=[0.0] * len(robot.torso.active_connections),
            ),
            NavigateAction(Pose.from_xyz_rpy(yaw=1.57, reference_frame=robot.root)),
            NavigateAction(standing_pose_in_front_of(PICK_POSE, world)),
            PlaceAction(parcel, pick_pose, Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=[
                    connection.name for connection in robot.torso.active_connections
                ],
                positions=[0.0] * len(robot.torso.active_connections),
            ),
        ],
        context=context,
    ).plan


def lowest_collision_point_of(robot: UnitreeG1, world: World) -> float:
    """
    :param robot: The robot to measure.
    :param world: The world the height is expressed in.
    :return: The height of the robot's lowest collision geometry above the world's floor.
    """
    return min(
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
        for body in world.get_kinematic_structure_entities_of_branch(robot.root)
        if body.collision
    )


# %% running the demo

world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

# Keeps PELVIS_HEIGHT_ABOVE_FLOOR honest: the robot has to stand on the floor rather than
# sink into it or hover above it.
assert abs(lowest_collision_point_of(robot, world)) < 1e-3

start_visualization(world)

with simulated_robot:
    for _ in range(10):
        build_plan(world, robot).perform()
        build_plan2(world, robot).perform()
    build_plan(world, robot).perform()

parcel_position = world.get_body_by_name("parcel").global_pose
print(f"parcel delivered to {np.round(parcel_position.to_position(), 3)}")
print(f"Expected parcel to be delivered to {np.round(PLACE_POSE.to_position(), 3)}")
assert np.allclose(parcel_position, PLACE_POSE, atol=0.05)
