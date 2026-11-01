"""
A Unitree G1 moves a parcel between two pallet stacks of the AWS RoboMaker small
warehouse.

Needs the ``aws_robomaker_small_warehouse_world`` package built in the workspace, since
the world and its meshes are read from its share directory.
"""

from __future__ import annotations

import numpy as np
from rclpy.node import Node

from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.testing import selected_visualizations
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Parcel
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from cramph.executor import ExecutorExtension
from cramph.statechart import Statechart
from cramph.composites import Sequence

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
Where the parcel starts.

Rotated 90 degrees so a FRONT grasp approach can reach it.
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
    world = WorldSpecification.from_gazebo(
        WORLD_URI,
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.homogeneous_matrix,
            )
        ],
        objects=[
            BodySpecification.box(
                "parcel",
                PARCEL_SCALE,
                color=Color.from_rgba(red=0.85, green=0.45, blue=0.1),
                parent_T_self=PICK_POSE.homogeneous_matrix,
            )
        ],
    ).to_domain_object()
    # The parcel stands in for any graspable object; the plan only needs an annotation
    # to name it by, not a particular kind of object.
    with world.modify_world():
        world.add_semantic_annotation(Parcel(root=world.get_body_by_name("parcel")))
    return world


def straighten_torso(robot: AbstractRobot) -> JointPositionList:
    """
    :param robot: The robot whose torso is straightened.
    :return: The goal driving every torso joint back to zero, so the robot carries the
        parcel upright while it drives.
    """
    return JointPositionList(
        goal_state=JointState.from_mapping(
            {connection: 0.0 for connection in robot.torso.active_connections}
        )
    )


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


def build_statechart(
    context: StatechartContext, source: Pose, destination: Pose, turn: float
) -> Statechart:
    """
    :param context: The context the plan runs in, holding the robot carrying it out.
    :param source: Where the parcel stands when the plan starts.
    :param destination: Where the parcel is to be put down.
    :param turn: The yaw the robot turns by on the spot before it walks to the
        destination, so it does not walk through the pallet stack it took the parcel
        from.
    :return: The statechart carrying the parcel from one pallet stack to the other.
    """
    world = context.world
    robot = context.require_extension(RobotAccess).robot
    parcel = world.get_semantic_annotations_by_type(Parcel)[0]
    statechart = Statechart(context=context)
    statechart.add_node(
        Sequence(
            nodes=[
                ParkArmsAction(arms=robot.all_arms),
                NavigateAction(
                    target_location=standing_pose_in_front_of(source, world)
                ),
                PickUpAction(
                    grasp=GraspCandidate.from_body_origin(parcel),
                    arm=robot.torso.left_arm,
                ),
                ParkArmsAction(arms=robot.all_arms),
                straighten_torso(robot),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        yaw=turn, reference_frame=robot.root
                    )
                ),
                NavigateAction(
                    target_location=standing_pose_in_front_of(destination, world)
                ),
                PlaceAction(
                    object_designator=parcel,
                    target_location=Pose(
                        destination.position,
                        destination.quaternion,
                        reference_frame=world.root,
                    ),
                ),
                ParkArmsAction(arms=robot.all_arms),
                straighten_torso(robot),
            ]
        )
    )
    return statechart


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


def run(
    source: Pose,
    destination: Pose,
    turn: float,
    world: World,
    robot: UnitreeG1,
    ros_node: Node,
    visualizations: list[ExecutorExtension],
) -> None:
    """
    Carry the parcel from `source` to `destination` with `robot` in `world`, simulated,
    see :func:`build_statechart`, showing it with `visualizations`.
    """
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world, [RobotAccess(robot)]),
        ros_node=ros_node,
        extensions=visualizations,
    )
    executor.compile(build_statechart(executor.context, source, destination, turn))
    executor.execute()


world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

# Keeps PELVIS_HEIGHT_ABOVE_FLOOR honest: the robot has to stand on the floor rather than
# sink into it or hover above it.
assert abs(lowest_collision_point_of(robot, world)) < 1e-3

with RobotDemonstrationRosSession.start("warehouse_demo") as ros_session:
    visualizations = selected_visualizations(ros_session.node)
    for _ in range(10):
        run(
            PICK_POSE, PLACE_POSE, -1.57, world, robot, ros_session.node, visualizations
        )
        run(PLACE_POSE, PICK_POSE, 1.57, world, robot, ros_session.node, visualizations)
    run(PICK_POSE, PLACE_POSE, -1.57, world, robot, ros_session.node, visualizations)

parcel_position = world.get_body_by_name("parcel").global_pose
print(f"parcel delivered to {np.round(parcel_position.position, 3)}")
print(f"Expected parcel to be delivered to {np.round(PLACE_POSE.position, 3)}")
assert np.allclose(parcel_position, PLACE_POSE, atol=0.05)
