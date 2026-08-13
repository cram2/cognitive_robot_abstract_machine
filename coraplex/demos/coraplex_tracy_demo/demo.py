import os
import math
import logging
import threading
import time
from math import pi
from typing import Any, Tuple, List
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from coraplex.alternative_motion_mappings.tracy_motion_mapping import TracyGripperMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import real_robot, simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world

logger = logging.getLogger(__name__)

ExecutionType = ExecutionType.SIMULATED
#real = False  # Set to True for real robot hardware, False for simulation

print("Initializing ROS2...")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

# Keep the rospy middleware pointer synced with the active rclpy node
rospy.node = node

# Start the background thread for active ROS2 executor spinning
executor = MultiThreadedExecutor()
executor.add_node(node)
thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

if ExecutionType == ExecutionType.REAL:
    print("REAL MODE: Fetching active world from service...")
    # Fetch world from service (up to 300s timeout for slow service starts)
    world = fetch_world_from_service(node=node, timeout_seconds=300)

    # Start state synchronizer
    world_sync = WorldSynchronizer(_world=world, node=node)

elif ExecutionType == ExecutionType.SIMULATED:
    print("SIMULATED MODE: Parsing local URDF/Xacro description...")
    # Parse local simulated robot URDF/Xacro directly
    tracy_urdf_path = "package://iai_tracy_description/urdf/tracy.urdf.xacro"
    tracy_parser = URDFParser.from_file(file_path=tracy_urdf_path)
    world = tracy_parser.parse()
    Tracy.from_world(world)

    # Start visualization markers and TF publishing for Rviz
    viz = VizMarkerPublisher(_world=world, node=node)
    viz.with_tf_publisher()

# Extract the semantic robot view
robot_view = world.get_semantic_annotations_by_type(Tracy)[0]

# Assemble the planning context with the ROS node assigned
context = Context(world, robot_view, ros_node=node)

resources_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "resources", "objects"
)
cup_path = os.path.join(resources_dir, "jeroen_cup.stl")
cup = STLParser(cup_path).parse()

with world.modify_world():
    world.merge_world(
        cup,
        FixedConnection(
            world.root,
            cup.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                0.5, 0.15, 0.966, reference_frame=world.root
            ),
        ),
    )

# Console status logs
print(f"World Root Node: {world.root.name}")
print(f"Active World Bodies: {[body.name for body in world.bodies]}")

# Print Joint Position States
for dof in context.robot.degrees_of_freedom_with_hardware_interface:
    print(f"{dof.name}: {dof.variables.position.resolve()}")

pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.LEFT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=context.robot.get_left_arm_if_specified().end_effector,
    manipulation_offset=0.02,
)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        PickUpAction(
            world.get_body_by_name("jeroen_cup.stl"), Arms.LEFT, pick_up_grasp
        ),
        PlaceAction(
            world.get_body_by_name("jeroen_cup.stl"),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                0.6, -0.1, 0.966, reference_frame=world.root
            ).to_pose(),
            Arms.LEFT,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    ],
    context,
)

if ExecutionType == ExecutionType.REAL:
    with real_robot:
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished. Shutting down ROS Node...")
node.destroy_node()
rclpy.shutdown()
print("Done.")
