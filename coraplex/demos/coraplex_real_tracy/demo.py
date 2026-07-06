import os
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

giskard_process = subprocess.Popen(
    ["ros2", "launch", "giskardpy_ros", "giskardpy_tracy_standalone.launch.py"],
    start_new_session=True,
)

time.sleep(8)  # Wait for the launch file to start

execition_mode = ExecutionType.REAL

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

if execition_mode == ExecutionType.REAL:
    world = fetch_world_from_service(node=node)

    WorldSynchronizer(_world=world, node=node)
elif execition_mode == ExecutionType.SIMULATED:
    world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    Tracy.from_world(world)
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

print("Setup World")
# Setup cubes
with world.modify_world():
    box1 = Body(
        name=PrefixedName("Box1"),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection(
            shapes=[Box(scale=Scale(0.1, 0.1, 0.1), color=Color(1, 0, 0))]
        ),
    )

    box2 = Body(
        name=PrefixedName("Box2"),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection(
            shapes=[Box(scale=Scale(0.1, 0.1, 0.1), color=Color(0, 1, 0))]
        ),
    )

    box3 = Body(
        name=PrefixedName("Box3"),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection(
            shapes=[Box(scale=Scale(0.1, 0.1, 0.1), color=Color(0, 0, 1))]
        ),
    )

    world.add_kinematic_structure_entity(box1)
    world.add_kinematic_structure_entity(box2)
    world.add_kinematic_structure_entity(box3)

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box1,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.0, 0.92
            ),
        )
    )

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box2,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.25, 0.92
            ),
        )
    )

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box3,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, -0.25, 0.92
            ),
        )
    )

# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Tracy)[0],
    ros_node=node,
    evaluate_conditions=False,
)

plan = sequential(
    [
        # Stack Box 2
        ParkArmsAction(Arms.BOTH),
        PickUpAction(
            box2,
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                context.robot.left_arm.end_effector,
            ),
        ),
        PlaceAction(
            box2,
            Pose.from_xyz_rpy(0.8, 0.0, 1.02, yaw=0, reference_frame=world.root),
            Arms.LEFT,
        ),
        # Stack Box 3
        ParkArmsAction(Arms.BOTH),
        PickUpAction(
            box3,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                context.robot.right_arm.end_effector,
            ),
        ),
        PlaceAction(
            box3,
            Pose.from_xyz_rpy(0.8, 0.0, 1.12, yaw=0, reference_frame=world.root),
            Arms.RIGHT,
        ),
    ],
    context=context,
)
try:
    print("Perform Plan")
    with ExecutionEnvironment(execution_type=execition_mode, collision_avoidance=False):
        plan.perform()
finally:
    os.killpg(os.getpgid(giskard_process.pid), signal.SIGTERM)
    giskard_process.wait()
