import os
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

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
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import GelatinBox
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import (
    PlanExecutor,
    RealRobotPlanExecutor,
    KinematicallySimulatedPlanExecutor,
)
from cramph.statechart import Statechart
from cramph.composites import Sequence

giskard_process = subprocess.Popen(
    ["ros2", "launch", "giskardpy_ros", "giskardpy_tracy_standalone.launch.py"],
    start_new_session=True,
)

time.sleep(8)  # Wait for the launch file to start

controls_real_robot = True

print("Init ROS")
rclpy.init()
node = rclpy.create_node("tracy_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

if controls_real_robot:
    # 300s matches giskardpy's own client (giskardpy/middleware/ros2/giskard_client_executor.py), which waits
    # this long for the same race: this demo's giskard/world-fetcher server is still parsing the URDF
    # and starting up when the client's default 10s budget would otherwise expire.
    world = fetch_world_from_service(node=node, timeout_seconds=300)

    WorldSynchronizer(_world=world, node=node)
else:
    world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    Tracy.from_world(world)
    VizMarkerPublisher(_world=world, node=node)

print("Setup World")
tracy = world.get_semantic_annotations_by_type(Tracy)[0]
# Narrower than the 85 mm Tracy's grippers open, so the fingers can close on a box.
box_edge = 0.06
box_scale = Scale(box_edge, box_edge, box_edge)
table_top_z = tracy.table.top_z
# Setup cubes
with world.modify_world():
    box1 = Body(
        name=PrefixedName("Box1"),
        collision=ShapeCollection(shapes=[Box(scale=box_scale)]),
        visual=ShapeCollection(
            shapes=[Box(scale=box_scale, color=Color.from_rgba(red=1, green=0, blue=0))]
        ),
    )

    box2 = Body(
        name=PrefixedName("Box2"),
        collision=ShapeCollection(shapes=[Box(scale=box_scale)]),
        visual=ShapeCollection(
            shapes=[Box(scale=box_scale, color=Color.from_rgba(red=0, green=1, blue=0))]
        ),
    )

    box3 = Body(
        name=PrefixedName("Box3"),
        collision=ShapeCollection(shapes=[Box(scale=box_scale)]),
        visual=ShapeCollection(
            shapes=[Box(scale=box_scale, color=Color.from_rgba(red=0, green=0, blue=1))]
        ),
    )

    world.add_kinematic_structure_entity(box1)
    world.add_kinematic_structure_entity(box2)
    world.add_kinematic_structure_entity(box3)

    # The boxes are annotated as gelatin boxes only so that the plan has an annotation
    # offering grasps to name them by; any graspable kind of object would do.
    box2_annotation = GelatinBox(root=box2)
    box3_annotation = GelatinBox(root=box3)
    world.add_semantic_annotations([box2_annotation, box3_annotation])

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box1,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.0, table_top_z + box_edge / 2
            ),
        )
    )

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box2,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.25, table_top_z + box_edge / 2
            ),
        )
    )

    world.add_connection(
        FixedConnection.create_with_dofs(
            parent=world.root,
            child=box3,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, -0.25, table_top_z + box_edge / 2
            ),
        )
    )

# The executor needs the ROS node to reach the real robot
executor_type: type[PlanExecutor] = (
    RealRobotPlanExecutor if controls_real_robot else KinematicallySimulatedPlanExecutor
)
executor = executor_type(
    StatechartContext.with_extensions(world, [RobotAccess(tracy)]), ros_node=node
)

plan = Sequence(
    nodes=[
        # Stack Box 2
        ParkArmsAction(arms=tracy.all_arms),
        PickUpAction(
            grasp=GraspCandidate(
                box2_annotation,
                Pose.from_xyz_rpy(
                    pitch=np.pi / 2, reference_frame=box2_annotation.root
                ),
            ),
            arm=tracy.left_arm,
        ),
        PlaceAction(
            object_designator=box2_annotation,
            target_location=Pose.from_xyz_rpy(
                0.8, 0.0, table_top_z + 1.5 * box_edge, reference_frame=world.root
            ),
        ),
        # Stack Box 3
        ParkArmsAction(arms=tracy.all_arms),
        PickUpAction(
            grasp=GraspCandidate(
                box3_annotation,
                Pose.from_xyz_rpy(
                    pitch=np.pi / 2, reference_frame=box3_annotation.root
                ),
            ),
            arm=tracy.right_arm,
        ),
        PlaceAction(
            object_designator=box3_annotation,
            target_location=Pose.from_xyz_rpy(
                0.8, 0.0, table_top_z + 2.5 * box_edge, reference_frame=world.root
            ),
        ),
    ]
)
try:
    print("Perform Plan")
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    executor.compile(statechart)
    executor.execute()
finally:
    executor.close()
    rclpy.shutdown()
    os.killpg(os.getpgid(giskard_process.pid), signal.SIGTERM)
    giskard_process.wait()
