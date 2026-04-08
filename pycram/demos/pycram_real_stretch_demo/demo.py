import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, MovementType
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import real_robot
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans import MoveToolCenterPointMotion
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import ReachAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, SetGripperAction, ParkArmsAction
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import ModelSynchronizer, StateSynchronizer
from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types.spatial_types import Pose
import pycram.alternative_motion_mappings.stretch_motion_mapping # type: ignore

rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = SingleThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = fetch_world_from_service(node)

print(len(world.bodies))

ModelSynchronizer(_world=world, node=node)
StateSynchronizer(_world=world, node=node)

robot_annotation = world.get_semantic_annotations_by_type(Stretch)[0]
context = Context(world, robot_annotation, node)

plan = sequential([
    ParkArmsAction(Arms.BOTH),
# MoveTorsoAction(TorsoState.MID),
#                    ReachAction(Pose.from_xyz_rpy(0.6, -1, 0.7, reference_frame=world.root), arm=Arms.LEFT,
#                                grasp_description=GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, robot_annotation.arm.manipulator))
NavigateAction(Pose.from_xyz_rpy(2.35, 0.79, 0, reference_frame=world.root)),
MoveToolCenterPointMotion(Pose.from_xyz_rpy(2.3, 0.38, 0.5, reference_frame=world.root), arm=Arms.LEFT, movement_type=MovementType.TRANSLATION)
], context).plan

with real_robot:
    plan.perform()


rclpy.shutdown()
