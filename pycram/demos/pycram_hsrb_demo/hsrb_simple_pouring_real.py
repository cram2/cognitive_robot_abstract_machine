import threading
import time
import os
import logging

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from suturo_resources.suturo_map import load_environment

from pycram.datastructures.enums import (
    TorsoState,
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot, real_robot
from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    fetch_world_from_service,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    Synchronizer,
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from pycram.robot_plans import (
    SimplePouringActionDescription,
    MoveTorsoActionDescription,
    PickUpAction,
    PickUpActionDescription,
)
from pycram.robot_plans import ParkArmsActionDescription
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.robots.abstract_robot import ParallelGripper
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
import pycram.alternative_motion_mappings.hsrb_motion_mapping


logger = logging.getLogger(__name__)
rclpy.init()


def _here(*parts: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


node = rclpy.create_node("viz_marker")
executor = SingleThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()
time.sleep(0.1)

# world from giskard
hsrb_world = fetch_world_from_service(node)
model_sync = ModelSynchronizer(world=hsrb_world, node=node)
state_sync = StateSynchronizer(world=hsrb_world, node=node)

try:
    hsrb_world.get_body_by_name("bowl.stl")
except Exception as e:
    env_world = load_environment()
    bowl_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
        )
    ).parse()
    with hsrb_world.modify_world():
        hsrb_world.merge_world(env_world)
        hsrb_world.merge_world_at_pose(
            milk_world,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.9, y=5.7, z=0.78, yaw=np.pi / 2
            ),
        )
        # hsrb_world.merge_world_at_pose(
        #     milk_world,
        #     pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.38, y=3.5, z=0.74),
        # )

VizMarkerPublisher(hsrb_world, node)

context = Context(
    hsrb_world, hsrb_world.get_semantic_annotations_by_type(HSRB)[0], ros_node=node
)
gripper = hsrb_world.get_semantic_annotations_by_type(ParallelGripper)[0]
# grasp=gripper.front_facing_orientation

# for arm_chain in self.robot_view.manipulator_chains:
grasp = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False)

#          .calculate_grasp_orientation(gripper.front_facing_orientation.to_np()))
#
# print(grasp)
plan = SequentialPlan(
    context,
    ParkArmsActionDescription(Arms.BOTH),
    # \MoveTorsoActionDescription(TorsoState.HIGH),
    # PouringActionDescription(world.get_body_by_name("milk.stl")),
    PickUpActionDescription(
        object_designator=hsrb_world.get_body_by_name("milk.stl"),
        arm=Arms.LEFT,
        grasp_description=grasp,
    ),
)
# )
# SimplePouringActionDescription(hsrb_world.get_body_by_name("bowl.stl"), Arms.LEFT),


with real_robot:
    plan.perform()
