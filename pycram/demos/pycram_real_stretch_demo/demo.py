import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, MovementType
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import real_robot
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans import MoveToolCenterPointMotion
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, SetGripperAction, ParkArmsAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import ModelSynchronizer, StateSynchronizer
from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
import pycram.alternative_motion_mappings.stretch_motion_mapping # type: ignore
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = SingleThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = fetch_world_from_service(node)

with world.modify_world():
    box = Body(name=PrefixedName("box"), collision=ShapeCollection([Box(scale=Scale(0.315,0.315,0.46))]))
    world.add_connection(FixedConnection(world.root, box, parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(2.38, 0.34, 0.23)))

    cereal = STLParser("/home/stretch/cognitive_robot_abstract_machine/pycram/resources/objects/breakfast_cereal.stl").parse()
    world.merge_world_at_pose(cereal, HomogeneousTransformationMatrix.from_xyz_rpy(2.35, 0.35, 0.53, yaw=-1.57, reference_frame=world.root))



print(len(world.bodies))

ModelSynchronizer(_world=world, node=node)
StateSynchronizer(_world=world, node=node)
VizMarkerPublisher(_world=world, node=node)

robot_annotation = world.get_semantic_annotations_by_type(Stretch)[0]
context = Context(world, robot_annotation, node, _debug=False, evaluate_conditions=False)

plan = sequential([
    # ParkArmsAction(Arms.BOTH),
    # MoveTorsoAction(TorsoState.HIGH),
    #                    ReachAction(Pose.from_xyz_rpy(0.6, -1, 0.7, reference_frame=world.root), arm=Arms.LEFT,
    #                                grasp_description=GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, robot_annotation.arm.manipulator))
    NavigateAction(Pose.from_xyz_rpy(2.35, 1.3, 0, yaw=-1.57, reference_frame=world.root)),
    # underspecified(NavigateAction)(
    #     target_location=variable(
    #         Pose,
    #         domain=CostmapLocation(
    #             target=world.get_body_by_name("breakfast_cereal.stl").global_pose,
    #             reachable_arm=Arms.LEFT,
    #             reachable=True,
    #             context=context,
    #             grasp_description=(grasp_desc := GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, robot_annotation.arm.manipulator)),
    #         ),
    #     ),
    #     keep_joint_states=True,
    # ),
    # MoveToolCenterPointMotion(Pose.from_xyz_rpy(2.3, 0.38, 0.5, reference_frame=world.root), arm=Arms.LEFT, movement_type=MovementType.TRANSLATION)
    # PickUpAction(world.get_body_by_name("breakfast_cereal.stl"), Arms.LEFT, grasp_desc)
    ],
    context).plan

with real_robot:
    plan.perform()


rclpy.shutdown()
