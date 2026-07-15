import os
import threading
import time
import math

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    Bowl,
    Spoon,
)

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.spatial_types import Pose
from spatial_types import Point3
from world import World

print("imports done")

SAMPLE_PLACE_POSES = False  # False → use hardcoded fallback poses instead

GARMI_ENV_XML = os.path.join(
    get_package_share_directory("iai_garmi_apartment"), "mjcf", "scene-bodies.xml"
)
GARMI_URDF = os.path.join("package://garmi_description/urdf/garmi.urdf")
GARMI_BASE_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0, 6, 0, yaw=math.pi / 2)

BOWL_STL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../",
    "resources",
    "objects",
    "bowl.stl",
)

BOWL_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 7.2, 1.0)
BOWL_TARGET_POINT_FALLBACK = Point3.from_iterable([1.6, 5.2, 0.8])
SPOON_TARGET_POINT_FALLBACK = Point3.from_iterable([1.6, 5.3, 0.8])

SPOON_STL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../",
    "resources",
    "objects",
    "spoon.stl",
)
SPOON_IN_DRAWER_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(-0.12, 0.0, 0.0)


def build_world() -> World:
    apartment_world = MJCFParser(GARMI_ENV_XML, use_visual_as_collision=True).parse()
    garmi_world = URDFParser.from_file(GARMI_URDF).parse()

    # Annotate the GARMI robot bodies before restructuring the kinematic tree.
    garmi_robot = Garmi.from_world(garmi_world)

    with garmi_world.modify_world():
        garmi_root = garmi_world.root
        map_body = Body(name=PrefixedName("map"))
        odom_combined = Body(name=PrefixedName("odom_combined"))

        map_C_odom = Connection6DoF.create_with_dofs(
            garmi_world, map_body, odom_combined
        )
        garmi_world.add_connection(map_C_odom)

        omni_drive = OmniDrive.create_with_dofs(
            parent=odom_combined,
            child=garmi_root,
            world=garmi_world,
            translation_velocity_limits=0.1,
            rotation_velocity_limits=0.1,
        )
        garmi_world.add_connection(omni_drive)
        omni_drive.has_hardware_interface = True

    omni_drive.origin = GARMI_BASE_POSE
    garmi_world.merge_world(apartment_world)

    # Add spoon inside drawer_1 (fixed to the drawer so it moves with it when opened).
    drawer_1 = garmi_world.get_body_by_name("drawer_1")
    spoon = STLParser(SPOON_STL).parse()
    spoon_in_drawer = FixedConnection(
        parent=drawer_1,
        child=spoon.root,
        parent_T_connection_expression=SPOON_IN_DRAWER_POSE,
    )
    garmi_world.merge_world(spoon, spoon_in_drawer)

    # Add bowl on the kitchen counter area.
    bowl = STLParser(BOWL_STL).parse()
    with garmi_world.modify_world():
        garmi_world.merge_world_at_pose(bowl, BOWL_POSE)

    return garmi_world, garmi_robot


def setup_semantics(world: World) -> None:
    world_reasoner = WorldReasoner(world)
    inferred = world_reasoner.infer_semantic_annotations()
    with world.modify_world():
        world.add_semantic_annotations(inferred)


world, garmi_robot = build_world()
setup_semantics(world)
assert isinstance(garmi_robot, Garmi)

if SAMPLE_PLACE_POSES:
    dining_table = Table(root=world.get_body_by_name("DiningTable"))
    with world.modify_world():
        world.add_semantic_annotation(dining_table)
        dining_table.calculate_supporting_surface()
    bowl = Bowl(root=world.get_body_by_name("bowl.stl"))
    spoon = Spoon(root=world.get_body_by_name("spoon.stl"))
    [bowl_place_point] = dining_table.sample_points_from_surface(
        body_to_sample_for=bowl,
        amount=1,
    )
    [spoon_place_point] = dining_table.sample_points_from_surface(
        body_to_sample_for=spoon,
        amount=1,
    )
    bowl_target_pose = Pose(
        position=bowl_place_point, reference_frame=bowl_place_point.reference_frame
    )
    spoon_target_pose = Pose(
        position=spoon_place_point, reference_frame=spoon_place_point.reference_frame
    )
else:
    bowl_target_pose = Pose(
        position=BOWL_TARGET_POINT_FALLBACK, reference_frame=world.root
    )
    spoon_target_pose = Pose(
        position=SPOON_TARGET_POINT_FALLBACK, reference_frame=world.root
    )

rclpy.init()
node = rclpy.create_node("garmi_apartment_viz")
publisher = VizMarkerPublisher(_world=world, node=node)
publisher.with_tf_publisher()

ros_executor = SingleThreadedExecutor()
ros_executor.add_node(node)
ros_thread = threading.Thread(
    target=ros_executor.spin, daemon=True, name="rclpy-executor"
)
ros_thread.start()

# Execute plan with the simulated robot.
context = Context.from_world(world)
with simulated_robot:
    sequential(
        [
            ParkArmsAction(arm=Arms.BOTH),
            # Note: always need TorsoState.HIGH or next(iter(self)) of CostmapLocation fails
            MoveTorsoAction(TorsoState.HIGH),
            TransportAction(
                object_designator=world.get_body_by_name("bowl.stl"),
                arm=Arms.RIGHT,
                grasp_description=GraspDescription(
                    ApproachDirection.RIGHT,
                    VerticalAlignment.TOP,
                    garmi_robot.get_right_arm_if_specified().end_effector,
                ),
                target_location=bowl_target_pose,
            ),
            MoveTorsoAction(TorsoState.HIGH),
            TransportAction(
                object_designator=world.get_body_by_name("spoon.stl"),
                arm=Arms.RIGHT,
                grasp_description=GraspDescription(
                    ApproachDirection.RIGHT,
                    VerticalAlignment.TOP,
                    garmi_robot.get_right_arm_if_specified().end_effector,
                ),
                target_location=spoon_target_pose,
            ),
        ],
        context,
    ).perform()

try:
    ros_thread.join()
except KeyboardInterrupt:
    ros_executor.shutdown()
