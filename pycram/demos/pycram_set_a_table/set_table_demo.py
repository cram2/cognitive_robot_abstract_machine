from pycram.testing import setup_world
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from pycram.robot_plans import *
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.enums import Arms, TorsoState, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan, ParallelPlan
from pycram.process_module import simulated_robot
import rclpy
import threading
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.world import World
from semantic_digital_twin.adapters.urdf import URDFParser
import os
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.adapters.procthor.procthor_semantic_annotations import Milk


def setup_world() -> World:
    logger.setLevel(logging.DEBUG)

    pr2_sem_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "robots",
            "hsrb.urdf",
        )
    ).parse()
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "worlds",
            "kitchen.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    # apartment_world.merge_world(pr2_sem_world)
    apartment_world.merge_world(milk_world)
    apartment_world.merge_world(cereal_world)

    with apartment_world.modify_world():
        pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
        apartment_root = apartment_world.root
        c_root_bf = OmniDrive.create_with_dofs(
            parent=apartment_root, child=pr2_root, world=apartment_world
        )
        apartment_world.merge_world(pr2_sem_world, c_root_bf)
        c_root_bf.origin = TransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

    apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
        TransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        )
    )
    apartment_world.get_body_by_name(
        "breakfast_cereal.stl"
    ).parent_connection.origin = TransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.05, reference_frame=apartment_world.root
    )
    milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world

world = setup_world()
robot = HSRB.from_world(world)
context = Context(world, robot)

# Define arms and torso
arm = Arms.RIGHT
torso_pose = TorsoState.HIGH

# Define object to pick up
milk_object = world.get_body_by_name("milk.stl")

# Define navigation poses
pickup_pose = PoseStamped.from_list([1.8, 2.2, 0.0], [0.0, 0.0, 0.0, 1], frame=world.root)
place_pose = PoseStamped.from_list([2.4, 1.8, 0.0], [0.0, 0.0, 0.0, 1], frame=world.root)
place_pose_high = PoseStamped.from_list([2.4, 1.8, 0.95], [0.0, 0.0, 0.0, 1], frame=world.root)

rclpy.init()
rclpy.create_node('demo_node')
threading.Thread(target=rclpy.spin, args=(rclpy.create_node('demo_node'),), daemon=True).start()

VizMarkerPublisher(world=world, node=rclpy.create_node('demo_node'))
simulated_robot.speedup = True  # Speeds up actions in simulation

with simulated_robot:
    print("=== Plan started ===")

    print("1. Parking arms")
    SequentialPlan(context, ParkArmsActionDescription(Arms)).perform()
    print("   Arms parked ✅")

    print("2. Moving torso up")
    SequentialPlan(context, MoveTorsoActionDescription([torso_pose])).perform()
    print("   Torso moved ✅")

    print("3. Navigating to pickup location")
    SequentialPlan(context, NavigateActionDescription([pickup_pose])).perform()
    print("   Reached pickup location ✅")

    print("4. Picking up milk")
    SequentialPlan(context, PickUpActionDescription(
        object_designator=milk_object,
        arm=[arm],
        grasp_description=GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
        )
    )).perform()
    print("   Milk picked up ✅")

    print("5. Navigating to place location")
    SequentialPlan(context, NavigateActionDescription([place_pose])).perform()
    print("   Reached place location ✅")

    print("6. Placing milk")
    SequentialPlan(context, PlaceActionDescription(
        object_designator=milk_object,
        target_location=[place_pose_high],
        arm=arm
    )).perform()
    print("   Milk placed ✅")

    print("=== Plan finished ===")
