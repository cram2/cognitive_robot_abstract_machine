from pycram.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2
from pycram.datastructures.dataclasses import Context
from pycram.robot_plans import *
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.enums import Arms, TorsoState, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
import rclpy
import threading
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher

world = setup_world()
pr2_view = PR2.from_world(world)
context = Context(world, pr2_view)

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

with simulated_robot:
    print("=== Plan started ===")

    print("1. Parking arms")
    SequentialPlan(context, ParkArmsActionDescription(Arms.BOTH)).perform()
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
