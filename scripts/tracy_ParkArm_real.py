import sys
import threading

import rclpy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from semantic_digital_twin.robots.tracy import Tracy
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.motion_executor import real_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction



def main():
    rclpy.init()

    node = rclpy.create_node("coraplex_real_park_arms")

    # Giskard action/service clients need the node to spin.
    spinner = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spinner.start()

    print("Connecting to Giskard...")
    print("Expected action server: /giskard/command")

    try:
        giskard = GiskardWrapper(node_handle=node)
    except Exception as e:
        print(f"Could not create GiskardWrapper: {e}")
        print("Check:")
        print("ros2 node list | grep giskard")
        print("ros2 action info /giskard/command")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print("Getting live world from Giskard...")
    world = giskard.world
    if world is None:
        print("GiskardWrapper.world is None.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print(f"World received with {len(list(world.bodies))} bodies.")

    print("Building Tracy semantic robot from Giskard world...")
    try:
        tracy = giskard.robot
    except Exception as e:
        print(f"Could not build Tracy from Giskard world: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    context = Context(world=world, robot=tracy, ros_node=node)
    context.evaluate_conditions = False

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan

    print("Executing ParkArmsAction on REAL robot through Giskard...")
    print("Keep E-stop reachable.")

    with real_robot:
        plan.perform()

    node.destroy_node()
    rclpy.shutdown()
    print("ParkArmsAction completed.")


if __name__ == "__main__":
    main()
