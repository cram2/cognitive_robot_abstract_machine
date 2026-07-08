import logging
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.datastructures.definitions import GripperState
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoGripMotion
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareTest")


def main():
    # 1. Initialize the Giskard ROS 2 node wrapper
    rospy.init_node("tracy_gripper_hardware_test")
    giskard = GiskardWrapper(node_handle=rospy.node)
    # 2. Instantiate your high-level alternative motion class
    motion_left = TiagoGripMotion(gripper=Arms.LEFT, position=0.19)
    # 3. Extract the underlying low-level RobotiqGripperActionServerTask
    hardware_task = motion_left._motion_chart
    logger.info(f"Extracted hardware task for topic: {hardware_task.action_topic}")

    # 4. Construct the Giskard MotionStatechart
    msc = MotionStatechart()
    msc.add_node(seq := Sequence([hardware_task]))
    msc.add_node(EndMotion.when_true(seq))

    # 5. Execute on the physical robot
    logger.info("Executing MotionStatechart via Giskard...")
    giskard.execute(msc)
    logger.info("Execution finished successfully!")

    time.sleep(2)
if __name__ == "__main__":
    main()