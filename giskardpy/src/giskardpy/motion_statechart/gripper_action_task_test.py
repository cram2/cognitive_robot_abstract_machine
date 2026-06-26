import rclpy
from control_msgs.action import ParallelGripperCommand
from giskardpy.executor import Executor
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import RobotiqGripperActionServerTask

from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.world import World
from test.conftest import tracy_world


def robotiq_gripper_dataclass_test():
    rospy.init_node('asdf')

    giskard = GiskardWrapper(node_handle=rospy.node)
    #
    # task1 = RobotiqGripperActionServerTask(
    #     action_topic="/right_gripper/robotiq_gripper_controller/gripper_cmd",
    #     message_type=ParallelGripperCommand,
    #     target_position=0.0,
    # )

    task2 = RobotiqGripperActionServerTask(
        action_topic="/right_gripper/robotiq_gripper_controller/gripper_cmd",
        message_type=ParallelGripperCommand,
        target_position=1.4,
    )

    msc = MotionStatechart()
    msc.add_node(seq := Sequence([task2]))
    msc.add_node(EndMotion.when_true(seq))
    giskard.execute(msc)

    # msc = MotionStatechart()
    # msc.add_node(seq := Sequence([task2]))
    # msc.add_node(EndMotion.when_true(seq))
    # giskard.execute(msc)

    print("Done")
if __name__ == "__main__":
    robotiq_gripper_dataclass_test()
