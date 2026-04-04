# Funky MonkeyPatching for ROS2 compatability
import builtin_interfaces.msg


def to_sec(self):
    """
    Returns the time in seconds from a builtin_interfaces.msg.Time message.

    :return: The time in seconds.
    """
    return self.sec


builtin_interfaces.msg.Time.to_sec = to_sec

import rclpy
import threading
from rclpy.node import Node

# rclpy.init()
# node = Node('pycram')
# threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


from pycram.ros.ros2.data_types import *
from pycram.ros.ros2.ros_tools import *
from pycram.ros.ros2.action_lib import *
from pycram.ros.ros2.service import *
from pycram.ros.ros2.publisher import *
from pycram.ros.ros2.subscriber import *
