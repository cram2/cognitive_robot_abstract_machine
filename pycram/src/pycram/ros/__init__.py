import os

if os.environ.get("ROS_VERSION") == "1":
    from pycram.ros.ros1 import *
elif os.environ.get("ROS_VERSION") == "2":
    from pycram.ros.ros2 import *
else:
    from pycram.ros.no_ros import *
