import os

if os.environ.get("ROS_VERSION") == "1":
    from coraplex.ros.ros1 import *
elif os.environ.get("ROS_VERSION") == "2":
    from coraplex.ros.ros2 import *
else:
    from coraplex.ros.no_ros import *
