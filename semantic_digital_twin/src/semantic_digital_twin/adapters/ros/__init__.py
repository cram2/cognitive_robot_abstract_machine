# Load all ros2 specific json serializers when the ros module is used
from semantic_digital_twin.adapters.ros.ros_msg_serializer import *
import os

# uncomment this if you need to see why ros message parsing fails
# os.environ["ROS_PYTHON_CHECK_FIELDS"] = "1"
