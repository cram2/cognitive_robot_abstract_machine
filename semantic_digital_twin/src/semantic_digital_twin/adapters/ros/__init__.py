# Load all ros2 specific json serializers when the ros module is used
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import *
from semantic_digital_twin.adapters.ros.ros_msg_serializer import *
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import *

# uncomment this if you need to see why ros message parsing fails
# import os
# os.environ["ROS_PYTHON_CHECK_FIELDS"] = "1"
