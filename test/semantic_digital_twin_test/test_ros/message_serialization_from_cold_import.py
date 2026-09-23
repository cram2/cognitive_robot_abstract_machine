"""
Serializes a ROS2 message in a process whose only reason to know how is its own import
of the serializer module, and prints the type name the serialized message carries.

Run as a subprocess by :mod:`test_ros2_msg_converter`. Importing
:mod:`semantic_digital_twin.adapters.ros.ros_msg_serializer` is what registers the
serializer; the package this module lives in no longer does it on anyone's behalf.
"""

from geometry_msgs.msg import Point

from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import to_json

import semantic_digital_twin.adapters.ros.ros_msg_serializer  # noqa: F401


def main() -> None:
    """
    Prints the type name of a serialized ROS2 message.
    """
    print(to_json(Point(x=1.0))[JSONField.TYPE])


if __name__ == "__main__":
    main()
