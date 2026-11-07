"""
Resolves one converter in each direction through the converter base classes, in a
process whose only reason to know the converters is its own import of their modules, and
prints the class names it found.

Run as a subprocess by :mod:`test_ros2_msg_converter`.
"""

from sensor_msgs.msg import LaserScan

from semantic_digital_twin.adapters.ros.msg_converter import (
    Ros2ToSemDTConverter,
    SemDTToRos2Converter,
)
from semantic_digital_twin.world_description.geometry import Box, Scale

import semantic_digital_twin.adapters.ros.ros2_to_semdt_converters  # noqa: F401
import semantic_digital_twin.adapters.ros.semdt_to_ros2_converters  # noqa: F401


def main() -> None:
    """
    Prints the name of the converter found for a ROS2 message and for a semDT object,
    one per line.
    """
    print(Ros2ToSemDTConverter.get_to_converter(LaserScan()).__name__)
    print(
        SemDTToRos2Converter.get_to_converter(Box(scale=Scale(1.0, 1.0, 1.0))).__name__
    )


if __name__ == "__main__":
    main()
