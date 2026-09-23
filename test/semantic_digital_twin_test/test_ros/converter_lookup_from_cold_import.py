"""
Resolves one converter in each direction in a process that has imported nothing but the
converter base classes, and prints the class names it found.

Run as a subprocess by :mod:`test_ros2_msg_converter`: in the process running the test
suite every converter module is already imported, so a lookup there proves nothing about
whether the registry finds them on its own.
"""

from sensor_msgs.msg import LaserScan

from semantic_digital_twin.adapters.ros.msg_converter import (
    Ros2ToSemDTConverter,
    SemDTToRos2Converter,
)
from semantic_digital_twin.world_description.geometry import Box, Scale


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
