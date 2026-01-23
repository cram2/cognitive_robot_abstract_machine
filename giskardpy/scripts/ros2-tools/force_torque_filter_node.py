import rclpy

from giskardpy.ros2_tools.force_torque_filter_node import (
    ForceTorqueFilterNode,
    FilterConfig,
)


def main():
    rclpy.init()

    node = ForceTorqueFilterNode(config=FilterConfig())

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
