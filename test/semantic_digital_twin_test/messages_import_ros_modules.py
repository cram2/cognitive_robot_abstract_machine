"""
Imports the world synchronization messages in a fresh process and prints, one per line,
every ROS module the import loaded: ``rclpy`` and the ROS message packages.

Run as a subprocess by :mod:`test_messages`, whose process may have ROS imported.
"""

# %% importing the messages

import sys

import semantic_digital_twin.messages  # noqa: F401

# %% reporting


def main() -> None:
    """
    Prints the name of every loaded top-level ROS package.
    """
    top_level_packages = {name.partition(".")[0] for name in sys.modules}
    for package in sorted(top_level_packages):
        if package == "rclpy" or package.endswith("_msgs"):
            print(package)


if __name__ == "__main__":
    main()
