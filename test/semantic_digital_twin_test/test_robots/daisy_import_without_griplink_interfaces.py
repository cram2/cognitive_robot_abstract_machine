"""
Imports the DAiSy robot description in a process where ``griplink_interfaces`` cannot be
imported, and prints the name of the robot class it found.

Run as a subprocess by :mod:`test_daisy`, whose process may have the package imported.
"""

import sys

# A None entry makes every import of the package raise ModuleNotFoundError.
sys.modules["griplink_interfaces"] = None
sys.modules["griplink_interfaces.action"] = None

from semantic_digital_twin.robots.daisy import DAiSy


def main() -> None:
    """
    Prints the name of the DAiSy robot class.
    """
    print(DAiSy.__name__)


if __name__ == "__main__":
    main()
