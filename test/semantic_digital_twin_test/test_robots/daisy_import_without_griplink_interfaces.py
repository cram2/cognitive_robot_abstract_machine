"""
Imports the DAiSy robot description in a process where ``griplink_interfaces`` cannot be
imported. Prints every record logged meanwhile as its logger's name and level, one per
line, and then the name of the robot class it found.

Run as a subprocess by :mod:`test_daisy`, whose process may have the package imported.
"""

# %% hiding griplink_interfaces

import logging
import sys

# A None entry makes every import of the package raise ModuleNotFoundError.
sys.modules["griplink_interfaces"] = None
sys.modules["griplink_interfaces.action"] = None

logging.basicConfig(stream=sys.stdout, format="%(name)s %(levelname)s")

from semantic_digital_twin.robots.daisy import DAiSy

# %% reporting


def main() -> None:
    """
    Prints the name of the DAiSy robot class.
    """
    print(DAiSy.__name__)


if __name__ == "__main__":
    main()
