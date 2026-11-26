"""
The launch files that bring up a giskard for Tracy.
"""

from __future__ import annotations

from enum import StrEnum


# %% launch files


class TiagoGiskardLaunchFile(StrEnum):
    """
    The launch files of ``giskardpy_ros`` that bring up a giskard for Tracy.
    """

    STANDALONE = "giskardpy_tiago_standalone.launch.py"
    """
    A giskard that simulates Tracy instead of commanding it, with nothing serving the
    gripper actions.
    """

    VELOCITY = "giskardpy_tiago_velocity.launch.py"
    """
    A giskard that commands the velocity controllers of the real Tracy.
    """
