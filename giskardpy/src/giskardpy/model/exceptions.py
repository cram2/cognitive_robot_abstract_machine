"""
Exceptions raised while building the world Giskard controls.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.data_types.exceptions import SetupException
from semantic_digital_twin.robots.robot_parts import AbstractRobot


# %% setup errors


@dataclass
class RobotDescriptionMissingError(SetupException):
    """
    Raised when a world is given no URDF and its robot class declares no description
    file either.
    """

    robot_class: type[AbstractRobot]
    """
    The robot class that declares no description file.
    """

    def error_message(self) -> str:
        return (
            f"No URDF was given and {self.robot_class.__name__} declares no "
            f"description file."
        )

    def suggest_correction(self) -> str:
        return (
            "Pass the URDF to the world config, or use a robot class that declares "
            "its description file."
        )
