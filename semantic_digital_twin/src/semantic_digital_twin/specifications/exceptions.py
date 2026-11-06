"""
Exceptions raised by specifications of things placed into a world.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% robot specifications


@dataclass
class VisualCollisionBackupDisabledError(UsageError):
    """
    Raised when a robot whose description leaves parts without collision geometry is
    given a parser that does not let those parts collide with their visual geometry.
    """

    robot_class: type[AbstractRobot]
    """
    The robot class that needs its visual geometry as collision backup.
    """

    def error_message(self) -> str:
        return (
            f"{self.robot_class.__name__} needs its visual geometry as collision "
            f"backup, but its description parser does not use it."
        )

    def suggest_correction(self) -> str:
        return (
            "Create the description parser with use_visual_as_collision_backup=True, "
            "or use RobotSpecification.specify_real_robot."
        )


@dataclass
class RobotNamesTakenError(UsageError):
    """
    Raised when a robot is spawned into a world that already holds entities with some of
    the robot's names, which would leave name lookups unable to tell them apart.
    """

    robot_class: type[AbstractRobot]
    """
    The robot class whose spawn was rejected.
    """

    taken_names: list[PrefixedName]
    """
    The names of the robot that the world already holds.
    """

    def error_message(self) -> str:
        names = ", ".join(str(name) for name in self.taken_names)
        return (
            f"{self.robot_class.__name__} cannot be spawned, because the world already "
            f"holds entities named {names}."
        )

    def suggest_correction(self) -> str:
        return (
            "Read the robot with a description_parser whose prefix no other robot in "
            "the world uses."
        )
