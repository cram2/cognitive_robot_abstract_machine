from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.robots.robot_parts import AbstractRobot


@dataclass(eq=False)
class Turtlebot(AbstractRobot):
    """
    Class that describes the Turtlebot Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_robot_part_semantic_annotations(self):
        pass
