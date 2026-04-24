from dataclasses import dataclass

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class Turtlebot(AbstractRobot):
    """
    Class that describes the Turtlebot Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Turtlebot robot, if it exists.
        """
        ...

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Body:
        return world.get_body_by_name("base_footprint")

    def _setup_collision_rules(self):
        pass

    def _setup_other_hardware_interfaces(self):
        pass
