from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World


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
    def from_world(cls, world: World) -> Self:
        """
        Creates a Turtlebot robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Turtlebot robot view.
        """

        with world.modify_world():
            turtlebot = cls(
                name=PrefixedName("turtlebot", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            world.add_semantic_annotation(turtlebot)

        return turtlebot

