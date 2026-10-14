from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DAiSyVelocityInterface,
    WorldWithDaisyConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that commands DAiSy, described by what the robot publishes on
    ``/robot_description``.

    .. note:: The ros node has to be initialized first, because the robot description is
        read from its topic.
    """
    return Giskard(
        world_config=WorldWithDaisyConfig(urdf=get_robot_description()),
        robot_interface_config=DAiSyVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
            target_frequency=80,
            braking_time=timedelta(seconds=0.36),
        ),
    )


def main():
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
