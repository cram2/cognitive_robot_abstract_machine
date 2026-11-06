from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchVelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.specifications.robots import RobotSpecification


def create_giskard() -> Giskard:
    """
    Build a Giskard that commands Stretch, described by what the robot publishes on
    ``/robot_description``.

    .. note:: The ros node has to be initialized first, because the robot description is
        read from its topic.
    """
    return Giskard(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification.specify_real_robot(
                    Stretch, urdf=get_robot_description()
                )
            ]
        ),
        robot_interface_config=StretchVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
            target_frequency=25,
            braking_time=timedelta(seconds=1.15),
        ),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
