from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates Tracy, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Tracy)]
        ),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
