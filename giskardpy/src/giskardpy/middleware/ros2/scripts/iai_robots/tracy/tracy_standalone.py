from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates Tracy, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_config=WorldWithTracyConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )


def main():
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
