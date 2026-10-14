from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates DAiSy, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_config=WorldWithDaisyConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )


def main():
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
