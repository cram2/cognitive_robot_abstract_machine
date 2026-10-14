#!/usr/bin/env python
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates the HSR, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_config=WorldWithHSRConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )


def main():
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
