#!/usr/bin/env python
from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates Stretch, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_config=WorldWithStretchConfigDiffDrive(),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
            target_frequency=25,
            braking_time=timedelta(seconds=1.15),
        ),
    )


def main():
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
