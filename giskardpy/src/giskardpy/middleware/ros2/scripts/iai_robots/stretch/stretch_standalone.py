#!/usr/bin/env python
from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world_description.connections import Connection6DoF


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates Stretch, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification(
                    semantic_annotation_type=Stretch,
                    description_parser=URDFParser.from_file(
                        Stretch.get_ros_file_path(), prefix=""
                    ),
                    localization_connection_type=Connection6DoF,
                )
            ]
        ),
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
