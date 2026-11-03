#!/usr/bin/env python
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world_description.connections import Connection6DoF


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates the HSR, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification(
                    semantic_annotation_type=HSRB,
                    description_parser=URDFParser.from_file(
                        HSRB.get_ros_file_path(), prefix=""
                    ),
                    localization_connection_type=Connection6DoF,
                )
            ]
        ),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
