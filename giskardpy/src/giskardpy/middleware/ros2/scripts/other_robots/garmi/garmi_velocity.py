#!/usr/bin/env python
import argparse
from dataclasses import dataclass, field

from semantic_digital_twin.adapters.ros import background_node
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GarmiInteractiveMarkerChain,
    GarmiVelocityInterface,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world_description.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.specifications.robots import RobotSpecification


@dataclass(kw_only=True)
class GarmiGiskard(Giskard):
    """
    Giskard for GARMI, with the velocity of its base weighted by
    :attr:`base_velocity_weight`.
    """

    base_velocity_weight: float = field(kw_only=True, default=0.02)
    """
    Objective weight of the velocity of the base's degrees of freedom, ten times the
    default every other degree of freedom gets, so Giskard prefers moving the arms over
    driving.
    """

    def setup(self) -> None:
        """
        Does what :meth:`Giskard.setup` does, then weights the base, whose degrees of
        freedom exist only once the world is built.
        """
        super().setup()
        drive = self.world.get_connections_by_type(OmniDrive)[0]
        for degree_of_freedom in drive.active_dofs:
            self.qp_controller_config.set_dof_weight(
                dof_name=degree_of_freedom.name,
                derivative=Derivatives.velocity,
                weight=self.base_velocity_weight,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="GARMI Giskard velocity controller.")
    parser.add_argument(
        "--interactive-marker",
        action="store_true",
        help="Also start the interactive marker server for Cartesian control via RViz.",
    )
    # parse_known_args ignores ROS 2 arguments (--ros-args ...) that argparse does not know about.
    args, _ = parser.parse_known_args()

    background_node.init_node("giskard")

    giskard = GarmiGiskard(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Garmi)]
        ),
        robot_interface_config=GarmiVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=40,
            prediction_horizon=20,
        ),
    )

    if args.interactive_marker:
        InteractiveMarkerNode.start_in_background_thread(
            chains=[chain.value for chain in GarmiInteractiveMarkerChain]
        )

    giskard.live()


if __name__ == "__main__":
    main()
