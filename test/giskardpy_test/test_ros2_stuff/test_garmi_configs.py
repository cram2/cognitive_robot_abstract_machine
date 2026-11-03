from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GarmiVelocityInterface,
)
from giskardpy.middleware.ros2.scripts.other_robots.garmi.garmi_velocity import (
    GarmiGiskard,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.world_description.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world_description.connections import Connection6DoF

# %% base weighting


def test_garmi_giskard_weights_the_velocity_of_its_base(init_rospy):
    giskard = GarmiGiskard(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification(
                    semantic_annotation_type=Garmi,
                    description_parser=URDFParser.from_file(
                        Garmi.get_ros_file_path(), prefix=""
                    ),
                    localization_connection_type=Connection6DoF,
                )
            ]
        ),
        robot_interface_config=GarmiVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=40),
    )

    giskard.setup()

    [drive] = giskard.executor.context.world.get_connections_by_type(OmniDrive)
    for degree_of_freedom in drive.active_dofs:
        assert (
            giskard.qp_controller_config.get_degree_of_freedom_weight(
                degree_of_freedom.name, Derivatives.velocity
            )
            == giskard.base_velocity_weight
        )
