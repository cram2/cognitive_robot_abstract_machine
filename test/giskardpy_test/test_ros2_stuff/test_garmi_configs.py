from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GarmiVelocityInterface,
    WorldWithGarmiConfig,
)
from giskardpy.middleware.ros2.scripts.other_robots.garmi.garmi_velocity import (
    GarmiGiskard,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import OmniDrive

# %% base weighting


def test_garmi_giskard_weights_the_velocity_of_its_base(init_rospy):
    giskard = GarmiGiskard(
        world_config=WorldWithGarmiConfig(urdf=load_xacro(Garmi.get_ros_file_path())),
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
