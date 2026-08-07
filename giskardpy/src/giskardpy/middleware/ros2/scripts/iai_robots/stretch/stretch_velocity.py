from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
    StretchVelocityInterface,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch


def main():
    rospy.init_node("giskard")

    # the loaded urdf should be equivalent to the following xacro:
    robot_description = load_xacro(Stretch.get_ros_file_path())
    giskard = Giskard(
        world_config=WorldWithStretchConfigDiffDrive(urdf=robot_description),
        robot_interface_config=StretchVelocityInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
