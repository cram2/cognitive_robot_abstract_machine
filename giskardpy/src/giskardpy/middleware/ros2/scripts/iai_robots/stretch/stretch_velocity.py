from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
    StretchVelocityInterface,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig


def main():
    rospy.init_node("giskard")
    # try:
    #     rospy.node.declare_parameters(
    #         namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    #     )
    #     robot_description = rospy.node.get_parameter_or("robot_description").value
    # except ParameterUninitializedException as e:
    robot_description = load_xacro(
        "package://stretch_description/urdf/stretch_description_RE2V0_tool_stretch_dex_wrist.xacro"
    )
    giskard = Giskard(
        world_config=WorldWithStretchConfigDiffDrive(urdf=robot_description),
        robot_interface_config=StretchVelocityInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=50
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
