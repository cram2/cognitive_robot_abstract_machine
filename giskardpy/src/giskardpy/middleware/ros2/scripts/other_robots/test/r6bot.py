#!/usr/bin/env python
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import (
    ClosedLoopBTConfig,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.configs import GenericWorldConfig
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class R6BotInterface(RobotInterfaceConfig):
    def setup(self):
        GiskardBlackboard()
        self.sync_joint_state_topic("/joint_states")
        self.add_joint_velocity_group_controller(
            "/r6bot_vel_controller/commands",
            connections=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        )


def main():
    # ros2 launch ros2_control_demo_example_7 r6bot_controller.launch.py
    rospy.init_node("giskard")
    robot_description = get_robot_description()
    giskard = Giskard(
        world_config=GenericWorldConfig(urdf=robot_description),
        robot_interface_config=R6BotInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=80),
    )
    giskard.live()


if __name__ == "__main__":
    main()
