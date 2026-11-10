"""
Driving grippers with a parallel gripper command server, such as the Robotiq grippers of
the real Tracy and Tiago, through that server.

.. note:: Importing this module needs ``control_msgs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.plan_transformation import ConfigurationTransformation
from coraplex.robot_plans.mixins import OperatesGripper
from giskardpy.motion_statechart.ros2_nodes.robotiq import (
    RobotiqGripperActionServerTask,
)
from semantic_digital_twin.robots.robotiq_85_gripper import (
    HasParallelGripperCommandServer,
)

# %% driving grippers through their parallel gripper command servers


@dataclass
class DriveRobotiqGrippersThroughTheirActionServers(
    ConfigurationTransformation[OperatesGripper]
):
    """
    Brings the grippers of a robot that have a ``control_msgs/ParallelGripperCommand``
    server into their configurations through that server, instead of with goals for
    their finger joints.

    Grippers whose servers take other action messages, such as the griplink grippers of
    DAiSy, are left as they are.
    """

    def is_applicable(self, plan_node: OperatesGripper) -> bool:
        return any(
            isinstance(end_effector, HasParallelGripperCommandServer)
            for end_effector in plan_node.robot.all_end_effectors
        )

    def apply(self, plan_node: OperatesGripper) -> None:
        plan_node.gripper_goal_type = RobotiqGripperActionServerTask
