"""
Driving the grippers of the real PR2 through their
``pr2_controllers_msgs/Pr2GripperCommand`` action servers.

.. note:: Importing this module needs ``pr2_controllers_msgs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.plans.plan_transformation import ConfigurationTransformation
from coraplex.robot_plans.mixins import OperatesGripper
from giskardpy.motion_statechart.ros2_nodes.pr2_gripper import (
    PR2GripperActionServerTask,
)
from semantic_digital_twin.robots.pr2 import PR2Gripper

# %% driving PR2 grippers through their action servers


@dataclass(kw_only=True)
class DrivePR2GrippersThroughTheirActionServers(
    ConfigurationTransformation[OperatesGripper]
):
    """
    Brings the grippers of a PR2 into their configurations through the action servers
    driving the gripper hardware, instead of with goals for their finger joints.
    """

    def is_applicable(self, *, plan_node: OperatesGripper) -> bool:
        return any(
            isinstance(end_effector, PR2Gripper)
            for end_effector in plan_node.robot.all_end_effectors
        )

    def apply(self, *, plan_node: OperatesGripper) -> None:
        plan_node.gripper_goal_type = PR2GripperActionServerTask


# %% running plans on the real PR2


@dataclass(kw_only=True)
class RealPR2PlanExecutor(RealRobotPlanExecutor):
    """
    Runs plans on the real PR2, whose grippers move through their action servers:

    :attr:`plan_adaptations` starts with :class:`DrivePR2GrippersThroughTheirActionServers`,
    so the adaptations after it see the grippers' action goals.
    """

    def __post_init__(self) -> None:
        self.plan_adaptations = [
            DrivePR2GrippersThroughTheirActionServers(),
            *self.plan_adaptations,
        ]
        super().__post_init__()
