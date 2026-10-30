"""
Driving griplink grippers, such as those of the real DAiSy, through their griplink
action servers.

.. note:: Importing this module needs ``griplink_interfaces``.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.plans.plan_transformation import ConfigurationTransformation
from coraplex.robot_plans.mixins import OperatesGripper
from giskardpy.motion_statechart.ros2_nodes.griplink import GriplinkActionServerTask
from semantic_digital_twin.robots.griplink_gripper import GriplinkGripper

# %% driving griplink grippers through their action servers


@dataclass
class DriveGriplinkGrippersThroughTheirActionServers(
    ConfigurationTransformation[OperatesGripper]
):
    """
    Brings the griplink grippers of a robot into their configurations through the
    griplink action servers driving the gripper hardware, instead of with goals for
    their finger joints.
    """

    def is_applicable(self, plan_node: OperatesGripper) -> bool:
        return any(
            isinstance(end_effector, GriplinkGripper)
            for end_effector in plan_node.robot.all_end_effectors
        )

    def apply(self, plan_node: OperatesGripper) -> None:
        plan_node.gripper_goal_type = GriplinkActionServerTask


# %% running plans on the real DAiSy


@dataclass
class RealDAiSyPlanExecutor(RealRobotPlanExecutor):
    """
    Runs plans on the real DAiSy, whose grippers move through their griplink action
    servers: :attr:`plan_adaptations` is extended by
    :class:`DriveGriplinkGrippersThroughTheirActionServers`.
    """

    def __post_init__(self) -> None:
        self.plan_adaptations = [
            *self.plan_adaptations,
            DriveGriplinkGrippersThroughTheirActionServers(),
        ]
        super().__post_init__()
