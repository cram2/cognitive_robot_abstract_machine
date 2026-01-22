from nav2_msgs.action import NavigateToPose

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
)
from semantic_digital_twin.robots.abstract_robot import ParallelGripper
from semantic_digital_twin.robots.hsrb import HSRB
from ..datastructures.enums import ExecutionType
from ..robot_description import ViewManager
from ..robot_plans import MoveMotion, MoveTCPMotion

from ..robot_plans.motions.base import AlternativeMotion


class HSRBMoveMotion(MoveMotion, AlternativeMotion[HSRB]):
    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> NavigateActionServerTask:
        return NavigateActionServerTask(
            target_pose=self.target.to_spatial_type(),
            base_link=self.robot_view.root,
            action_topic="/hsrb/move_base",
            message_type=NavigateToPose,
        )


class HSRMoveTCPSim(MoveTCPMotion, AlternativeMotion[HSRB]):

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self) -> CartesianPose:
        tip = self.robot_view._world.get_semantic_annotations_by_type(ParallelGripper)[
            0
        ]
        return CartesianPose(
            root_link=self.world.root,
            tip_link=tip,
            goal_pose=self.target.to_spatial_type(),
        )


class HSRMoveTCPReal(MoveTCPMotion, AlternativeMotion[HSRB]):

    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> CartesianPose:
        # tip = self.robot_view.arms[0]
        # Temp hack from Simon
        tip = self.robot_view._world.get_semantic_annotations_by_type(ParallelGripper)[
            0
        ]
        return CartesianPose(
            root_link=self.world.root,
            tip_link=tip.tool_frame,
            goal_pose=self.target.to_spatial_type(),
        )
