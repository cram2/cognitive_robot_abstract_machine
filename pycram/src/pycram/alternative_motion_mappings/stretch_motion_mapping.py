from copy import deepcopy

from nav2_msgs.action import NavigateToPose

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.open_close import Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import NavigateActionServerTask
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pointing import Pointing
from pycram.datastructures.enums import ExecutionType
from pycram.robot_plans import MoveToolCenterPointMotion, MoveMotion, ClosingMotion
from pycram.robot_plans.motions.base import AlternativeMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


class StretchMoveToolCenterPoint(MoveToolCenterPointMotion, AlternativeMotion[Stretch]):
    """
    Better motions for stretch to move the tool center point to the given goal, first rotates the base such that the
    gripper is pointing at the goal pose and then uses full body control to move the TCP to the goal.
    """

    execution_type = (ExecutionType.SIMULATED, ExecutionType.REAL)

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        goal_copy = deepcopy(self.target)
        goal_copy = self.world.transform(goal_copy, self.world.root)
        goal_point = goal_copy.to_position()
        goal_point.z = 0
        return Sequence(
            [
                Parallel(
                    [
                        Pointing(
                            root_link=self.world.root,
                            tip_link=self.robot.root,
                            goal_point=goal_point,
                            pointing_axis=Vector3(
                                0, -1, 0, reference_frame=self.robot.root
                            ),
                            binding_policy=GoalBindingPolicy.Bind_at_build,
                        ),
                        JointPositionList(
                            goal_state=JointState.from_str_dict(
                                {"joint_wrist_yaw": 0.0}, world=self.world
                            )
                        ),
                    ]
                ),
                Parallel(
                    [
                        # AlignPlanes(
                        #     root_link=self.world.root,
                        #     tip_link=self.robot.root,
                        #     tip_normal=Vector3(
                        #         1, 0, 0, reference_frame=self.robot.root
                        #     ),
                        #     goal_normal=Vector3(
                        #         -1, 0, 0, reference_frame=self.world.root
                        #     ),
                        # ),
                        CartesianPose(
                            root_link=self.world.root,
                            tip_link=tip,
                            goal_pose=self.target,
                        ),
                    ],
                ),
            ]
        )


class StretchMoveSim(MoveMotion, AlternativeMotion[Stretch]):
    """
    Different giskard goal for moving stretch to a goal pose, this uses a goal optimal for a diff drive
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):

        return DifferentialDriveBaseGoal(
            goal_pose=self.target,
        )


class StretchMoveReal(MoveMotion, AlternativeMotion[Stretch]):
    """
    Uses a Nav2 action server to move the base of the real HSRB
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> NavigateActionServerTask:
        return DifferentialDriveBaseGoal(goal_pose=self.target, threshold=0.1)
        return NavigateActionServerTask(
            target_pose=self.target,
            base_link=self.robot.root,
            action_topic="/navigate_to_pose",
            message_type=NavigateToPose,
        )


class StretchClose(ClosingMotion, AlternativeMotion[Stretch]):
    """
    Optimized close motion for the stretch robot. This puts the stretch directly in front of the container while holding
    the handle and then pushes it arm forward to close the container.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        cart = CartesianPose(
            name="Keep holding handle",
            root_link=self.object_part,
            tip_link=tip,
            goal_pose=Pose(reference_frame=tip),
        )
        align = AlignPlanes(
            root_link=self.world.root,
            tip_link=self.robot.root,
            goal_normal=Vector3(1, 0, 0, reference_frame=self.object_part),
            tip_normal=Vector3(0, -1, 0, self.robot.root),
        )
        close = Close(tip_link=tip, environment_link=self.object_part)
        return Parallel([cart, align, close])
