from dataclasses import dataclass
from typing import Optional, List

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from pycram.datastructures.enums import (
    Arms,
    MovementType,
    WaypointsMovementType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans.motions.base import BaseMotion
from pycram.utils import translate_pose_along_local_axis
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ReachMotion(BaseMotion):
    """ """

    object_designator: Body
    """
    Object designator_description describing the object that should be picked up
    """
    arm: Arms
    """
    The arm that should be used for pick up
    """
    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object
    """
    movement_type: MovementType = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """
    reverse_pose_sequence: bool = False
    """
    Reverses the sequence of poses, i.e., moves away from the object instead of towards it. Used for placing objects.
    """

    def _calculate_pose_sequence(self) -> List[PoseStamped]:
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot_view)

        target_pose = GraspDescription.get_grasp_pose(
            self.grasp_description, end_effector, self.object_designator
        )
        target_pose.rotate_by_quaternion(
            GraspDescription.calculate_grasp_orientation(
                self.grasp_description,
                end_effector.front_facing_orientation.to_np(),
            )
        )
        target_pre_pose = translate_pose_along_local_axis(
            target_pose,
            end_effector.front_facing_axis.to_np()[:3],
            -0.05,  # TODO: Maybe put these values in the semantic annotates
        )

        pose = PoseStamped.from_spatial_type(
            self.world.transform(target_pre_pose.to_spatial_type(), self.world.root)
        )

        sequence = [target_pre_pose, pose]
        return sequence.reverse() if self.reverse_pose_sequence else sequence

    def perform(self):
        pass

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        nodes = [
            CartesianPose(
                root_link=self.robot_view.root,
                tip_link=tip,
                goal_pose=pose.to_spatial_type(),
                threshold=0.005,
                name="Reach",
            )
            for pose in self._calculate_pose_sequence()
        ]
        motion_state_chart_nodes = self._only_allow_gripper_collision_rules(self.arm)
        motion_state_chart_nodes.append(Sequence(nodes=nodes))
        return Parallel(motion_state_chart_nodes)


@dataclass
class MoveGripperMotion(BaseMotion):
    """
    Opens or closes the gripper
    """

    motion: GripperState
    """
    Motion that should be performed, either 'open' or 'close'
    """
    arm_of_gripper: Arms
    """
    Name of the gripper that should be moved
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        arm = ViewManager().get_end_effector_view(self.arm_of_gripper, self.robot_view)

        motion_state_chart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm_of_gripper)
            if self.allow_gripper_collision
            else [ExternalCollisionAvoidance(robot=self.robot_view)]
        )

        motion_state_chart_nodes.append(
            JointPositionList(
                goal_state=arm.get_joint_state_by_type(self.motion),
                name=(
                    "OpenGripper"
                    if self.motion == GripperState.OPEN
                    else "CloseGripper"
                ),
            )
        )
        return Parallel(motion_state_chart_nodes)


@dataclass
class MoveTCPMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: PoseStamped
    """
    Target pose to which the TCP should be moved
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something
    """
    movement_type: Optional[MovementType] = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        root = (
            self.world.root
            if self.robot_view.full_body_controlled
            else self.robot_view.root
        )
        if self.movement_type == MovementType.TRANSLATION:
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.to_spatial_type().to_position(),
                name="MoveTCP",
            )
        else:
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target.to_spatial_type(),
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_ABOVE_CA,
            )

        motion_state_chart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm)
            if self.allow_gripper_collision
            else [ExternalCollisionAvoidance(robot=self.robot_view)]
        )
        motion_state_chart_nodes.append(task)
        return Parallel(motion_state_chart_nodes)


@dataclass
class MoveTCPWaypointsMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    waypoints: List[PoseStamped]
    """
    Waypoints the TCP should move along 
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        root = (
            self.world.root
            if self.robot_view.full_body_controlled
            else self.robot_view.root
        )
        nodes = [
            CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=pose.to_spatial_type(),
            )
            for pose in self.waypoints
        ]
        motion_state_chart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm)
            if self.allow_gripper_collision
            else [ExternalCollisionAvoidance(robot=self.robot_view)]
        )
        motion_state_chart_nodes.append(Sequence(nodes=nodes))
        return Parallel(motion_state_chart_nodes)
