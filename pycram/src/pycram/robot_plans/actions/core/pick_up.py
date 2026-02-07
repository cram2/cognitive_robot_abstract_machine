from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Union, Optional, Type, Any, Iterable

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from ...motions.gripper import MoveGripperMotion, MoveTCPMotion
from ....config.action_conf import ActionConfig
from ....datastructures.enums import (
    Arms,
    MovementType,
    FindBodyInRegionMethod,
)
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....failures import ObjectNotGraspedError
from ....failures import ObjectNotInGraspingArea
from ....has_parameters import has_parameters
from ....language import SequentialPlan
from ....robot_description import ViewManager
from ....robot_plans.actions.base import ActionDescription
from ....utils import translate_pose_along_local_axis

logger = logging.getLogger(__name__)


@has_parameters
@dataclass
class ReachAction(ActionDescription):
    """
    Let the robot reach a specific pose.
    """

    target_pose: PoseStamped
    """
    Pose that should be reached.
    """

    arm: Arms
    """
    The arm that should be used for pick up
    """

    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object
    """

    object_designator: Body = None
    """
    Object designator_description describing the object that should be picked up
    """

    reverse_reach_order: bool = False

    def __post_init__(self):
        super().__post_init__()

    def execute(self) -> None:

        target_pre_pose, target_pose, _ = self.grasp_description._pose_sequence(
            self.target_pose, self.object_designator, reverse=self.reverse_reach_order
        )

        SequentialPlan(
            self.context,
            MoveTCPMotion(target_pre_pose, self.arm, allow_gripper_collision=False),
            MoveTCPMotion(
                target_pose,
                self.arm,
                allow_gripper_collision=False,
                movement_type=MovementType.CARTESIAN,
            ),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if object is contained in the gripper such that it can be grasped and picked up.
        """
        fingers_link_names = self.arm_chain.end_effector.fingers_link_names
        if fingers_link_names:
            if not is_body_between_fingers(
                self.object_designator,
                fingers_link_names,
                method=FindBodyInRegionMethod.MultiRay,
            ):
                raise ObjectNotInGraspingArea(
                    self.object_designator,
                    World.robot,
                    self.arm,
                    self.grasp_description,
                )
        else:
            logger.warning(
                f"Cannot validate reaching to pick up action for arm {self.arm} as no finger links are defined."
            )

    @classmethod
    def description(
        cls,
        target_pose: Union[Iterable[PoseStamped], PoseStamped],
        arm: Union[Iterable[Arms], Arms] = None,
        grasp_description: Union[Iterable[GraspDescription], GraspDescription] = None,
        object_designator: Union[Iterable[Body], Body] = None,
        reverse_reach_order: Union[Iterable[bool], bool] = False,
    ) -> PartialDesignator[ReachAction]:
        return PartialDesignator[ReachAction](
            ReachAction,
            target_pose=target_pose,
            arm=arm,
            grasp_description=grasp_description,
            object_designator=object_designator,
            reverse_reach_order=reverse_reach_order,
        )


@has_parameters
@dataclass
class PickUpAction(ActionDescription):
    """
    Let the robot pick up an object.
    """

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
    The GraspDescription that should be used for picking up the object
    """

    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

    def execute(self) -> None:
        SequentialPlan(
            self.context,
            MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm),
            ReachActionDescription(
                target_pose=PoseStamped.from_spatial_type(
                    self.object_designator.global_pose
                ),
                object_designator=self.object_designator,
                arm=self.arm,
                grasp_description=self.grasp_description,
            ),
            MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm),
        ).perform()
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot_view)

        # Attach the object to the end effector
        with self.world.modify_world():
            self.world.move_branch_with_fixed_connection(
                self.object_designator, end_effector.tool_frame
            )

        _, _, lift_to_pose = self.grasp_description.grasp_pose_sequence(
            self.object_designator
        )
        SequentialPlan(
            self.context,
            MoveTCPMotion(
                lift_to_pose,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.TRANSLATION,
            ),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if picked up object is in contact with the gripper.
        """
        if not has_gripper_grasped_body(self.arm, self.object_designator):
            raise ObjectNotGraspedError(
                self.object_designator, World.robot, self.arm, self.grasp_description
            )

    @classmethod
    def description(
        cls,
        object_designator: Union[Iterable[Body], Body],
        arm: Union[Iterable[Arms], Arms] = None,
        grasp_description: Union[Iterable[GraspDescription], GraspDescription] = None,
    ) -> PartialDesignator[PickUpAction]:
        return PartialDesignator[PickUpAction](
            PickUpAction,
            object_designator=object_designator,
            arm=arm,
            grasp_description=grasp_description,
        )


@has_parameters
@dataclass
class GraspingAction(ActionDescription):
    """
    Grasps an object described by the given Object Designator description
    """

    object_designator: Body
    """
    Object Designator for the object that should be grasped
    """
    arm: Arms
    """
    The arm that should be used to grasp
    """
    grasp_description: GraspDescription
    """
    The distance in meters the gripper should be at before grasping the object
    """

    def execute(self) -> None:
        pre_pose, grasp_pose, _ = self.grasp_description.grasp_pose_sequence(
            self.object_designator
        )

        SequentialPlan(
            self.context,
            MoveTCPMotion(pre_pose, self.arm),
            MoveGripperMotion(GripperState.OPEN, self.arm),
            MoveTCPMotion(grasp_pose, self.arm, allow_gripper_collision=True),
            MoveGripperMotion(
                GripperState.CLOSE, self.arm, allow_gripper_collision=True
            ),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        body = self.object_designator
        contact_links = body.get_contact_points_with_body(World.robot).get_all_bodies()
        arm_chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        gripper_links = arm_chain.end_effector.links
        if not any([link.name in gripper_links for link in contact_links]):
            raise ObjectNotGraspedError(
                self.object_designator, World.robot, self.arm, None
            )

    @classmethod
    def description(
        cls,
        object_designator: Union[Iterable[Body], Body],
        arm: Union[Iterable[Arms], Arms] = None,
        grasp_description: Union[
            Iterable[GraspDescription], GraspDescription
        ] = ActionConfig.grasping_prepose_distance,
    ) -> PartialDesignator[GraspingAction]:
        return PartialDesignator[GraspingAction](
            GraspingAction,
            object_designator=object_designator,
            arm=arm,
            grasp_description=grasp_description,
        )


ReachActionDescription = ReachAction.description
PickUpActionDescription = PickUpAction.description
GraspingActionDescription = GraspingAction.description
