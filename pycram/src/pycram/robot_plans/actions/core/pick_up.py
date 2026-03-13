from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_, or_, not_
from pycram.datastructures.dataclasses import Context
from pycram.pose_validator import (
    reachability_validator,
    pose_sequence_reachability_validator,
)
from pycram.querying.predicates import GripperIsFree
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.world_description.world_entity import Body
from pycram.robot_plans.motions.gripper import MoveGripperMotion, MoveTCPMotion
from pycram.config.action_conf import ActionConfig
from pycram.datastructures.enums import (
    Arms,
    MovementType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped
from pycram.failures import ObjectNotGraspedError
from pycram.failures import ObjectNotInGraspingArea
from pycram.language import SequentialPlan
from pycram.view_manager import ViewManager
from pycram.robot_plans.actions.base import ActionDescription, DescriptionType
from pycram.utils import translate_pose_along_local_axis

logger = logging.getLogger(__name__)


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

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        test_world = deepcopy(context.world)
        grasp_pose_sequence = kwargs["grasp_description"]._pose_sequence(
            kwargs["target_pose"],
            kwargs["object_designator"],
            reverse=kwargs["reverse_reach_order"],
        )
        return and_(
            pose_sequence_reachability_validator(
                grasp_pose_sequence,
                manipulator.tool_frame,
                context.robot.from_world(test_world),
                test_world,
                context.robot.full_body_controlled,
            ),
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        manipulator = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        return is_body_in_gripper(
            kwargs["object_designator"], manipulator
        ) > 0.9 or np.allclose(
            kwargs["object_designator"].global_pose.to_position(),
            ViewManager.get_end_effector_view(
                kwargs["arm"], context.robot
            ).tool_frame.global_pose.to_position(),
            atol=3e-2,
        )

    @classmethod
    def description(
        cls,
        target_pose: DescriptionType[PoseStamped],
        arm: DescriptionType[Arms] = None,
        grasp_description: DescriptionType[GraspDescription] = None,
        object_designator: DescriptionType[Body] = None,
        reverse_reach_order: DescriptionType[bool] = False,
    ) -> PartialDesignator[ReachAction]:
        return PartialDesignator[ReachAction](
            ReachAction,
            target_pose=target_pose,
            arm=arm,
            grasp_description=grasp_description,
            object_designator=object_designator,
            reverse_reach_order=reverse_reach_order,
        )


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

    @staticmethod
    def pre_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        test_world = deepcopy(context.world)
        grasp_pose_sequence = kwargs["grasp_description"].grasp_pose_sequence(
            kwargs["object_designator"]
        )
        return and_(
            GripperIsFree(manipulator),
            pose_sequence_reachability_validator(
                grasp_pose_sequence,
                manipulator.tool_frame,
                context.robot.from_world(test_world),
                test_world,
                context.robot.full_body_controlled,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return or_(
            not_(GripperIsFree(manipulator)),
            is_body_in_gripper(kwargs["object_designator"], manipulator) > 0.9,
        )

    @classmethod
    def description(
        cls,
        object_designator: DescriptionType[Body],
        arm: DescriptionType[Arms] = None,
        grasp_description: DescriptionType[GraspDescription] = None,
    ) -> PartialDesignator[PickUpAction]:
        return PartialDesignator[PickUpAction](
            PickUpAction,
            object_designator=object_designator,
            arm=arm,
            grasp_description=grasp_description,
        )


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

    @classmethod
    def description(
        cls,
        object_designator: DescriptionType[Body],
        arm: DescriptionType[Arms] = None,
        grasp_description: DescriptionType[GraspDescription] = None,
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
