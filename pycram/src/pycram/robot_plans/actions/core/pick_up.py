from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_, or_, not_, variable_from
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    MovementType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.attachment_nodes import AttachNode
from pycram.plans.factories import sequential, execute_single
from pycram.plans.plan_node import PlanNode
from pycram.pose_validator import (
    pose_sequence_reachability_validator,
)
from pycram.querying.predicates import GripperIsFree
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class ReachAction(ActionDescription):
    """
    Let the robot reach a specific pose.
    """

    target_pose: Pose
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

    @property
    def action_plan(self) -> PlanNode:
        target_pre_pose, target_pose, _ = self.grasp_description._pose_sequence(
            self.target_pose, self.object_designator, reverse=self.reverse_reach_order
        )
        return sequential(
            children=[
                MoveToolCenterPointMotion(
                    target_pre_pose, self.arm, allow_gripper_collision=False
                ),
                MoveToolCenterPointMotion(
                    target_pose,
                    self.arm,
                    allow_gripper_collision=False,
                    movement_type=MovementType.CARTESIAN,
                ),
            ]
        )

    def execute(self) -> Any:
        self.add_subplan(self.action_plan).perform()

    # def execute(self) -> None:
    #
    # target_pre_pose, target_pose, _ = self.grasp_description._pose_sequence(
    #     self.target_pose, self.object_designator, reverse=self.reverse_reach_order
    # )
    # self.add_subplan(
    #     sequential(
    #         children=[
    #             MoveToolCenterPointMotion(
    #                 target_pre_pose, self.arm, allow_gripper_collision=False
    #             ),
    #             MoveToolCenterPointMotion(
    #                 target_pose,
    #                 self.arm,
    #                 allow_gripper_collision=False,
    #                 movement_type=MovementType.CARTESIAN,
    #             ),
    #         ]
    #     )
    # ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The sequence in which the robot would reach the target pose needs to be achiveable
        """
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
        """
        The end effector needs to be close to the target pose
        """
        manipulator = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        return or_(
            is_body_in_gripper(variable_from(kwargs["object_designator"]), manipulator)
            > 0.9,
            allclose(
                variable_from(kwargs["object_designator"].global_pose.to_position()),
                ViewManager.get_end_effector_view(
                    kwargs["arm"], context.robot
                ).tool_frame.global_pose.to_position(),
                atol=3e-2,
            ),
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

    @property
    def action_plan(self) -> PlanNode:

        _, _, lift_to_pose = self.grasp_description.grasp_pose_sequence(
            self.object_designator
        )
        return sequential(
            children=[
                MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm),
                ReachAction(
                    target_pose=self.object_designator.global_pose,
                    object_designator=self.object_designator,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ).action_plan,
                MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm),
                AttachNode(
                    body=self.object_designator,
                    new_parent=ViewManager.get_end_effector_view(
                        self.arm, self.robot
                    ).tool_frame,
                ),
                MoveToolCenterPointMotion(
                    lift_to_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    movement_type=MovementType.TRANSLATION,
                ),
            ]
        )

    def execute(self) -> Any:
        self.add_subplan(self.action_plan).perform()

    # def execute(self) -> None:
    #     self.add_subplan(
    #         sequential(
    #             children=[
    #                 MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm),
    #                 ReachAction(
    #                     target_pose=self.object_designator.global_pose,
    #                     object_designator=self.object_designator,
    #                     arm=self.arm,
    #                     grasp_description=self.grasp_description,
    #                 ),
    #                 MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm),
    #             ]
    #         )
    #     ).perform()
    #     end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
    #
    #     # Attach the object to the end effector
    #     with self.world.modify_world():
    #         self.world.move_branch_with_fixed_connection(
    #             self.object_designator, end_effector.tool_frame
    #         )
    #
    #     _, _, lift_to_pose = self.grasp_description.grasp_pose_sequence(
    #         self.object_designator
    #     )
    #     self.add_subplan(
    #         execute_single(
    #             MoveToolCenterPointMotion(
    #                 lift_to_pose,
    #                 self.arm,
    #                 allow_gripper_collision=True,
    #                 movement_type=MovementType.TRANSLATION,
    #             )
    #         )
    #     ).perform()

    @staticmethod
    def pre_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The gripper with which to grasp the object needs to be free and the object needs to be reachable
        """
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
        """
        The object needs to be in the griper frame
        """
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return or_(
            not_(GripperIsFree(manipulator)),
            is_body_in_gripper(kwargs["object_designator"], manipulator) > 0.9,
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

        self.add_subplan(
            sequential(
                [
                    MoveToolCenterPointMotion(pre_pose, self.arm),
                    MoveGripperMotion(GripperState.OPEN, self.arm),
                    MoveToolCenterPointMotion(
                        grasp_pose, self.arm, allow_gripper_collision=True
                    ),
                    MoveGripperMotion(
                        GripperState.CLOSE, self.arm, allow_gripper_collision=True
                    ),
                ]
            )
        ).perform()
