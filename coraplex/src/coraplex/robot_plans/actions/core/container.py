from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    and_,
    or_,
    variable_from,
    ConditionType,
)
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.motions.container import (
    ClosingMotion,
    OpeningMotion,
)
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be opened
    """
    arm: Arms
    """
    Arm that should be used for opening the container.
    """

    approach_direction: ApproachDirection = ApproachDirection.FRONT
    """
    The side of the handle the gripper approaches from.
    """

    goal_joint_state: Optional[float] = None
    """
    How far the container is opened.

    None opens it as far as its own limit allows.
    """

    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters the gripper should be at in the x-axis away from the handle.
    """

    @property
    def _action_plan(self) -> PlanNode:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        end_effector = arm.end_effector

        grasp_description = GraspDescription(
            self.approach_direction,
            VerticalAlignment.NoAlignment,
            end_effector,
        )

        return sequential(
            [
                GraspingAction(self.object_designator, self.arm, grasp_description),
                OpeningMotion(self.object_designator, self.arm, self.goal_joint_state),
                MoveGripperMotion(
                    GripperState.OPEN, self.arm, allow_gripper_collision=True
                ),
            ]
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper with which to open the container has to be free and the handle has
        to be reachable.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            IsObjectReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=kwargs["arm"],
                object_designator=kwargs["object_designator"],
                as_single_grasp=True,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The handle has to be in the gripper of the robot and the container has to be
        open.
        """
        end_effector = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        parent_connection = kwargs[
            "object_designator"
        ].get_first_parent_connection_of_type(ActiveConnection1DOF)
        return and_(
            or_(
                is_body_in_gripper(
                    variable_from(kwargs["object_designator"]), end_effector
                )
                > 0.9,
                allclose(
                    variable_from(
                        kwargs["object_designator"]
                    ).global_pose.to_position(),
                    variable_from(end_effector.tool_frame).global_pose.to_position(),
                    atol=3e-2,
                ),
            ),
            variable_from(parent_connection).position > 0.3,
        )


@dataclass
class CloseAction(ActionDescription):
    """
    Closes a container like object.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be closed.
    """

    arm: Arms
    """
    Arm that should be used for closing.
    """

    approach_direction: ApproachDirection = ApproachDirection.FRONT
    """
    The side of the handle the gripper approaches from.
    """

    goal_joint_state: float = 0.01
    """
    How far the container is left open.
    """

    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters between the gripper and the handle before approaching to
    grasp.
    """

    goal_joint_state_tolerance: float = 0.09
    """
    How far short of its goal, in the unit of its own degree of freedom, the container
    may stop and still count as closed.

    The last stretch onto a mechanism's own limit is only approached asymptotically, so
    a container that has to arrive exactly never reports that it got there.
    """

    @property
    def _action_plan(self) -> PlanNode:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        end_effector = arm.end_effector

        grasp_description = GraspDescription(
            self.approach_direction,
            VerticalAlignment.NoAlignment,
            end_effector,
        )

        return sequential(
            [
                GraspingAction(self.object_designator, self.arm, grasp_description),
                ClosingMotion(self.object_designator, self.arm, self.goal_joint_state),
                MoveGripperMotion(
                    GripperState.OPEN, self.arm, allow_gripper_collision=True
                ),
            ]
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The container has to be closed as far as it was asked to be.
        """
        close_connection = kwargs[
            "object_designator"
        ].get_first_parent_connection_of_type(ActiveConnection1DOF)

        return (
            variable_from(close_connection).position
            < kwargs["goal_joint_state"] + kwargs["goal_joint_state_tolerance"]
        )
