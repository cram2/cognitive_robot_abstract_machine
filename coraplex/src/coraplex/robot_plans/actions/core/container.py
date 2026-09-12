from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict

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
from coraplex.datastructures.enums import Arms
from coraplex.locations.pose_validator import IsReachableBy
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.motions.container import OpeningMotion, ClosingMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object.
    """

    object_designator: Handle
    """
    The handle of the container that should be opened.
    """
    arm: Arms
    """
    Arm that should be used for opening the container.
    """

    approach_clearance: float = ActionConfig.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                GraspingAction(
                    self.object_designator,
                    self.arm,
                    Pose(reference_frame=self.object_designator.root),
                    approach_clearance=self.approach_clearance,
                ),
                OpeningMotion(self.object_designator.root, self.arm),
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
            IsReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                pose=end_effector.tool_frame_goal(
                    Pose(reference_frame=kwargs["object_designator"].root)
                ),
                tip_link=end_effector.tool_frame,
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
        handle_body = kwargs["object_designator"].root
        parent_connection = handle_body.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        return and_(
            or_(
                is_body_in_gripper(variable_from(handle_body), end_effector)
                > 0.9,
                allclose(
                    variable_from(handle_body).global_pose.to_position(),
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

    object_designator: Handle
    """
    The handle of the container that should be closed.
    """

    arm: Arms
    """
    Arm that should be used for closing.
    """

    approach_clearance: float = ActionConfig.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                GraspingAction(
                    self.object_designator,
                    self.arm,
                    Pose(reference_frame=self.object_designator.root),
                    approach_clearance=self.approach_clearance,
                ),
                ClosingMotion(self.object_designator.root, self.arm),
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
        The container has to be closed.
        """
        close_connection = kwargs[
            "object_designator"
        ].root.get_first_parent_connection_of_type(ActiveConnection1DOF)

        return variable_from(close_connection).position < 0.1
