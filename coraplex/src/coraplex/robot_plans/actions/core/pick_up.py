from __future__ import annotations

import logging
from dataclasses import dataclass

from typing_extensions import Any, Dict

from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    or_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    MovementType,
)
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import (
    GripperHolds,
    GripperIsFree,
    ToolFrameIsAtGrasp,
)
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
    GraspParameters,
    PickUpTuningParameters,
    ReachTuningParameters,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped

logger = logging.getLogger(__name__)


@dataclass
class ReachAction(
    ActionDescription,
    GraspParameters,
    ReachTuningParameters,
):
    """
    Let the robot reach a specific pose.
    """

    reverse_reach_order: bool = False
    """
    Whether to come down onto the grasp from the retreat pose above it, as a release
    does, instead of from the pre-grasp pose.
    """

    open_gripper_at_pre_pose: bool = False
    """
    Whether to open the gripper once the pre-pose is reached, used by
    :class:`PickUpAction` to open before its slower final approach.
    """

    @property
    def _action_plan(self) -> PlanNode:
        poses = self.grasp_pose_sequence(
            self.grasp.grasp_pose, self.arm.end_effector, self.grasp
        )
        pre_pose = poses.retreat if self.reverse_reach_order else poses.pre_grasp
        children = [
            MoveToolCenterPointMotion(
                pre_pose,
                arm=self.arm,
                allow_gripper_collision=True,
                max_linear_velocity=self.pre_approach_linear_velocity,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            ),
        ]
        if self.open_gripper_at_pre_pose:
            children.append(
                MoveGripperMotion(
                    motion=GripperState.OPEN, end_effector=self.arm.end_effector
                )
            )
        children.append(
            MoveToolCenterPointMotion(
                poses.grasp,
                arm=self.arm,
                allow_gripper_collision=True,
                max_linear_velocity=self.final_approach_linear_velocity,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            )
        )
        return sequential(children=children)

    def execute(self) -> Any:
        self.add_subplan(self.action_plan).perform()

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The end effector needs to be close to the target pose.
        """
        end_effector = kwargs["arm"].end_effector
        return or_(
            is_body_gripped(
                variable_from(kwargs["grasp"].graspable.root),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
            ToolFrameIsAtGrasp(end_effector, kwargs["grasp"]),
        )


@dataclass
class PickUpAction(
    ActionDescription,
    GraspParameters,
    PickUpTuningParameters,
):
    """
    Let the robot pick up an object: take hold of it and lift it clear of its support.
    """

    tolerate_grasp_stall: bool = False
    """
    Whether the CLOSE motion's completion also tolerates a stalled grasp (see
    :attr:`~coraplex.robot_plans.motions.gripper.MoveGripperMotion.tolerate_stall`).

    Opt-in rather than always on: building the stall monitor needs a velocity variable
    for every one of the gripper's connections, which is not guaranteed for every robot
    -- it crashes on Tracy's real-execution gripper, whose connections do not all have
    one.
    """

    def _grasp_attempt_plan(self) -> PlanNode:
        """
        A pick-up is a grasp the world is then told about: the object hangs off the tool
        frame afterwards, which is what makes it move with the arm.

        :return: One attempt at taking :attr:`grasp`, without lifting the object.
        """
        return sequential(
            children=[
                GraspingAction(
                    grasp=self.grasp,
                    arm=self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                    pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                    final_approach_linear_velocity=self.final_approach_linear_velocity,
                    grasp_closing_velocity=self.grasp_closing_velocity,
                    grasp_stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_grasp_stall=self.tolerate_grasp_stall,
                    grasp_detection_threshold=self.grasp_detection_threshold,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                ReAttachNode(
                    body=self.grasp.graspable.root,
                    new_parent=self.arm.end_effector.tool_frame,
                ),
            ],
        )

    @property
    def _action_plan(self) -> PlanNode:
        lift_to_pose = self.grasp_pose_sequence(
            self.grasp.grasp_pose, self.arm.end_effector, self.grasp
        ).retreat
        return sequential(
            children=[
                self._grasp_attempt_plan(),
                MoveToolCenterPointMotion(
                    lift_to_pose,
                    arm=self.arm,
                    allow_gripper_collision=True,
                    movement_type=MovementType.TRANSLATION,
                    max_linear_velocity=self.lift_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
            ],
        )

    @staticmethod
    def pre_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper needs to be free.
        """
        return GripperIsFree(variables["arm"].end_effector)

    @staticmethod
    def post_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object itself needs to be in the gripper, not merely something.
        """
        end_effector = variables["arm"].end_effector
        object_body = kwargs["grasp"].graspable.root
        return or_(
            GripperHolds(end_effector, object_body),
            is_body_gripped(
                variable_from(object_body),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
        )


@dataclass
class GraspingAction(
    ActionDescription,
    GraspParameters,
    PickUpTuningParameters,
):
    """
    Let the robot take hold of an object: reach onto a grasp and close on it.

    What a pick-up does before it lifts, and the whole of it when the object is meant to
    stay where it is -- a handle being pulled, say.
    """

    tolerate_grasp_stall: bool = False
    """
    Whether the CLOSE motion's completion also tolerates a stalled grasp (see
    :attr:`~coraplex.robot_plans.motions.gripper.MoveGripperMotion.tolerate_stall`).
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            children=[
                ReachAction(
                    grasp=self.grasp,
                    arm=self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                    pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                    final_approach_linear_velocity=self.final_approach_linear_velocity,
                    open_gripper_at_pre_pose=True,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                    grasp_detection_threshold=self.grasp_detection_threshold,
                ),
                MoveGripperMotion(
                    motion=GripperState.CLOSE,
                    end_effector=self.arm.end_effector,
                    allow_gripper_collision=True,
                    finger_velocity=self.grasp_closing_velocity,
                    stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_stall=self.tolerate_grasp_stall,
                ),
            ]
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Any], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper needs to be free.
        """
        return GripperIsFree(variables["arm"].end_effector)

    @staticmethod
    def post_condition(
        variables: Dict[str, Any], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be between the gripper's fingers, or the gripper at the
        grasp when a thin handle or rim leaves too little between them for the rays to
        see.
        """
        end_effector = variables["arm"].end_effector
        return or_(
            is_body_gripped(
                variable_from(kwargs["grasp"].graspable.root),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
            ToolFrameIsAtGrasp(end_effector, kwargs["grasp"]),
        )
