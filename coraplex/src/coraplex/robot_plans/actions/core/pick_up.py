from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, Optional

from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    or_,
    not_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    MovementType,
    DetectionTechnique,
)
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
    HasApproachesGraspPoses,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
    PickUpTuningParameters,
    ReachTuningParameters,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class HasGraspChoice:
    """
    Adds to an action the grasp it takes hold by.

    Shared by every action that closes a gripper on something: which grasp is taken, and
    whether the gripper is free to take it, are the same questions however much the
    action goes on to do with the object afterwards. The grasp names the object it is
    on, so that is not asked for separately.
    """

    grasp: GraspPose
    """
    The grasp to take hold by.

    One of the object's own
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspPoses.grasp_poses`.
    """

    arm: Arm
    """
    The arm that should be used.
    """

    @staticmethod
    def can_take_hold(
        variables: Dict[str, Any], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper needs to be free.

        :param variables: The action's bound variables.
        :param context: The context the check runs in.
        :param kwargs: The action's parameters.
        :return: The condition.
        """
        return GripperIsFree(variables["arm"].end_effector)


@dataclass
class ReachAction(
    ActionDescription,
    HasApproachesGraspPoses,
    ReachTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Let the robot reach a specific pose.
    """

    arm: Arm
    """
    The arm that should be used for pick up.
    """

    grasp: GraspPose
    """
    The grasp the tool frame should reach, which also names the object it is on.
    """

    reverse_reach_order: bool = False
    """
    Whether the grasp pose sequence should be approached in reverse order.
    """

    open_gripper_at_pre_pose: bool = False
    """
    Whether to open the gripper once the pre-pose is reached, used by
    :class:`PickUpAction` to open before its slower final approach.
    """

    perceive_before_grasp: bool = False
    """
    Whether to look at the target and detect the object before the final approach.

    When False the reach goes straight from the pre-pose to the target, grasping at the
    pose the world already holds.
    """

    @property
    def _action_plan(self) -> PlanNode:
        pre_grasp_pose, tool_goal, _ = self.grasp_pose_sequence(
            self.grasp.root_T_grasp,
            self.arm.end_effector,
            self.grasp,
            reverse=self.reverse_reach_order,
        )
        children = [
            MoveToolCenterPointMotion(
                pre_grasp_pose,
                self.arm,
                allow_gripper_collision=True,
                max_linear_velocity=self.pre_approach_linear_velocity,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            ),
        ]
        if self.open_gripper_at_pre_pose:
            children.append(
                MoveGripperMotion(
                    motion=GripperState.OPEN, gripper=self.arm.end_effector
                )
            )
        if self.perceive_before_grasp:
            children.extend(
                [
                    LookAtAction(tool_goal),
                    DetectAction(
                        DetectionTechnique.TYPES,
                        object_sem_annotation=type(self.grasp.graspable),
                        accept_first_if_multiple=True,
                    ),
                ]
            )
        children.append(
            MoveToolCenterPointMotion(
                tool_goal,
                self.arm,
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
        object_body = kwargs["grasp"].graspable.root
        return or_(
            is_body_gripped(
                variable_from(object_body),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
            allclose(
                variable_from(object_body).global_pose.to_position(),
                variable_from(end_effector.tool_frame).global_pose.to_position(),
                atol=3e-2,
            ),
        )


@dataclass
class PickUpAction(
    ActionDescription,
    HasGraspChoice,
    HasApproachesGraspPoses,
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
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

    perceive_before_grasp: bool = False
    """
    Whether to look at the object and detect it before the final approach.

    Passed on to the reach this pick-up is built from; see
    :attr:`ReachAction.perceive_before_grasp`.
    """

    def _grasp_attempt_plan(self, grasp: GraspPose) -> PlanNode:
        """
        :param grasp: The grasp to attempt, so the attempt and the lift that
            follows it are built around the same one.
        :return: One attempt at taking :attr:`grasp`, without lifting the object.

        A pick-up is a grasp the world is then told about: the object hangs off the tool
        frame afterwards, which is what makes it move with the arm.
        """
        return sequential(
            children=[
                GraspingAction(
                    grasp=grasp,
                    arm=self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                    pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                    final_approach_linear_velocity=self.final_approach_linear_velocity,
                    grasp_closing_velocity=self.grasp_closing_velocity,
                    grasp_stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_grasp_stall=self.tolerate_grasp_stall,
                    perceive_before_grasp=self.perceive_before_grasp,
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
        _, _, lift_to_pose = self.grasp_pose_sequence(
            self.grasp.root_T_grasp,
            self.arm.end_effector,
            self.grasp,
        )
        return sequential(
            children=[
                self._grasp_attempt_plan(self.grasp),
                MoveToolCenterPointMotion(
                    lift_to_pose,
                    self.arm,
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
        return HasGraspChoice.can_take_hold(variables, context, kwargs)

    @staticmethod
    def post_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be in the gripper frame.
        """
        end_effector = variables["arm"].end_effector
        return or_(
            not_(GripperIsFree(end_effector)),
            is_body_gripped(
                variable_from(kwargs["grasp"].graspable.root),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
        )


@dataclass
class GraspingAction(
    ActionDescription,
    HasGraspChoice,
    HasApproachesGraspPoses,
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
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

    perceive_before_grasp: bool = False
    """
    Whether to look at the object and detect it before the final approach.

    Passed on to the reach this grasp is built from; see
    :attr:`ReachAction.perceive_before_grasp`.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            children=[
                # The grasp is defined relative to the object, so it stays correct even
                # if the object's pose is updated after the goal was defined.
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
                    perceive_before_grasp=self.perceive_before_grasp,
                    grasp_detection_threshold=self.grasp_detection_threshold,
                ),
                MoveGripperMotion(
                    motion=GripperState.CLOSE,
                    gripper=self.arm.end_effector,
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
        return HasGraspChoice.can_take_hold(variables, context, kwargs)

    @staticmethod
    def post_condition(
        variables: Dict[str, Any], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be between the gripper's fingers.
        """
        return is_body_gripped(
            variable_from(kwargs["grasp"].graspable.root),
            variables["arm"].end_effector,
            threshold=kwargs["grasp_detection_threshold"],
        )
