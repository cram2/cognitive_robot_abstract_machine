from __future__ import annotations

import logging
from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional

from coraplex.locations.pose_validator import AreReachableBy, IsObjectReachableBy
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    and_,
    or_,
    not_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    MovementType,
    DetectionTechnique,
)
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree
from coraplex.exceptions import PerceptionTargetMissing
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
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


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

    grasp_pose: Pose
    """
    The grasp frame that should be reached, as
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspPoses.grasp_poses`
    defines it.
    """

    arm: Arms
    """
    The arm that should be used for pick up.
    """

    object_designator: Optional[HasGraspPoses] = None
    """
    The annotation of the object that should be picked up.
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
        if self.perceive_before_grasp and self.object_designator is None:
            raise PerceptionTargetMissing(self)
        object_body = self.object_designator.root if self.object_designator else None

        target_pre_pose, target_pose, _ = self.grasp_pose_sequence(
            self.grasp_pose,
            ViewManager.get_end_effector_view(self.arm, self.robot),
            self._grasp_in_body_frame(self.grasp_pose, object_body),
            reverse=self.reverse_reach_order,
        )
        children = [
            MoveToolCenterPointMotion(
                target_pre_pose,
                self.arm,
                allow_gripper_collision=True,
                max_linear_velocity=self.pre_approach_linear_velocity,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            ),
        ]
        if self.open_gripper_at_pre_pose:
            children.append(
                MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm)
            )
        if self.perceive_before_grasp:
            children.extend(
                [
                    LookAtAction(target_pose),
                    DetectAction(
                        DetectionTechnique.TYPES,
                        object_sem_annotation=type(self.object_designator),
                        accept_first_if_multiple=True,
                    ),
                ]
            )
        children.append(
            MoveToolCenterPointMotion(
                target_pose,
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
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The sequence in which the robot would reach the target pose needs to be
        achievable.
        """
        object_designator = kwargs["object_designator"]
        return and_(
            IsObjectReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=variables["arm"],
                object_designator=object_designator.root if object_designator else None,
                grasp_pose=kwargs["grasp_pose"],
                reverse=kwargs["reverse_reach_order"],
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The end effector needs to be close to the target pose.
        """
        end_effector = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        object_designator = kwargs["object_designator"]
        object_body = object_designator.root if object_designator else None
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
    HasApproachesGraspPoses,
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Let the robot pick up an object.
    """

    object_designator: HasGraspPoses
    """
    The annotation of the object that should be picked up.
    """

    arm: Arms
    """
    The arm that should be used for pick up.
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

    @property
    def grasp_pose(self) -> Pose:
        """
        The grasp frame the object is picked up at.

        Asked of the object rather than taken from the caller, so that an object which
        knows where it may be held -- a bowl by its rim -- cannot be grasped anywhere
        else by a caller who does not. Which of them is taken is the gripper's call.

        :return: The grasp of :attr:`object_designator` the gripper is closest to.
        """
        return ViewManager.get_end_effector_view(
            self.arm, self.robot
        ).grasp_poses_by_distance(self.object_designator)[0]

    def _grasp_attempt_plan(self) -> PlanNode:
        """
        :return: One reach-and-close attempt at grasping :attr:`object_designator`,
            without lifting it.
        """
        return sequential(
            children=[
                # defining the target_pose relative to the object ensures it stays correct even if the object pose is
                # updated after defining the goal
                ReachAction(
                    grasp_pose=self.grasp_pose,
                    object_designator=self.object_designator,
                    arm=self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                    pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                    final_approach_linear_velocity=self.final_approach_linear_velocity,
                    open_gripper_at_pre_pose=True,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                    perceive_before_grasp=self.perceive_before_grasp,
                ),
                MoveGripperMotion(
                    motion=GripperState.CLOSE,
                    gripper=self.arm,
                    allow_gripper_collision=True,
                    finger_velocity=self.grasp_closing_velocity,
                    stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_stall=self.tolerate_grasp_stall,
                ),
                AttachNode(
                    body=self.object_designator.root,
                    new_parent=ViewManager.get_end_effector_view(
                        self.arm, self.robot
                    ).tool_frame,
                ),
            ],
        )

    @property
    def _action_plan(self) -> PlanNode:
        _, _, lift_to_pose = self.grasp_pose_sequence(
            self.grasp_pose,
            ViewManager.get_end_effector_view(self.arm, self.robot),
            self._grasp_in_body_frame(self.grasp_pose, self.object_designator.root),
        )
        return sequential(
            children=[
                self._grasp_attempt_plan(),
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
        The gripper with which to grasp the object needs to be free and the object needs
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
                arm=variables["arm"],
                object_designator=kwargs["object_designator"].root,
                grasp_pose=end_effector.grasp_poses_by_distance(
                    kwargs["object_designator"]
                )[0],
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict, context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be in the gripper frame.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return or_(
            not_(GripperIsFree(end_effector)),
            is_body_gripped(
                variable_from(kwargs["object_designator"].root),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
        )


@dataclass
class GraspingAction(ActionDescription, HasApproachesGraspPoses, HasTcpGoalThresholds):
    """
    Grasps an object described by the given Object Designator description.
    """

    object_designator: Body
    """
    Object Designator for the object that should be grasped.
    """

    arm: Arms
    """
    The arm that should be used to grasp.
    """

    grasp_pose: Pose
    """
    The grasp frame the object is grasped at.
    """

    @property
    def _action_plan(self) -> PlanNode:
        pre_pose, grasp_pose, _ = self.grasp_pose_sequence(
            self.grasp_pose,
            ViewManager.get_end_effector_view(self.arm, self.robot),
            self._grasp_in_body_frame(self.grasp_pose, self.object_designator),
        )

        return sequential(
            [
                MoveToolCenterPointMotion(
                    pre_pose,
                    self.arm,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                    allow_gripper_collision=True,
                ),
                MoveGripperMotion(GripperState.OPEN, self.arm),
                MoveToolCenterPointMotion(
                    grasp_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveGripperMotion(
                    GripperState.CLOSE, self.arm, allow_gripper_collision=True
                ),
            ]
        )
