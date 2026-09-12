from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, Optional

from coraplex.locations.pose_validator import AreReachableBy, IsGraspReachableBy
from coraplex.plans.attachment_nodes import ReAttachNode
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
from coraplex.exceptions import (
    GraspPoseMissing,
    OffersNoGrasp,
    PerceptionTargetMissing,
)
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
class HasGraspChoice:
    """
    Adds to an action the object it takes hold of and the grasp it does so by.

    Shared by every action that closes a gripper on something: which grasp is taken, and
    whether that grasp can be reached, are the same questions however much the action
    goes on to do with the object afterwards.
    """

    object_designator: HasGraspPoses
    """
    The annotation of the object to take hold of.
    """

    arm: Arms
    """
    The arm that should be used.
    """

    grasp_pose: Optional[Pose] = None
    """
    The grasp frame to take hold by, in the object's own frame.

    ``None`` takes the first grasp the object offers. A caller that wants a particular
    one -- because it worked out which is reachable from where the robot will stand --
    passes it, and gets that one.
    """

    def __post_init__(self):
        self.grasp_pose = self.resolve_grasp_pose(
            self.grasp_pose, self.object_designator
        )

    @staticmethod
    def resolve_grasp_pose(
        grasp_pose: Optional[Pose], object_designator: HasGraspPoses
    ) -> Pose:
        """
        The grasp an action described this way takes.

        A caller that named a grasp gets that one. A caller that named none gets the
        first grasp the object offers, which depends on the object alone -- deciding it
        by where the robot happens to stand would make the same description mean
        different things at different times. A caller wanting a considered choice makes
        it outside the action and passes it in.

        ..note:: Also reached statically, because a pre-condition is built from an
            action's parameters rather than from the action.

        :param grasp_pose: The grasp a caller settled on, or ``None``.
        :param object_designator: The annotation of the object being grasped.
        :raises OffersNoGrasp: When no grasp was named and the object offers none.
        :return: The grasp frame, in the object's own frame.
        """
        if grasp_pose is not None:
            return grasp_pose
        first_grasp = next(iter(object_designator.grasp_poses()), None)
        if first_grasp is None:
            raise OffersNoGrasp(object_designator)
        return first_grasp

    @staticmethod
    def can_take_hold(
        variables: Dict[str, Any], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper needs to be free, and the grasp the action takes needs to be
        reachable.

        :param variables: The action's bound variables.
        :param context: The context the check runs in.
        :param kwargs: The action's parameters.
        :return: The condition.
        """
        return and_(
            GripperIsFree(
                ViewManager.get_end_effector_view(variables["arm"], context.robot)
            ),
            IsGraspReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=variables["arm"],
                grasp_pose=HasGraspChoice.resolve_grasp_pose(
                    kwargs["grasp_pose"], kwargs["object_designator"]
                ),
                object_designator=kwargs["object_designator"].root,
            ),
        )


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

    arm: Arms
    """
    The arm that should be used for pick up.
    """

    grasp_pose: Optional[Pose] = None
    """
    The grasp frame that should be reached, as
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspPoses.grasp_poses`
    defines it, in :attr:`object_designator`'s own frame when there is one.

    ``None`` takes the first grasp :attr:`object_designator` offers, so a reach onto a
    bare pose has to name one.
    """

    object_designator: Optional[HasGraspPoses] = None
    """
    The annotation of the object that should be picked up.

    ``None`` reaches a pose with no object around it, a handle being pulled, say.
    """

    def __post_init__(self):
        if self.grasp_pose is None:
            if self.object_designator is None:
                raise GraspPoseMissing(self)
            self.grasp_pose = HasGraspChoice.resolve_grasp_pose(
                self.grasp_pose, self.object_designator
            )

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
        target_pre_pose, target_pose, _ = self.grasp_pose_sequence(
            self.grasp_pose,
            ViewManager.get_end_effector_view(self.arm, self.robot),
            self.grasp_pose if self.object_designator is not None else None,
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
            IsGraspReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=variables["arm"],
                grasp_pose=kwargs["grasp_pose"],
                object_designator=object_designator.root if object_designator else None,
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

    def _grasp_attempt_plan(self, grasp_pose: Pose) -> PlanNode:
        """
        :param grasp_pose: The grasp to attempt, so the attempt and the lift that
            follows it are built around the same one.
        :return: One attempt at grasping :attr:`object_designator`, without lifting it.

        A pick-up is a grasp the world is then told about: the object hangs off the tool
        frame afterwards, which is what makes it move with the arm.
        """
        return sequential(
            children=[
                GraspingAction(
                    object_designator=self.object_designator,
                    arm=self.arm,
                    grasp_pose=grasp_pose,
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
                    body=self.object_designator.root,
                    new_parent=ViewManager.get_end_effector_view(
                        self.arm, self.robot
                    ).tool_frame,
                ),
            ],
        )

    @property
    def _action_plan(self) -> PlanNode:
        grasp_pose = self.grasp_pose
        _, _, lift_to_pose = self.grasp_pose_sequence(
            grasp_pose,
            ViewManager.get_end_effector_view(self.arm, self.robot),
            grasp_pose,
        )
        return sequential(
            children=[
                self._grasp_attempt_plan(grasp_pose),
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
        The gripper needs to be free and a grasp this pick-up may take needs to be
        reachable.

        The same question the grasp it is built from asks, so it is asked once.
        """
        return HasGraspChoice.can_take_hold(variables, context, kwargs)

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
                    grasp_detection_threshold=self.grasp_detection_threshold,
                ),
                MoveGripperMotion(
                    motion=GripperState.CLOSE,
                    gripper=self.arm,
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
        The gripper needs to be free and a grasp this action may take needs to be
        reachable.
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
            variable_from(kwargs["object_designator"].root),
            ViewManager.get_end_effector_view(variables["arm"], context.robot),
            threshold=kwargs["grasp_detection_threshold"],
        )
