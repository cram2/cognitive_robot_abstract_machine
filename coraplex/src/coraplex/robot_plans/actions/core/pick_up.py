from __future__ import annotations

import logging
from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional, List

from coraplex.exceptions import DataclassException
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.plan_node import PlanNode
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
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential, execute_single

from coraplex.querying.predicates import GripperIsFree
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
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
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class ReachAction(
    ActionDescription,
    ReachTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Let the robot reach a specific pose.
    """

    target_pose: Pose
    """
    Pose that should be reached.
    """

    arm: Arms
    """
    The arm that should be used for pick up.
    """

    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object.
    """

    object_designator: Optional[HasRootBody] = None
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

    @property
    def _action_plan(self) -> PlanNode:
        object_body = self.object_designator.root if self.object_designator else None

        target_pre_pose, target_pose, _ = self.grasp_description.pose_sequence(
            self.target_pose, object_body, reverse=self.reverse_reach_order
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
                grasp_description=kwargs["grasp_description"],
                target_pose=kwargs["target_pose"],
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
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Let the robot pick up an object.
    """

    object_designator: HasRootBody
    """
    The annotation of the object that should be picked up.
    """

    arm: Arms
    """
    The arm that should be used for pick up.
    """

    grasp_description: GraspDescription
    """
    The GraspDescription that should be used for picking up the object.
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
        :return: One reach-and-close attempt at grasping :attr:`object_designator`,
            without lifting it.
        """
        return sequential(
            children=[
                # defining the target_pose relative to the object ensures it stays correct even if the object pose is
                # updated after defining the goal
                ReachAction(
                    target_pose=Pose(reference_frame=self.object_designator.root),
                    object_designator=self.object_designator,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                    pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                    final_approach_linear_velocity=self.final_approach_linear_velocity,
                    open_gripper_at_pre_pose=True,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveGripperMotion(
                    motion=GripperState.CLOSE,
                    gripper=self.arm,
                    allow_gripper_collision=True,
                    finger_velocity=self.grasp_closing_velocity,
                    stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_stall=self.tolerate_grasp_stall,
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
        _, _, lift_to_pose = self.grasp_description.grasp_pose_sequence(
            self.object_designator.root
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
                grasp_description=kwargs["grasp_description"],
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
class GraspingAction(ActionDescription, HasTcpGoalThresholds):
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

    grasp_description: GraspDescription
    """
    The grasp description that should be used to grasp the object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        pre_pose, grasp_pose, _ = self.grasp_description.grasp_pose_sequence(
            self.object_designator
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


@dataclass
class SimoxPickUpAction(ActionDescription):
    """
    Pick up an object using grasp poses from the Simox physics-based planner.

    Pipeline:
    1. Call Simox /plan_grasp — returns ALL physically valid grasps from all
       directions (top, front, back, left, right), each with a quality score.
       Results are stored as Dict[str, List[GraspPose]] grouped by approach direction.
    2. Build the trial order from ``preferred_approaches`` first, then any
       remaining directions not listed (as automatic fallback).
    3. Within each direction, try candidates best-quality-first.
    4. For each candidate: reach → close gripper → attach object → lift.
       If ANY step fails, detach the object, reopen the gripper, and continue.
    5. If all candidates in all directions fail, raise BodyUnfetchable.

    :param object_designator: The CoraPlex Body to grasp.
    :param arm: Which arm to use (Arms.RIGHT or Arms.LEFT).
    :param end_effector_name: Simox EEF name, e.g. 'r_gripper' or 'l_gripper'.
    :param kinematic_chain_name: Simox kinematic chain, e.g. 'RightArm'/'LeftArm'.
    :param preferred_approaches: Ordered list of preferred approach directions,
        e.g. ['right', 'front']. The system will try these first (in order),
        then fall back to any remaining directions automatically.
        Use None or [] to let the system decide based purely on Simox quality.
        Valid values: 'top', 'front', 'back', 'left', 'right'.
    :param robot_xml: Absolute path to pr2.xml (Simox robot wrapper).
    :param lift_height: Height in meters to lift the object after grasping.
    :param num_grasps_to_plan: Number of grasp candidates to request from Simox.
    :param quality_threshold: Minimum Simox quality score (0.0–1.0).
    """

    object_designator: Body
    arm: Arms
    end_effector_name: str
    kinematic_chain_name: str
    preferred_approaches: Optional[List[str]] = None
    """
    Ordered list of preferred approach directions, e.g. ['right', 'top'].

    System tries these first, then falls back to remaining directions automatically.
    None or [] means no preference — the system sorts all directions by best quality.
    """

    robot_xml: str = ""
    lift_height: float = 0.1
    num_grasps_to_plan: int = 50
    quality_threshold: float = 0.001

    # ── Lift height constants ─────────────────────────────────────────────────
    #   TOP grasps: wrist is already close to the shoulder's upper reach envelope;
    #               a small clearance (7 cm) is enough to clear the surface and
    #               prevents the arm from hitting its kinematic limit.
    #   SIDE / FRONT / BACK grasps: full configured lift_height is safe because
    #               the arm can travel vertically while keeping a side orientation.
    _TOP_LIFT_HEIGHT: float = 0.07

    def _adaptive_lift_height(self, grasp_pose) -> float:
        """
        Return a kinematically safe lift height in meters for the given grasp.

        For top-down grasps the arm is close to its upper reach envelope, so we use a
        short clearance (``_TOP_LIFT_HEIGHT``) that guarantees the object clears the
        supporting surface without hitting joint limits.

        For all other approach directions (front, back, left, right, unknown) the
        configured ``lift_height`` is used (default 0.12 m).

        :param grasp_pose: A ``GraspPose`` carrying the ``approach`` label set by the
            Simox interface.
        :return: Lift height in meters.
        """
        approach = getattr(grasp_pose, "approach", "")
        if approach == "top":
            logger.debug(
                "Top-down grasp detected — using reduced lift height %.2f m"
                " (configured %.2f m)",
                self._TOP_LIFT_HEIGHT,
                self.lift_height,
            )
            return self._TOP_LIFT_HEIGHT
        return self.lift_height

    @property
    def _action_plan(self) -> PlanNode:
        self.execute()
        return sequential([])

    def execute(self) -> None:  # noqa: WPS231 (complexity ok here)

        from coraplex.external_interfaces.simox_grasp_planner import (
            plan_grasps_for_body,
            DEFAULT_ROBOT_XML,
        )
        from coraplex.plans.failures import PlanFailure, BodyUnfetchable
        from coraplex.robot_plans.actions.core.robot_body import (
            MoveManipulatorAction,
            SetGripperAction,
        )
        from copy import deepcopy

        robot_xml = self.robot_xml or DEFAULT_ROBOT_XML
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)

        # 1. Get ALL physics-based grasp poses from Simox, grouped by approach direction.
        #    Format: {'top': [GraspPose, ...], 'right': [...], 'front': [...], ...}
        #    Within each group, poses are sorted best-quality-first by Simox wrench score.
        grasp_dict = plan_grasps_for_body(
            body=self.object_designator,
            arm=self.arm,
            end_effector_name=self.end_effector_name,
            kinematic_chain_name=self.kinematic_chain_name,
            world=self.world,
            robot_xml=robot_xml,
            num_grasps_to_plan=self.num_grasps_to_plan,
            quality_threshold=self.quality_threshold,
        )

        if not grasp_dict:
            raise BodyUnfetchable(body=self.object_designator, arm=self.arm)

        # 2. Build the trial order:
        #    a) preferred_approaches first (in the specified order, skipping any
        #       direction Simox has no grasps for)
        #    b) then all remaining directions sorted by their best quality score
        preferred = list(self.preferred_approaches or [])
        preferred_order = [d for d in preferred if d in grasp_dict]

        remaining = sorted(
            [d for d in grasp_dict if d not in preferred_order],
            key=lambda d: grasp_dict[d][0].quality if grasp_dict[d] else 0.0,
            reverse=True,
        )
        trial_order = preferred_order + remaining

        total_candidates = sum(len(grasp_dict[d]) for d in trial_order)
        logger.info(
            "SimoxPickUpAction: %d total candidates for '%s' — trial order: %s",
            total_candidates,
            self.object_designator.name,
            trial_order,
        )

        # 3. Open gripper before trying any pose
        self.add_subplan(
            execute_single(SetGripperAction(gripper=self.arm, motion=GripperState.OPEN))
        ).perform()

        # 4. Iterate directions in trial order, best-quality-first within each group.
        #    Full Reach → Close → Attach → Lift cycle is inside the retry loop.
        #    Any failure → detach, reopen gripper, continue to next candidate.
        last_failure: Exception = BodyUnfetchable(
            body=self.object_designator, arm=self.arm
        )
        attached = False
        candidate_index = 0

        for direction in trial_order:
            poses_in_direction = grasp_dict[direction]
            for grasp_pose in poses_in_direction:
                candidate_index += 1
                approach = direction
                quality = getattr(grasp_pose, "quality", 0.0)
                logger.info(
                    "SimoxPickUpAction: [%d/%d] approach=%s quality=%.4f",
                    candidate_index,
                    total_candidates,
                    approach,
                    quality,
                )
                attached = False
                try:
                    # a) Reach grasp pose
                    self.add_subplan(
                        execute_single(
                            MoveManipulatorAction(
                                target_pose=grasp_pose,
                                end_effector=end_effector,
                                allow_gripper_collision=True,
                            )
                        )
                    ).perform()

                    # b) Close gripper
                    self.add_subplan(
                        execute_single(
                            SetGripperAction(
                                gripper=self.arm, motion=GripperState.CLOSE
                            )
                        )
                    ).perform()

                    # c) Attach object in the digital twin so Giskard treats it as
                    #    rigidly held during the lift motion.
                    with self.world.modify_world():
                        self.world.move_branch_with_fixed_connection(
                            self.object_designator, end_effector.tool_frame
                        )
                    attached = True

                    # d) Lift — position-ONLY translation so Giskard can freely flex
                    #    the wrist and elbow without a rigid orientation constraint.
                    safe_height = self._adaptive_lift_height(grasp_pose)
                    lift_point = deepcopy(grasp_pose.to_position())
                    lift_point.z = float(lift_point.z) + safe_height

                    self.add_subplan(
                        execute_single(
                            MoveToolCenterPointMotion(
                                target=Pose.from_xyz_quaternion(
                                    pos_x=float(lift_point.x),
                                    pos_y=float(lift_point.y),
                                    pos_z=float(lift_point.z),
                                    quat_x=0.0,
                                    quat_y=0.0,
                                    quat_z=0.0,
                                    quat_w=1.0,
                                    reference_frame=grasp_pose.reference_frame,
                                ),
                                arm=self.arm,
                                allow_gripper_collision=True,
                                movement_type=MovementType.TRANSLATION,
                            )
                        )
                    ).perform()

                    logger.info(
                        "SimoxPickUpAction: ✓ picked up '%s' "
                        "(approach=%s, quality=%.4f, lift=%.2f m)",
                        self.object_designator.name,
                        approach,
                        quality,
                        safe_height,
                    )
                    return

                except (PlanFailure, DataclassException, RuntimeError) as plan_failure:
                    logger.warning(
                        "SimoxPickUpAction: [%d/%d] approach=%s FAILED — %s",
                        candidate_index,
                        total_candidates,
                        approach,
                        plan_failure,
                    )
                    last_failure = plan_failure

                    # Detach object if it was attached before failure
                    if attached:
                        try:
                            from semantic_digital_twin.world_description.connections import (
                                Connection6DoF,
                            )

                            world_root = self.world.root
                            obj_transform = self.world.compute_forward_kinematics(
                                world_root, self.object_designator
                            )
                            with self.world.modify_world():
                                self.world.remove_connection(
                                    self.object_designator.parent_connection
                                )
                                connection = Connection6DoF.create_with_dofs(
                                    parent=world_root,
                                    child=self.object_designator,
                                    world=self.world,
                                )
                                self.world.add_connection(connection)
                                connection.origin = obj_transform
                            attached = False
                        except (RuntimeError, KeyError, AttributeError) as detach_exc:
                            logger.warning(
                                "SimoxPickUpAction: could not detach '%s' — %s",
                                self.object_designator.name,
                                detach_exc,
                            )

                    # Reopen gripper so next candidate starts clean
                    try:
                        self.add_subplan(
                            execute_single(
                                SetGripperAction(
                                    gripper=self.arm, motion=GripperState.OPEN
                                )
                            )
                        ).perform()
                    except (PlanFailure, DataclassException, RuntimeError) as open_exc:
                        logger.warning(
                            "SimoxPickUpAction: could not reopen gripper — %s", open_exc
                        )

                    continue

        # 5. All candidates in all directions exhausted
        raise last_failure
