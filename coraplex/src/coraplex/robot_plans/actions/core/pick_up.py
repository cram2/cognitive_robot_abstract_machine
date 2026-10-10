from __future__ import annotations

import logging
from dataclasses import dataclass

from typing_extensions import Any, Dict, List, Optional

from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import MovementType
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    ConditionType,
    or_,
    variable_from,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from coraplex.querying.predicates import (
    GripperHolds,
    GripperIsFree,
    ToolFrameIsAtGrasp,
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate

logger = logging.getLogger(__name__)


@dataclass
class HasGraspChoice:
    """
    Adds to an action the grasp it takes hold by.

    Shared by every action that closes a gripper on something. The grasp names the
    object it is on, so that is not asked for separately.
    """

    grasp: GraspCandidate
    """
    The grasp to take hold by.

    One of the object's own
    :meth:`~semantic_digital_twin.grasping.grasp_candidates.HasGraspCandidates.grasp_candidates`.
    """

    arm: Arm
    """
    The arm that should be used.
    """


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

    grasp: GraspCandidate
    """
    The grasp the tool frame should reach, which also names the object it is on.
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
        children.append(
            MoveToolCenterPointMotion(
                poses.grasp,
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


# %% Simox Pick-Up Action


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
    :param arm: Which arm to use (Arm instance or Arms enum).
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
    arm: Any
    end_effector_name: str
    kinematic_chain_name: str
    preferred_approaches: Optional[List[str]] = None
    robot_xml: str = ""
    lift_height: float = 0.1
    num_grasps_to_plan: int = 50
    quality_threshold: float = 0.001

    _TOP_LIFT_HEIGHT: float = 0.07

    def _adaptive_lift_height(self, grasp_pose: Any) -> float:
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

    def execute(self) -> None:
        from copy import deepcopy

        from coraplex.external_interfaces.simox_grasp_planner import (
            DEFAULT_ROBOT_XML,
            plan_grasps_for_body,
        )
        from coraplex.plans.failures import BodyUnfetchable, PlanFailure
        from coraplex.robot_plans.actions.core.robot_body import (
            MoveManipulatorAction,
            SetGripperAction,
        )
        from krrood.exceptions import DataclassException

        robot_xml = self.robot_xml or DEFAULT_ROBOT_XML

        if hasattr(self.arm, "end_effector"):
            end_effector = self.arm.end_effector
        else:
            arm_name = str(self.arm).lower()
            if "left" in arm_name:
                end_effector = self.robot.left_arm.end_effector
            else:
                end_effector = self.robot.right_arm.end_effector

        # 1. Get ALL physics-based grasp poses from Simox, grouped by approach direction.
        grasp_dict = plan_grasps_for_body(
            body=self.object_designator,
            arm=self.arm,
            end_effector_name=self.end_effector_name,
            kinematic_chain_name=self.kinematic_chain_name,
            robot_xml=robot_xml,
            num_grasps=self.num_grasps_to_plan,
            quality_threshold=self.quality_threshold,
        )

        if not grasp_dict:
            raise BodyUnfetchable(body=self.object_designator, arm=self.arm)

        # 2. Build the trial order:
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
            execute_single(
                SetGripperAction(gripper=end_effector, motion=GripperState.OPEN)
            )
        ).perform()

        # 4. Iterate directions in trial order, best-quality-first within each group.
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
                                gripper=end_effector, motion=GripperState.CLOSE
                            )
                        )
                    ).perform()

                    # c) Attach object in the digital twin
                    with self.world.modify_world():
                        self.world.move_branch_with_fixed_connection(
                            self.object_designator, end_effector.tool_frame
                        )
                    attached = True

                    # d) Lift
                    safe_height = self._adaptive_lift_height(grasp_pose)
                    lift_point = deepcopy(grasp_pose.position)
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
                                    gripper=end_effector, motion=GripperState.OPEN
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
