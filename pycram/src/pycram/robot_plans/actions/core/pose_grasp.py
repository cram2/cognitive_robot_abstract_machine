from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, ClassVar, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import and_
from pycram.datastructures.dataclasses import Context

from pycram.datastructures.enums import Arms, RetractDirection
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential, execute_single
from pycram.plans.failures import PlanFailure
from pycram.utils import translate_pose_along_local_axis
from semantic_digital_twin.spatial_types.spatial_types import Pose
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.robot_plans.motions.pose_grasp import (
    PoseGraspMotion,
    RetractMotion,
)
from pycram.view_manager import ViewManager
from pycram.querying.predicates import GripperIsFree
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    blocking,
    robot_holds_body,
)
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPose


@dataclass
class PoseGraspAction(ActionDescription):
    """
    Move the gripper to an object's annotated grasp pose and close the gripper.
    Requires the object to carry a HasGraspPose semantic annotation.
    """

    target_object: HasGraspPose
    """The object to grasp."""

    arm: Arms
    """The arm to use for grasping."""

    use_collision_avoidance: bool = False
    """Whether to enable collision avoidance during the grasp motion."""

    check_blocking: bool = False
    """Whether to skip grasp poses that are blocked by obstacles (IK + collision check)."""

    pre_grasp_offset: float = 0.1
    """Extra standoff (in meters) added beyond the object's edge for the pre-grasp pose."""

    collision_buffer_distance: Optional[float] = None
    """Override for the external-collision buffer (soft standoff) distance in meters.
    None keeps the robot's default."""

    _resolved_grasp_pose: ClassVar[Optional[Pose]] = None

    def _build_pose_sequence(
        self, grasp_pose: Pose, grasp_desc: GraspDescription, manipulator
    ) -> list:
        offset = grasp_desc.edge_offset(self.target_object.root) + self.pre_grasp_offset
        approach_axis = manipulator.front_facing_axis.to_list()[:3]
        pre_grasp_pose = translate_pose_along_local_axis(
            deepcopy(grasp_pose), approach_axis, -offset
        )
        return [pre_grasp_pose, grasp_pose]

    def _resolve_grasp_pose(self) -> Pose:
        if self.check_blocking:
            pose = next(
                (p for p in self.target_object.grasp_poses() if self._is_graspable(p)),
                None,
            )
        else:
            pose = next(self.target_object.grasp_poses(), None)
        if not isinstance(pose, Pose):
            raise PlanFailure()
        return pose

    def execute(self) -> None:
        manipulator = ViewManager.get_end_effector_view(self.arm, self.robot)
        grasp_desc = GraspDescription.calculate_grasp_descriptions(
            manipulator, Pose(reference_frame=self.target_object.root)
        )[0]
        self._resolved_grasp_pose = self._resolve_grasp_pose()
        pose_sequence = self._build_pose_sequence(
            self._resolved_grasp_pose, grasp_desc, manipulator
        )
        self.add_subplan(
            sequential(
                [
                    MoveGripperMotion(gripper=self.arm, motion=GripperState.OPEN),
                    PoseGraspMotion(
                        arm=self.arm,
                        pose_sequence=pose_sequence,
                        allowed_collision_bodies=list(self.target_object.bodies),
                        use_collision_avoidance=self.use_collision_avoidance,
                        collision_buffer_distance=self.collision_buffer_distance,
                    ),
                    MoveGripperMotion(gripper=self.arm, motion=GripperState.CLOSE),
                ]
            )
        ).perform()

    def _is_graspable(self, pose: Pose) -> bool:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        tool_frame = hand.tool_frame

        with self.world.reset_state_context():
            blocking_bods = blocking(
                pose.to_homogeneous_matrix(), self.world.root, tool_frame
            )

        target_bodies = set(self.target_object.bodies)
        actual_blocking = [
            collision_pair
            for collision_pair in blocking_bods
            if collision_pair.body_a not in target_bodies
            and collision_pair.body_b not in target_bodies
            and collision_pair.distance < 0
        ]

        return not actual_blocking

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """The hand must be free and the object must expose a grasp pose."""
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        has_grasp_pose = isinstance(
            next(kwargs["target_object"].grasp_poses(), None), Pose
        )
        return and_(GripperIsFree(manipulator), has_grasp_pose)

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """The tool frame must have reached the object's grasp location."""
        manipulator = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        return allclose(
            kwargs["target_object"].root.global_pose.to_position(),
            manipulator.tool_frame.global_pose.to_position(),
            atol=2e-2,
        )


@dataclass
class PoseGraspAndLiftAction(ActionDescription):
    """
    Complete grasping sequence: open gripper, approach via grasp pose, close gripper,
    attach object, then retract.
    Requires the object to carry a HasGraspPose semantic annotation
    """

    target: HasGraspPose
    """The object to grasp and lift."""

    arm: Arms
    """The arm to use for grasping."""

    retract_distance: float = 0.1
    """Distance to retract after grasping (in meters)."""

    retract_direction: RetractDirection = RetractDirection.WORLD_Z
    """Direction to retract along after grasping."""

    max_retract_velocity: float = 0.2
    """Maximum velocity during retract."""

    use_collision_avoidance: bool = False
    """Whether to enable collision avoidance during the grasp and retract motions."""

    check_blocking: bool = False
    """Whether to skip grasp poses that are blocked by obstacles (IK + collision check)."""

    pre_grasp_offset: float = 0.1
    """Extra standoff (in meters) added beyond the object's edge for the pre-grasp pose."""

    collision_buffer_distance: Optional[float] = None
    """Override for the external-collision buffer distance in meters.
    None keeps the robot's default."""

    def execute(self) -> None:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)

        self.add_subplan(
            execute_single(
                PoseGraspAction(
                    target_object=self.target,
                    arm=self.arm,
                    use_collision_avoidance=self.use_collision_avoidance,
                    check_blocking=self.check_blocking,
                    pre_grasp_offset=self.pre_grasp_offset,
                    collision_buffer_distance=self.collision_buffer_distance,
                )
            )
        ).perform()

        with self.world.modify_world():
            self.world.move_branch_with_fixed_connection(
                self.target.root, hand.tool_frame
            )

        self.add_subplan(
            execute_single(
                RetractMotion(
                    arm=self.arm,
                    allowed_collision_bodies=list(self.target.bodies),
                    distance=self.retract_distance,
                    direction=self.retract_direction,
                    reference_velocity=self.max_retract_velocity,
                    use_collision_avoidance=self.use_collision_avoidance,
                    collision_buffer_distance=self.collision_buffer_distance,
                )
            )
        ).perform()

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """The hand must be free and the object must expose a grasp pose."""
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        has_grasp_pose = isinstance(next(kwargs["target"].grasp_poses(), None), Pose)
        return and_(GripperIsFree(manipulator), has_grasp_pose)

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """The grasped object must be held by the robot after the lift."""
        return robot_holds_body(context.robot, kwargs["target"].root)
