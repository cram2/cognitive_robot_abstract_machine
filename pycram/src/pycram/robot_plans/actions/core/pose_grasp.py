from dataclasses import dataclass, field
from typing import Optional, Any

from pycram.datastructures.enums import Arms, RetractDirection
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential, execute_single
from pycram.plans.failures import PlanFailure, BodyUnfetchable
from semantic_digital_twin.spatial_types.spatial_types import Pose
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.robot_plans.motions.pose_grasp import (
    PoseGraspMotion,
    RetractMotion,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    blocking,
)
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPose


@dataclass
class PoseGraspAction(ActionDescription):
    """
    Move the gripper to an object's annotated grasp pose and close the gripper.
    Requires the object to carry a HasGraspPose semantic annotation.
    """

    target: HasGraspPose
    """The object to grasp."""

    arm: Arms
    """The arm to use for grasping."""

    use_collision_avoidance: bool = True
    """Whether to enable collision avoidance during the grasp motion."""

    check_blocking: bool = True
    """Whether to skip grasp poses that are blocked by obstacles (IK + collision check)."""

    _resolved_grasp_pose: Optional[Pose] = field(default=None, init=False, repr=False)
    """Grasp pose resolved during precondition validation, consumed in execute."""

    def execute(self) -> None:
        manipulator = ViewManager.get_end_effector_view(self.arm, self.robot)
        grasp_desc = GraspDescription.calculate_grasp_descriptions(
            manipulator, Pose(reference_frame=self.target.root)
        )[0]
        pose_sequence = grasp_desc.approach_sequence(
            self._resolved_grasp_pose, self.target.root
        )
        self.add_subplan(
            sequential(
                [
                    MoveGripperMotion(gripper=self.arm, motion=GripperState.OPEN),
                    PoseGraspMotion(
                        arm=self.arm,
                        pose_sequence=pose_sequence,
                        allowed_collision_bodies=list(self.target.bodies),
                        use_collision_avoidance=self.use_collision_avoidance,
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

        target_bodies = set(self.target.bodies)
        actual_blocking = [
            collision_pair
            for collision_pair in blocking_bods
            if collision_pair.body_a not in target_bodies
            and collision_pair.body_b not in target_bodies
            and collision_pair.distance < 0
        ]

        return not actual_blocking

    def validate_precondition(self):
        if not isinstance(next(self.target.grasp_poses(), None), Pose):
            raise PlanFailure()
        if self.check_blocking:
            self._resolved_grasp_pose = next(
                (
                    pose
                    for pose in self.target.grasp_poses()
                    if self._is_graspable(pose)
                ),
                None,
            )
        else:
            self._resolved_grasp_pose = next(self.target.grasp_poses(), None)
        if self._resolved_grasp_pose is None:
            raise PlanFailure()

    def validate_postcondition(self, result: Optional[Any] = None):
        # TODO change when validate_postcondition() actually is called after base motions are finished
        pass
        # if not robot_holds_body(self.robot, self.target.root):
        #     raise BodyUnfetchable(self.target.root, self.arm)


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

    use_collision_avoidance: bool = True
    """Whether to enable collision avoidance during the grasp and retract motions."""

    check_blocking: bool = True
    """Whether to skip grasp poses that are blocked by obstacles (IK + collision check)."""

    def execute(self) -> None:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)

        self.add_subplan(
            execute_single(
                PoseGraspAction(
                    target=self.target,
                    arm=self.arm,
                    use_collision_avoidance=self.use_collision_avoidance,
                    check_blocking=self.check_blocking,
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
                )
            )
        ).perform()

    def validate_precondition(self):
        if not isinstance(next(self.target.grasp_poses(), None), Pose):
            raise PlanFailure()

    def validate_postcondition(self, result: Optional[Any] = None):
        # TODO change when validate_postcondition() actually is called after base motions are finished
        # if not robot_holds_body(self.robot, self.target.root):
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        if self.world.get_connection(hand.tool_frame, self.target.root) is None:
            raise BodyUnfetchable(self.target.root, self.arm)
