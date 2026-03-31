from dataclasses import dataclass, field
from typing import Union, Iterable, Optional, Any

from pycram.datastructures.enums import Arms
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.language import SequentialPlan
from pycram.failures import ObjectNotGraspedError, PlanFailure
from semantic_digital_twin.spatial_types.spatial_types import Pose
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.robot_plans.motions.pose_grasp import (
    PoseGraspMotion,
    RetractMotion,
    RetractDirection,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    robot_in_collision,
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

    pre_grasp_distance: float = 0.15
    """Distance to offset the pre-grasp pose along the negative z-axis of the grasp pose (in meters)."""

    use_collision_avoidance: bool = True
    """Whether to enable collision avoidance during the grasp motion."""

    _resolved_grasp_pose: Optional[Pose] = field(default=None, init=False, repr=False)
    """Grasp pose resolved during precondition validation, consumed in execute."""

    def execute(self) -> None:
        SequentialPlan(
            self.context,
            MoveGripperMotion(gripper=self.arm, motion=GripperState.OPEN),
            PoseGraspMotion(
                arm=self.arm,
                grasp_pose=self._resolved_grasp_pose,
                allowed_collision_bodies=list(self.target.bodies),
                pre_grasp_distance=self.pre_grasp_distance,
                use_collision_avoidance=self.use_collision_avoidance,
            ),
            MoveGripperMotion(gripper=self.arm, motion=GripperState.CLOSE),
        ).perform()

    def _is_graspable(self, pose: Pose) -> bool:
        return True
        hand = ViewManager.get_end_effector_view(self.arm, self.robot_view)
        tool_frame = hand.tool_frame

        return not blocking(pose, self.world.root, tool_frame)

    def validate_precondition(self):
        if not isinstance(next(self.target.grasp_poses(), None), Pose):
            raise PlanFailure(
                f"Cannot perform PoseGraspAction: {self.target} has no grasp pose set."
            )
        self._resolved_grasp_pose = next(
            (pose for pose in self.target.grasp_poses() if self._is_graspable(pose)),
            None,
        )
        if self._resolved_grasp_pose is None:
            raise PlanFailure(
                f"Cannot perform PoseGraspAction: all grasp poses for {self.target} are blocked."
            )

    def validate_postcondition(self, result: Optional[Any] = None):
        # TODO change when validate_postcondition() actually is called after base motions are finished
        pass
        # if not robot_holds_body(self.robot_view, self.target.root):
        #     raise ObjectNotGraspedError(self.target.root, self.robot_view, self.arm)

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        target: HasGraspPose,
        pre_grasp_distance: Union[Iterable[float], float] = 0.15,
        use_collision_avoidance: Union[Iterable[bool], bool] = True,
    ) -> PartialDesignator["PoseGraspAction"]:
        return PartialDesignator[PoseGraspAction](
            PoseGraspAction,
            arm=arm,
            target=target,
            pre_grasp_distance=pre_grasp_distance,
            use_collision_avoidance=use_collision_avoidance,
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

    pre_grasp_distance: float = 0.15
    """Distance to offset the pre-grasp pose along the negative z-axis of the grasp pose (in meters)."""

    retract_distance: float = 0.1
    """Distance to retract after grasping (in meters)."""

    retract_direction: RetractDirection = RetractDirection.WORLD_Z
    """Direction to retract along after grasping."""

    max_retract_velocity: float = 0.2
    """Maximum velocity during retract."""

    use_collision_avoidance: bool = True
    """Whether to enable collision avoidance during the grasp and retract motions."""

    def execute(self) -> None:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot_view)

        SequentialPlan(
            self.context,
            PoseGraspActionDescription(
                target=self.target,
                arm=self.arm,
                pre_grasp_distance=self.pre_grasp_distance,
                use_collision_avoidance=self.use_collision_avoidance,
            ),
        ).perform()

        with self.world.modify_world():
            self.world.move_branch_with_fixed_connection(
                self.target.root, hand.tool_frame
            )

        SequentialPlan(
            self.context,
            RetractMotion(
                arm=self.arm,
                allowed_collision_bodies=list(self.target.bodies),
                distance=self.retract_distance,
                direction=self.retract_direction,
                reference_velocity=self.max_retract_velocity,
                use_collision_avoidance=self.use_collision_avoidance,
            ),
        ).perform()

    def validate_precondition(self):
        if not isinstance(next(self.target.grasp_poses(), None), Pose):
            raise PlanFailure(
                f"Cannot perform PoseGraspAndLiftAction: {self.target} has no grasp pose set."
            )

    def validate_postcondition(self, result: Optional[Any] = None):
        # TODO change when validate_postcondition() actually is called after base motions are finished
        # if not robot_holds_body(self.robot_view, self.target.root):
        hand = ViewManager.get_end_effector_view(self.arm, self.robot_view)
        if self.world.get_connection(hand.tool_frame, self.target.root) is None:
            raise ObjectNotGraspedError(self.target.root, self.robot_view, self.arm)

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        target: HasGraspPose,
        pre_grasp_distance: Union[Iterable[float], float] = 0.15,
        retract_distance: Union[Iterable[float], float] = 0.1,
        retract_direction: Union[
            Iterable[RetractDirection], RetractDirection
        ] = RetractDirection.WORLD_Z,
        max_retract_velocity: Union[Iterable[float], float] = 0.2,
        use_collision_avoidance: Union[Iterable[bool], bool] = True,
    ) -> PartialDesignator["PoseGraspAndLiftAction"]:
        return PartialDesignator[PoseGraspAndLiftAction](
            PoseGraspAndLiftAction,
            arm=arm,
            target=target,
            pre_grasp_distance=pre_grasp_distance,
            retract_distance=retract_distance,
            retract_direction=retract_direction,
            max_retract_velocity=max_retract_velocity,
            use_collision_avoidance=use_collision_avoidance,
        )


PoseGraspActionDescription = PoseGraspAction.description
PoseGraspAndLiftActionDescription = PoseGraspAndLiftAction.description
