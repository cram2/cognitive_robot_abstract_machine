from __future__ import annotations

from dataclasses import dataclass, field

from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Parallel
from coraplex.robot_plans.mixins import (
    GraspPoseSequence,
    HasVariablePoses,
    ReattachesBranches,
    HasApproachesGraspPoses,
    HasGraspDetectionThreshold,
    MovesToolCenterPoint,
    OperatesGripper,
    PlaceTuningParameters,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from krrood.patterns.field_metadata import JSONMetadata
from krrood_spatial_math.spatial_types import Pose


# %% placing a held object


@dataclass(eq=False, repr=False, kw_only=True)
class PlaceAction(
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
    HasVariablePoses,
    SequentialAction,
    HasApproachesGraspPoses,
    PlaceTuningParameters,
    HasGraspDetectionThreshold,
):
    """
    Places an object at a position with the arm that holds it.

    The grasp the arm holds the object by is measured when the place starts, so the
    place can be planned before the object is picked up.
    """

    arm: Arm
    """
    The arm that holds :attr:`object_designator`.
    """

    object_designator: HasGraspCandidates
    """
    The annotation of the object that should be placed.
    """

    target_location: Pose
    """
    Pose in the world at which the object should be placed.
    """

    grasp_release_threshold: float = 0.1
    """
    Maximum fraction of sampled rays between the gripper's fingers that may still hit
    :attr:`object_designator` for it to count as released (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """

    _tool_goals: GraspPoseSequence | None = field(
        default=None,
        init=False,
        repr=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )
    """
    The tool frame goals releasing the object, set to the release of the grasp measured
    when this place starts.

    Sent along with the place, because a receiver does not expand it again.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for :attr:`arm`'s gripper to touch the surroundings of the
            object, which it is around from setting it down until it has retracted, the
            moment the object is put back under the world included.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        reference_frame = self.target_location.reference_frame
        self._tool_goals = GraspPoseSequence(
            pre_grasp=self.create_variable_pose(
                name="pre_grasp", reference_frame=reference_frame
            ),
            grasp=self.create_variable_pose(
                name="grasp", reference_frame=reference_frame
            ),
            retreat=self.create_variable_pose(
                name="retreat", reference_frame=reference_frame
            ),
        )
        pre_grasp, grasp, retreat = self._tool_goals.poses
        # A release runs the grasp backwards: down from above the target, then out
        # along the way the grasp was approached.
        return [
            self.tool_center_point_goal(
                retreat,
                self.arm,
                max_linear_velocity=self.transport_linear_velocity,
            ),
            self.tool_center_point_goal(
                grasp,
                self.arm,
                max_linear_velocity=self.placing_linear_velocity,
            ),
            Parallel(
                nodes=[
                    self.gripper_goal_type.moving_gripper_to(
                        self.arm.end_effector.default_configuration(
                            GripperState.OPEN,
                            finger_velocity=self.release_opening_velocity,
                        )
                    ),
                    # Keeps the gripper in place while it opens, whatever else
                    # moves the robot meanwhile.
                    self.tool_center_point_goal(grasp, self.arm),
                ]
            ),
            *self.reattach(
                self.object_designator.root,
                self.world.root,
            ),
            # The gripper starts out around the object it has just let go of.
            self.tool_center_point_goal(
                pre_grasp,
                self.arm,
                max_linear_velocity=self.retract_linear_velocity,
            ),
        ]

    @property
    def variable_poses(self) -> list[Pose]:
        return self._tool_goals.poses

    def on_start(self) -> None:
        """
        Differs from :meth:`~coraplex.robot_plans.actions.base.Action.on_start` in also
        setting :attr:`_tool_goals` to the release of the grasp :attr:`arm` has on the
        object now.
        """
        super().on_start()
        end_effector = self.arm.end_effector
        grasp = GraspCandidate(
            self.object_designator,
            end_effector.body_T_grasp(body=self.object_designator.root),
        )
        place_sequence = self.grasp_pose_sequence(
            grasp.moved_to(self.target_location), end_effector, grasp
        )
        for variable_pose, pose in zip(self._tool_goals.poses, place_sequence.poses):
            self.set_variable_pose(variable_pose=variable_pose, value=pose)
