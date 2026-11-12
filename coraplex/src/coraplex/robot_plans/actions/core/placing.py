from __future__ import annotations

from dataclasses import dataclass, field, fields

from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Parallel
from coraplex.robot_plans.mixins import (
    GraspPoseSequence,
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
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose


# %% placing a held object


@dataclass(eq=False, repr=False, kw_only=True)
class PlaceAction(
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
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

    _tool_goals: list[HomogeneousTransformationMatrix] = field(
        default_factory=list,
        init=False,
        repr=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )
    """
    The tool frame goals releasing the object, as the transformations of
    :attr:`~coraplex.robot_plans.mixins.GraspPoseSequence.poses`, over float variables
    that are set to the release of the grasp measured when this place starts.

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
        self._tool_goals = [
            self._variable_tool_goal(index=index)
            for index in range(len(fields(GraspPoseSequence)))
        ]
        pre_grasp, grasp, retreat = (goal.pose for goal in self._tool_goals)
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

    def set_up(self) -> None:
        """
        Registers the float variables of :attr:`_tool_goals`.
        """
        super().set_up()
        for goal in self._tool_goals:
            self.context.float_variable_data.register_expression(expression=goal)

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
        release = self.grasp_pose_sequence(
            grasp.moved_to(self.target_location), end_effector, grasp
        )
        for goal, released in zip(self._tool_goals, release.poses):
            reference_T_goal = self.world.transform(
                released, self.target_location.reference_frame
            )
            self.context.float_variable_data.set_value(
                expression=goal, value=reference_T_goal.to_np()[:3, :4].T.flatten()
            )

    def _variable_tool_goal(self, *, index: int) -> HomogeneousTransformationMatrix:
        """
        :param index: Which of the tool goals of this place it is, which names its
            variables.
        :return: A new tool frame goal in the frame of :attr:`target_location`, over
            float variables of its own.
        """
        reference_T_goal = HomogeneousTransformationMatrix.create_with_variables(
            f"{self.name}/tool_goal_{index}"
        )
        reference_T_goal.reference_frame = self.target_location.reference_frame
        return reference_T_goal
