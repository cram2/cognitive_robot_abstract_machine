from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.entity_query_language.factories import (
    variable_from,
    ConditionType,
)
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.exceptions import ObjectIsNotHeld
from coraplex.querying.predicates import GripperHolds
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Parallel
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.mixins import (
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
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types.spatial_types import Pose


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

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper holding :attr:`object_designator` to
            touch the surroundings of the object, which it is around from setting it down
            until it has retracted, the moment the object is put back under the world
            included.
        """
        arm, _ = self._holding_arm_and_grasp()
        return [AllowCollisionForEndEffector(end_effector=arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        arm, grasp = self._holding_arm_and_grasp()
        # A release runs the grasp backwards: down from above the target, then out
        # along the way the grasp was approached.
        poses = self.grasp_pose_sequence(
            grasp.moved_to(self.target_location), arm.end_effector, grasp
        )

        return [
            self.tool_center_point_goal(
                poses.retreat,
                arm,
                max_linear_velocity=self.transport_linear_velocity,
            ),
            self.tool_center_point_goal(
                poses.grasp,
                arm,
                max_linear_velocity=self.placing_linear_velocity,
            ),
            Parallel(
                nodes=[
                    self.gripper_goal_type.moving_gripper_to(
                        arm.end_effector.default_configuration(
                            GripperState.OPEN,
                            finger_velocity=self.release_opening_velocity,
                        )
                    ),
                    # Keeps the gripper in place while it opens, whatever else
                    # moves the robot meanwhile.
                    self.tool_center_point_goal(poses.grasp, arm),
                ]
            ),
            *self._retract_steps(arm, poses.pre_grasp),
        ]

    def _retract_steps(self, arm: Arm, retract_pose: Pose) -> list[StatechartNode]:
        """
        :param arm: The arm that placed the object.
        :param retract_pose: Where its tool frame withdraws to.
        :return: The steps that re-parent the placed object back to the world and
            retract the end effector away from it.
        """
        return [
            *self.reattach(
                self.object_designator.root,
                self.world.root,
            ),
            # The gripper starts out around the object it has just let go of.
            self.tool_center_point_goal(
                retract_pose,
                arm,
                max_linear_velocity=self.retract_linear_velocity,
            ),
        ]

    def _holding_arm_and_grasp(self) -> tuple[Arm, GraspCandidate]:
        """
        The arm that holds :attr:`object_designator`, and the grasp it holds it by.

        Read off the gripper while it holds the object; while the statechart is still
        being built, taken from the latest pick-up of the object before this place.

        :return: The arm and its grasp on the object.
        :raises ObjectIsNotHeld: If no arm holds the object and no pick-up precedes this
            place.
        """
        object_body = self.object_designator.root
        for arm in self.robot.all_arms:
            end_effector = arm.end_effector
            if GripperHolds(end_effector, object_body)():
                return arm, GraspCandidate(
                    self.object_designator, end_effector.held_body_T_grasp
                )
        for previous_pick in self.statechart.get_preceding_nodes_by_type(
            self, PickUpAction
        ):
            if previous_pick.grasp.graspable.root is object_body:
                return previous_pick.arm, previous_pick.grasp
        raise ObjectIsNotHeld(self.object_designator)

    def _grasp_on_the_held_object(self) -> GraspCandidate:
        """
        :return: The grasp :attr:`object_designator` is held by, as
            :meth:`_holding_arm_and_grasp` finds it.
        """
        return self._holding_arm_and_grasp()[1]

    @staticmethod
    def _grips_of_every_arm(
        context: StatechartContext, kwargs: dict[str, Any], threshold: float
    ) -> list[ConditionType]:
        """
        :param threshold: The fraction of rays between the fingers that has to hit the
            object for it to count as gripped.
        :return: For every arm of the robot, whether its gripper grips the object.
        """
        return [
            is_body_gripped(
                variable_from(kwargs["object_designator"].root),
                arm.end_effector,
                threshold=threshold,
            )
            for arm in context.require_extension(RobotAccess).robot.all_arms
        ]
