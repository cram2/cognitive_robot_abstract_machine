"""
The action that puts down what an arm holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.exceptions import HeldBodyIsNotGraspable, ObjectIsNotHeld
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Parallel
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.mixins import (
    HasArmParameter,
    HasGraspApproachParameters,
    HasGraspDetectionThresholdParameter,
    HasPlaceTuningParameters,
    HasTargetPoseParameter,
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable


@dataclass(eq=False, repr=False, kw_only=True)
class PlaceAction(
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
    SequentialAction,
    HasArmParameter,
    HasTargetPoseParameter,
    HasGraspApproachParameters,
    HasPlaceTuningParameters,
    HasGraspDetectionThresholdParameter,
):
    """
    Puts down what :attr:`arm` holds, so that the object ends up at :attr:`target_pose`
    however it is held.
    """

    # TODO: nothing reads this field or the inherited grasp_detection_threshold; decide
    # whether PlaceAction needs them.
    grasp_release_threshold: float = 0.1
    """
    Maximum fraction of sampled rays between the gripper's fingers that may still hit
    the object for it to count as released (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper of :attr:`arm` to touch the surroundings of
            the object, which it is around from setting it down until it has retracted,
            the moment the object is put back under the world included.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        grasp = self._grasp_on_the_held_object()
        # A release runs the grasp backwards: down from above the target, then out
        # along the way the grasp was approached.
        poses = self.grasp_pose_sequence(
            grasp.moved_to(self.target_pose), self.arm.end_effector, grasp
        )

        return [
            self.tool_center_point_goal(
                poses.retreat,
                self.arm,
                max_linear_velocity=self.transport_linear_velocity,
            ),
            self.tool_center_point_goal(
                poses.grasp,
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
                    self.tool_center_point_goal(poses.grasp, self.arm),
                ]
            ),
            *self.reattach(grasp.graspable.root, self.world.root),
            # The gripper starts out around the object it has just let go of.
            self.tool_center_point_goal(
                poses.pre_grasp,
                self.arm,
                max_linear_velocity=self.retract_linear_velocity,
            ),
        ]

    def _grasp_on_the_held_object(self) -> GraspCandidate:
        """
        The grasp :attr:`arm` holds its object by when this place runs.

        That is the grasp of the latest pick-up of :attr:`arm` before this place;
        without one, the grasp its gripper has on what it holds now.

        :return: The grasp, which names the object.
        :raises ObjectIsNotHeld: If the gripper holds nothing and no pick-up of
            :attr:`arm` precedes this place.
        :raises HeldBodyIsNotGraspable: If what the gripper holds is the root of no
            annotation that can be grasped.
        """
        for previous_pick in self.statechart.get_preceding_nodes_by_type(
            self, PickUpAction
        ):
            if previous_pick.arm is self.arm:
                return previous_pick.grasp
        end_effector = self.arm.end_effector
        held_body = end_effector.held_body
        if held_body is None:
            raise ObjectIsNotHeld(self.arm)
        for graspable in self.world.get_semantic_annotations_by_type(IsGraspable):
            if graspable.root is held_body:
                return GraspCandidate(graspable, end_effector.held_body_T_grasp)
        raise HeldBodyIsNotGraspable(self.arm, held_body)
