"""
The actions that open and close a container by moving its handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


from krrood.entity_query_language.factories import (
    a,
    variable_from,
    variable,
)
from krrood.entity_query_language.query.match import Match
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.mixins import (
    HasGraspApproach,
    HasApproachClearanceParameter,
    HasArmParameter,
    HasGraspCandidateParameter,
    MovesToolCenterPoint,
    OperatesGripper,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
)
from krrood_spatial_math.spatial_types import Pose, Pose2D
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

# %% taking a container by its handle


@dataclass(eq=False, repr=False, kw_only=True)
class MovesContainerByItsHandle(
    MovesToolCenterPoint,
    OperatesGripper,
    SequentialAction,
    HasGraspCandidateParameter,
    HasArmParameter,
    HasApproachClearanceParameter,
    ABC,
):
    """
    An action that takes hold of a container's handle by :attr:`grasp_candidate`, moves
    the container with it, lets go and clears the handle.
    """

    release_clearance: float = 0.05
    """
    How far in meters the gripper backs off once it has let go, past where it leaves the
    bounding box of the body :attr:`mechanism` moves.
    """

    @classmethod
    def from_handle(cls, *, handle: Handle, arm: Arm) -> Match:
        """
        :param handle: The handle of the container.
        :param arm: The arm that takes hold of the handle.
        :return: This action on `handle`, with the grasp left open among the grasps
            the handle offers, the closest to where the robot stands when it is grounded
            first.
        """
        # A variable, so the grasps are ranked when the action is grounded, from where
        # the robot stands then.
        where_the_robot_stands = variable(
            Pose2D, domain=[Pose2D(reference_frame=arm._robot.root)]
        )
        return a(cls)(
            grasp_candidate=variable_from(
                GraspsByCloseness(
                    graspable=handle,
                    standing_position=where_the_robot_stands,
                    number_of_grasps=None,
                )
            ),
            arm=arm,
        )

    @abstractmethod
    def _create_mechanism_goal(self) -> MotionStatechartNode:
        """
        :return: A new goal moving the container while its handle is held.
        """

    @property
    def mechanism(self) -> ActiveConnection1DOF:
        """
        The connection the container moves along while its handle is held.
        """
        return self.grasp_candidate.graspable.root.get_first_parent_connection_of_type(
            connection_type=ActiveConnection1DOF
        )

    def back_off_pose(self, *, grasp_pose: Pose, end_effector: EndEffector) -> Pose:
        """
        The tool frame goal that clears the released handle and the container, back
        along the way the gripper came in, so an open gripper still straddling the
        handle is drawn off it rather than across it.

        :param grasp_pose: The grasp frame the handle was held by.
        :param end_effector: The end effector that held it.
        :return: The pose the tool frame backs off to, in `grasp_pose`'s frame.
        """
        distance = (
            HasGraspApproach.distance_to_boundary(
                body=self.mechanism.child, reference_T_grasp=grasp_pose
            )
            + self.release_clearance
        )
        return HasGraspApproach.pre_grasp_pose(
            reference_T_grasp=grasp_pose, end_effector=end_effector, distance=distance
        )

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper on the handle to touch its surroundings,
            from taking hold of the handle until it has backed off it.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        end_effector = self.arm.end_effector
        return [
            GraspingAction(
                grasp_candidate=self.grasp_candidate,
                arm=self.arm,
                approach_clearance=self.approach_clearance,
            ),
            self._create_mechanism_goal(),
            self.gripper_goal_type.create_gripper_goal(
                configuration=end_effector.default_configuration(
                    state_type=GripperState.OPEN
                )
            ),
            self.tool_center_point_goal(
                target=self.back_off_pose(
                    grasp_pose=self.grasp_candidate.grasp_pose,
                    end_effector=end_effector,
                ),
                arm=self.arm,
            ),
        ]


# %% opening and closing


@dataclass(eq=False, repr=False, kw_only=True)
class OpenAction(MovesContainerByItsHandle):
    """
    Opens a container like object, driving its mechanism to its upper limit.
    """

    def _create_mechanism_goal(self) -> MotionStatechartNode:
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=self.grasp_candidate.graspable.root,
            goal_position=self.mechanism.dof.limits.upper.position,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class CloseAction(MovesContainerByItsHandle):
    """
    Closes a container like object.
    """

    closed_joint_position: float = 0.01
    """
    The joint position, in meters or radians, the container's mechanism is driven to, or
    its lower limit if that lies above.
    """

    def _create_mechanism_goal(self) -> MotionStatechartNode:
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=self.grasp_candidate.graspable.root,
            goal_position=self.closed_joint_position,
        )
