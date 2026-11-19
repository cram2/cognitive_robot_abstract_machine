"""
Joint-driven screwing actions for the Coraplex statechart API.

A :class:`~semantic_digital_twin.world_description.connections.ScrewConnection` couples
rotation and translation into one degree of freedom, so :class:`ScrewJointAction`,
:class:`ScrewThreadAction` and :class:`InsertScrewAction` drive the screw joint directly.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass

from coraplex.robot_plans.actions.base import (
    Action,
    GoalAction,
    SequentialAction,
)
from coraplex.robot_plans.actions.core.exceptions import (
    MissingScrewDisplacement,
    MissingSingleDofConnection,
    NegativeScrewTurns,
    NonToolGraspForScrewing,
    UnsupportedScrewConnection,
    ZeroAlignmentAxis,
)
from coraplex.robot_plans.actions.core.pick_up import HasGraspChoice
from coraplex.robot_plans.mixins import (
    HasApproachesGraspPoses,
    MovesToolCenterPoint,
    OperatesGripper,
    PlaceTuningParameters,
    ReattachesBranches,
)
from cramph.composites import Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bolt,
    Tool,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    ScrewConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% spatial and thread geometry helpers


def offset_pose_along_local_z(*, pose: Pose, distance: float) -> Pose:
    """
    Translate a pose along its own Z axis, keeping its orientation.

    :param pose: The pose to move from.
    :param distance: Metres to move along local +Z; negative backs off.
    :return:``pose`` translated along its local Z axis in its reference frame.
    """
    delta = Pose.from_xyz_axis_angle(
        z=distance, axis=Vector3.Z(), angle=0.0, reference_frame=None
    )
    return pose.homogeneous_matrix @ delta


def _vector_components(*, vector: Vector3) -> tuple[float, float, float]:
    """
    :param vector: A constant spatial vector.
    :return: Its ``(x, y, z)`` components as plain floats.
    """
    return float(vector[0]), float(vector[1]), float(vector[2])


def align_local_z_rotation(*, axis: Vector3) -> Pose:
    """
    Compute the pure rotation that aligns a frame's local +Z axis onto ``axis``.

    :param axis: Non-zero direction vector to align local +Z with.
    :return: A pure rotation without a reference frame.
    :raises ZeroAlignmentAxis: If ``axis`` has zero length.
    """
    x, y, z = _vector_components(vector=axis)
    length = math.sqrt(x * x + y * y + z * z)
    if math.isclose(length, 0.0, abs_tol=1e-12):
        raise ZeroAlignmentAxis(axis=axis)
    x, y, z = x / length, y / length, z / length

    # Rotate local +Z onto ``axis`` via ``Z x axis = (-y, x, 0)``.
    cross_x, cross_y = -y, x
    cross_length = math.hypot(cross_x, cross_y)
    if math.isclose(cross_length, 0.0, abs_tol=1e-9):
        if z > 0.0:
            return Pose.from_xyz_axis_angle(
                axis=Vector3.Z(), angle=0.0, reference_frame=None
            )
        return Pose.from_xyz_axis_angle(
            axis=Vector3.X(), angle=math.pi, reference_frame=None
        )
    return Pose.from_xyz_axis_angle(
        axis=Vector3(cross_x / cross_length, cross_y / cross_length, 0.0),
        angle=math.atan2(cross_length, z),
        reference_frame=None,
    )


def compute_screw_joint_goal_angle(
    *,
    connection: ActiveConnection1DOF,
    turns: float | None = None,
    travel_distance: float | None = None,
) -> float:
    """
    Compute the joint angle that drives a screw joint by ``turns`` or
    ``travel_distance``.

    Positive ``turns`` advance inwards against an axis pointing from parent to screw.

    :param connection: The screw joint to compute the goal for.
    :param turns: Revolutions to screw inwards.
    :param travel_distance: Signed translation in metres; overrides ``turns``.
    :return: The signed joint angle in radians, clamped to the lower limit.
    """
    if not isinstance(connection, ScrewConnection):
        raise UnsupportedScrewConnection(connection=connection)
    if travel_distance is None:
        if turns is None:
            raise MissingScrewDisplacement(connection=connection)
        if turns < 0.0:
            raise NegativeScrewTurns(turns=turns)
        travel_distance = -turns * connection.screw_pitch

    angle = float(connection.rotation_angle_for_travel_distance(travel_distance))
    lower_limit = connection.dof.limits.lower.position
    if lower_limit is not None:
        angle = max(float(lower_limit), angle)
    return angle


# %% joint-driven screwing actions


@dataclass(eq=False, repr=False, kw_only=True)
class TargetsScrewBody(Action, ABC):
    """
    Shared base for actions that drive a threaded body's single-DoF parent joint.
    """

    arm: Arm
    """
    The arm that manipulates or turns the screw.
    """

    screw_body: Body
    """
    The threaded body whose parent :class:`ActiveConnection1DOF` is driven.
    """

    allow_gripper_collision: bool = True
    """
    Whether the end effector and any held body may touch the surroundings while
    screwing.
    """

    @property
    def screw_connection(self) -> ActiveConnection1DOF:
        """
        :return: The single-degree-of-freedom parent connection of :attr:`screw_body`.
        :raises MissingSingleDofConnection: If :attr:`screw_body` has no such parent.
        """
        world = self.screw_body._world
        entity: KinematicStructureEntity = self.screw_body
        while (
            world is not None
            and entity != world.root
            and entity.parent_connection is not None
        ):
            connection = entity.parent_connection
            if isinstance(connection, ActiveConnection1DOF):
                return connection
            entity = connection.parent
        raise MissingSingleDofConnection(body=self.screw_body)

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        Differs from
        :meth:`~coraplex.robot_plans.actions.base.Action.create_temporary_collision_rules`
        by allowing :attr:`arm`'s end effector and held bodies to touch surroundings.
        """
        if not self.allow_gripper_collision:
            return []
        rules: list[CollisionRule] = [
            AllowCollisionForEndEffector(end_effector=self.arm.end_effector)
        ]
        held = self.arm.end_effector.held_bodies
        if held:
            rules.append(AllowCollisionForBodies(allowed_collision_bodies=set(held)))
        return rules


@dataclass(eq=False, repr=False, kw_only=True)
class ScrewJointAction(TargetsScrewBody, GoalAction):
    """
    Turns a threaded body by driving its own degree of freedom with
    :class:`~giskardpy.motion_statechart.tasks.grasped_mechanism.ArticulateGraspedMechanism`.
    """

    turns: float = 1.0
    """
    Revolutions to screw inwards.

    Ignored when :attr:`travel_distance` is set.
    """

    travel_distance: float | None = None
    """
    Signed translation along the joint axis in metres, positive moving out of the
    thread.
    """

    @property
    def goal_joint_state(self) -> float:
        """
        :return: The joint angle in radians reaching the requested insertion.
        """
        return compute_screw_joint_goal_angle(
            connection=self.screw_connection,
            turns=self.turns,
            travel_distance=self.travel_distance,
        )

    def create_goal(self) -> StatechartNode:
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=self.screw_body,
            goal_joint_state=self.goal_joint_state,
            mechanism_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ScrewThreadAction(
    TargetsScrewBody, MovesToolCenterPoint, OperatesGripper, SequentialAction
):
    """
    Screws a threaded body in by turning its joint, either by hand or with a held
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Tool`.
    """

    turns: float = 1.0
    """
    Revolutions the screw stands inwards from its joint's zero position after screwing.
    """

    tool_grasp: GraspCandidate | None = None
    """
    Grasp by which :attr:`arm` already holds a driving tool, or ``None`` to turn by
    hand.
    """

    grasp_local_pose: Pose | None = None
    """
    Engage pose of the gripper or tool tip in :attr:`screw_body`'s frame; ``None`` uses
    :attr:`default_grasp_local_pose`.
    """

    grasp_offset: float | None = None
    """
    Metres along :attr:`screw_body`'s local +X axis where the gripper or tip engages.
    """

    pre_approach_distance: float = 0.05
    """
    Metres to back off along the thread axis before moving onto the screw.
    """

    retract_distance: float = 0.05
    """
    Metres to back off along the thread axis after screwing.
    """

    release: bool = True
    """
    Whether to open the gripper before retracting; ignored while holding a tool.
    """

    max_linear_velocity: float | None = None
    """
    Linear speed cap in m/s for the point-to-point moves; ``None`` leaves it uncapped.
    """

    @property
    def insertion_axis(self) -> Vector3:
        """
        :return: Direction the screw advances inwards, i.e. the negated joint axis.
        """
        x, y, z = _vector_components(vector=self.screw_connection.axis)
        return Vector3(-x, -y, -z)

    @property
    def default_grasp_local_pose(self) -> Pose:
        """
        :return: Local pose with +Z aligned to :attr:`insertion_axis` and optional +X
            offset.
        """
        rotation = align_local_z_rotation(axis=self.insertion_axis)
        offset = self.grasp_offset
        if offset is None and self.screw_body._world is not None:
            bolts = [
                bolt
                for bolt in self.screw_body._world.get_semantic_annotations_by_type(
                    Bolt
                )
                if bolt.root is self.screw_body
            ]
            if bolts:
                offset = bolts[0].head_offset
        if offset is None:
            return rotation
        translation = Pose.from_xyz_axis_angle(x=offset)
        return (translation.homogeneous_matrix @ rotation.homogeneous_matrix).pose

    @property
    def engage_pose(self) -> Pose:
        """
        :return: Where the end effector or tool tip meets the screw, relative to
            :attr:`screw_body`.
        """
        if self.grasp_local_pose is not None:
            return self.grasp_local_pose
        screw_frame = HomogeneousTransformationMatrix(reference_frame=self.screw_body)
        return (screw_frame @ self.default_grasp_local_pose.homogeneous_matrix).pose

    @property
    def tool_tip_T_tool_frame(self) -> Pose | None:
        """
        :return: Pose of the gripper's tool frame relative to the held tool's tip, or
            ``None`` without a tool.
        :raises NonToolGraspForScrewing: If :attr:`tool_grasp` does not belong to a
            :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Tool`.
        """
        if self.tool_grasp is None:
            return None
        tool = self.tool_grasp.graspable
        if not isinstance(tool, Tool):
            raise NonToolGraspForScrewing(graspable=tool)
        tool_T_tool_frame = self.arm.end_effector.tool_frame_goal_for_grasp(
            self.tool_grasp.grasp_pose
        )
        return tool.root._world.transform(tool_T_tool_frame, tool.get_tool_frame())

    def tool_frame_goal(self, *, tip_goal: Pose) -> Pose:
        """
        :param tip_goal: Target pose for the end effector or the held tool's tip.
        :return: Matching goal pose for :attr:`arm`'s tool frame.
        """
        tip_T_tool_frame = self.tool_tip_T_tool_frame
        if tip_T_tool_frame is None:
            return tip_goal
        return (tip_goal.homogeneous_matrix @ tip_T_tool_frame.homogeneous_matrix).pose

    def create_steps(self) -> list[StatechartNode]:
        end_effector = self.arm.end_effector
        holds_tool = self.tool_grasp is not None
        engage_pose = self.engage_pose
        steps: list[StatechartNode] = [
            self.tool_center_point_goal(
                self.tool_frame_goal(
                    tip_goal=offset_pose_along_local_z(
                        pose=engage_pose, distance=-self.pre_approach_distance
                    )
                ),
                self.arm,
                max_linear_velocity=self.max_linear_velocity,
            ),
            self.tool_center_point_goal(
                self.tool_frame_goal(tip_goal=engage_pose),
                self.arm,
                max_linear_velocity=self.max_linear_velocity,
            ),
        ]
        if not holds_tool:
            steps.append(
                self.gripper_goal_type.create_gripper_goal(
                    configuration=end_effector.default_configuration(GripperState.CLOSE)
                )
            )
        steps.append(
            ScrewJointAction(
                screw_body=self.screw_body,
                arm=self.arm,
                turns=self.turns,
                allow_gripper_collision=self.allow_gripper_collision,
            )
        )
        if self.release and not holds_tool:
            steps.append(
                self.gripper_goal_type.create_gripper_goal(
                    configuration=end_effector.default_configuration(GripperState.OPEN)
                )
            )
        steps.append(
            self.tool_center_point_goal(
                self.tool_frame_goal(
                    tip_goal=offset_pose_along_local_z(
                        pose=engage_pose, distance=-self.retract_distance
                    )
                ),
                self.arm,
                max_linear_velocity=self.max_linear_velocity,
            )
        )
        return steps


# %% inserting a held screw into its hole


@dataclass(eq=False, repr=False, kw_only=True)
class InsertScrewAction(
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
    SequentialAction,
    HasGraspChoice,
    HasApproachesGraspPoses,
    PlaceTuningParameters,
):
    """
    Sets a held screw into its hole, reattaches it to :attr:`counterpart`, optionally
    gives it an initial turn by hand, then lets go and withdraws.

    The held screw must carry a
    :class:`~semantic_digital_twin.world_description.connections.ScrewConnection`,
    preserved across :class:`~cramph.world_modification_nodes.MoveBranch`.
    """

    counterpart: Body
    """
    The body the screw is threaded into, such as a plate or an airplane part.
    """

    target_location: Pose
    """
    The pose the screw takes once seated in its hole.
    """

    turn_by_hand: bool = True
    """
    Whether to turn the screw by :attr:`hand_turns` before letting go.
    """

    hand_turns: float = 0.1
    """
    Revolutions the screw stands inwards after turning by hand.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        Differs from
        :meth:`~coraplex.robot_plans.actions.base.Action.create_temporary_collision_rules`
        by allowing :attr:`arm`'s end effector to touch the hole surroundings.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        end_effector = self.arm.end_effector
        screw = self.grasp.graspable.root
        at_hole = self.grasp_pose_sequence(
            self.grasp.moved_to(self.target_location), end_effector, self.grasp
        )
        steps: list[StatechartNode] = [
            self.tool_center_point_goal(
                at_hole.retreat,
                self.arm,
                max_linear_velocity=self.transport_linear_velocity,
            ),
            self.tool_center_point_goal(
                at_hole.grasp,
                self.arm,
                max_linear_velocity=self.placing_linear_velocity,
            ),
            *self.reattach(screw, self.counterpart),
        ]
        if self.turn_by_hand:
            steps.append(
                ScrewJointAction(screw_body=screw, arm=self.arm, turns=self.hand_turns)
            )
        steps.extend(
            [
                Parallel(
                    nodes=[
                        self.gripper_goal_type.create_gripper_goal(
                            configuration=end_effector.default_configuration(
                                GripperState.OPEN,
                                finger_velocity=self.release_opening_velocity,
                            )
                        ),
                        self.tool_center_point_goal(at_hole.grasp, self.arm),
                    ]
                ),
                self.tool_center_point_goal(
                    at_hole.pre_grasp,
                    self.arm,
                    max_linear_velocity=self.retract_linear_velocity,
                ),
            ]
        )
        return steps
