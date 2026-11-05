"""
Reusable parameters for actions: the inputs an action is given and the knobs that tune
how it carries them out, and the behaviour several actions share.

.. note:: This module does not use ``from __future__ import annotations``, so the
    fields an action inherits from it carry their types as type objects rather than as
    strings that would have to be resolved against the action's own module.
"""

from abc import ABC
from dataclasses import dataclass, field

import numpy as np


from coraplex.robot_plans.actions.base import Action
from cramph.node import StatechartNode
from cramph.world_modification_nodes import MoveBranch
from cramph.composites import Parallel
from giskardpy.motion_statechart.goals.gripper import (
    MoveFingers,
    MovesGripperToConfiguration,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.mixins import HasPoseThresholdParameters
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Arm, Camera, EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Pose2D,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass(kw_only=True, eq=False)
class DesignatorParameterMixin:
    """
    Base of the reusable action parameters: mixins that each add the fields of one
    parameter, and the helpers that read them, to an action.
    """


# %% what an action acts on


@dataclass(kw_only=True, eq=False)
class HasArmParameter(DesignatorParameterMixin):
    """
    Mixin for actions that operate one of the robot's arms.
    """

    arm: Arm
    """
    The arm the action uses.
    """


@dataclass(kw_only=True, eq=False)
class HasCameraParameter(DesignatorParameterMixin):
    """
    Mixin for actions that point a camera.
    """

    camera: Camera | None = None
    """
    The camera the action points; ``None`` selects the robot's default camera.
    """


@dataclass(kw_only=True, eq=False)
class HasGraspCandidateParameter(DesignatorParameterMixin):
    """
    Mixin for actions that close a gripper on something. The grasp names the object it
    is on, so that is not asked for separately.
    """

    grasp: GraspCandidate
    """
    The grasp to take hold by.

    One of the object's own
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.IsGraspable.grasp_candidates`.
    """


@dataclass(kw_only=True, eq=False)
class HasToolParameter(DesignatorParameterMixin):
    """
    Mixin for actions that manipulate an object with a held tool.
    """

    tool: Tool
    """
    The tool the action uses.
    """


# %% where an action goes


@dataclass(kw_only=True, eq=False)
class HasNavigationTargetParameter(DesignatorParameterMixin):
    """
    Mixin for actions that drive the robot's base to a spot on the floor.
    """

    target_location: Pose2D
    """
    Where the robot's base ends up on the floor, and which way the robot faces, given as
    the heading of the pose.
    """


@dataclass(kw_only=True, eq=False)
class HasTargetPoseParameter(DesignatorParameterMixin):
    """
    Mixin for actions that bring something an arm moves to a pose.
    """

    target_pose: Pose
    """
    The pose the action brings what it moves to.
    """


@dataclass(kw_only=True, eq=False)
class HasLookTargetParameter(DesignatorParameterMixin):
    """
    Mixin for actions that orient the robot toward a pose.
    """

    target: Pose
    """
    The pose the action orients the robot toward.
    """


# %% robot states an action sets


@dataclass(kw_only=True, eq=False)
class HasGripperConfigurationParameter(DesignatorParameterMixin):
    """
    Mixin for actions that bring a gripper into a configuration.
    """

    configuration: GripperConfiguration
    """
    The configuration the gripper is to reach.
    """


@dataclass(kw_only=True, eq=False)
class HasTorsoStateParameter(DesignatorParameterMixin):
    """
    Mixin for actions that set the torso to a defined state.
    """

    torso_state: TorsoState
    """
    The torso state the action sets.
    """


@dataclass(kw_only=True, eq=False)
class HasGripperCollisionParameter(DesignatorParameterMixin):
    """
    Mixin for actions that may permit the gripper to collide with the environment.
    """

    allow_gripper_collision: bool = False
    """
    Whether the gripper is allowed to touch its surroundings during the action.
    """


# %% thresholds


@dataclass(kw_only=True, eq=False)
class HasGraspDetectionThresholdParameter(DesignatorParameterMixin):
    """
    Mixin for actions that check whether an object is held between the gripper's
    fingers.
    """

    grasp_detection_threshold: float = 0.9
    """
    Minimum fraction of sampled rays between the gripper's fingers that must hit the
    target object for it to count as grasped/held (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """


# %% per-action tuning


@dataclass(kw_only=True, eq=False)
class HasReachTuningParameters(DesignatorParameterMixin):
    """
    Mixin for actions that reach towards a target, with tunable approach speeds.
    """

    pre_approach_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for the initial pre-pose approach, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    final_approach_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for the final approach onto the target pose, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass(kw_only=True, eq=False)
class HasPickUpTuningParameters(DesignatorParameterMixin):
    """
    Mixin for actions that take hold of an object, with tunable grasp and lift speeds
    and target-object friction.
    """

    grasp_closing_velocity: float | None = None
    """
    Maximum finger joint velocity (in m/s) used while closing onto the object, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    lift_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for lifting the object clear of the table after
    grasping, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    object_friction: float | None = None
    """
    Sliding friction coefficient to apply to the target object's geom before this pick,
    overriding the world's default. Not consumed by this action itself -- applying it is
    the caller's responsibility (see
    :meth:`~physics_simulators.mujoco_simulator.MujocoSimulator.set_geom_friction`);
    recorded here for persistence. ``None`` leaves the friction untouched.
    """


@dataclass(kw_only=True, eq=False)
class HasPlaceTuningParameters(DesignatorParameterMixin):
    """
    Mixin for actions that put an object down, with tunable transport, placing, release
    and retract speeds.
    """

    placing_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for the final descent onto the target location,
    enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    transport_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for carrying the held object above the target
    location, before the final descent, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    release_opening_velocity: float | None = None
    """
    Maximum finger joint velocity (in m/s) used while opening the gripper to release
    the object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    retract_linear_velocity: float | None = None
    """
    Maximum linear speed (in m/s) for retracting the end effector away from the placed
    object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


# %% grasp approach poses


@dataclass
class GraspPoseSequence:
    """
    The tool frame goals that approach a grasp, reach it and withdraw from it.
    """

    pre_grasp: Pose
    """
    Where the gripper waits before it moves onto the grasp, clear of the object.
    """

    grasp: Pose
    """
    The tool frame goal at the grasp itself.
    """

    retreat: Pose
    """
    Where the gripper rises to when it leaves the grasp.
    """


@dataclass(kw_only=True, eq=False)
class HasGraspApproachParameters(DesignatorParameterMixin):
    """
    Turns a grasp frame (x-axis along the approach, see
    :class:`~semantic_digital_twin.grasping.grasp_candidates.GraspCandidate`) into the
    tool frame goals that approach it, reach it and withdraw from it.
    """

    approach_clearance: float = 0.1
    """
    The gap in meters between the object and the gripper at the pre-grasp pose.
    """

    retreat_distance: float = 0.1
    """
    How far in meters the gripper rises when it leaves a grasp or a placed object.
    """

    def grasp_pose_sequence(
        self,
        reference_T_grasp: Pose,
        end_effector: EndEffector,
        grasp: GraspCandidate | None = None,
    ) -> GraspPoseSequence:
        """
        :param reference_T_grasp: The grasp frame to reach; for a release, where the
            object is to be put.
        :param end_effector: The end effector that is to reach it.
        :param grasp: The grasp whose object the pre-grasp pose has to stay outside of;
            ``None`` keeps only :attr:`approach_clearance`.
        :return: The tool frame goals around the grasp.
        """
        tool_goal = end_effector.tool_frame_goal_for_grasp(reference_T_grasp)
        return GraspPoseSequence(
            pre_grasp=self.pre_grasp_pose(
                reference_T_grasp, end_effector, self._approach_distance(grasp)
            ),
            grasp=tool_goal,
            retreat=self._retreat_pose(reference_T_grasp, tool_goal),
        )

    @staticmethod
    def pre_grasp_pose(
        reference_T_grasp: Pose, end_effector: EndEffector, distance: float
    ) -> Pose:
        """
        The goal back along the direction the gripper approaches a grasp from, aimed at
        the point the fingers meet as the grasp itself is.

        :param reference_T_grasp: The grasp frame, whose x-axis is the approach.
        :param end_effector: The end effector that reaches it.
        :param distance: How far back along the approach direction, in meters.
        :return: The tool frame goal, in ``reference_T_grasp``'s frame.
        """
        grasp_T_pre_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(x=-distance)
        return end_effector.tool_frame_goal_for_grasp(
            (reference_T_grasp.homogeneous_matrix @ grasp_T_pre_grasp).pose
        )

    def _approach_distance(self, grasp: GraspCandidate | None) -> float:
        """
        :param grasp: The grasp on the object, or ``None`` when there is no object.
        :return: The distance in meters from the grasp back to the pre-grasp pose: the
            distance to the object's bounding box along the approach, plus
            :attr:`approach_clearance`.
        """
        if grasp is None or not grasp.graspable.root.has_collision():
            return self.approach_clearance
        return (
            self.distance_to_boundary(grasp.graspable.root, grasp.grasp_pose)
            + self.approach_clearance
        )

    @staticmethod
    def distance_to_boundary(body: Body, reference_T_grasp: Pose) -> float:
        """
        The distance the gripper has to retrace from a grasp before it leaves the
        bounding box of `body`.

        :param body: The body whose bounding box the gripper has to leave.
        :param reference_T_grasp: The grasp frame, whose x-axis is the approach.
        :return: The distance in meters, zero when the grasp already lies outside the
            box.
        """
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()

        body_T_grasp = body._world.transform(reference_T_grasp, body).to_np()
        grasp_position = body_T_grasp[:3, 3]
        # The grasp frame's x-axis is where the gripper comes from, so it retraces -x.
        retrace_direction = -body_T_grasp[:3, 0]
        intervals = (
            bounding_box.x_interval,
            bounding_box.y_interval,
            bounding_box.z_interval,
        )
        minimum = np.array([interval.lower for interval in intervals])
        maximum = np.array([interval.upper for interval in intervals])

        distances = [
            (
                (maximum[axis] if retrace_direction[axis] > 0 else minimum[axis])
                - grasp_position[axis]
            )
            / retrace_direction[axis]
            for axis in range(3)
            if not np.isclose(retrace_direction[axis], 0)
        ]
        return max(min(distances, default=0.0), 0.0)

    def _retreat_pose(self, reference_T_grasp: Pose, tool_goal: Pose) -> Pose:
        """
        The tool frame goal :attr:`retreat_distance` above the grasp along the world's
        z-axis (a grasp frame's own z-axis can lie flat).

        :param reference_T_grasp: The grasp frame that was reached.
        :param tool_goal: The tool frame goal at the grasp, whose orientation is kept.
        :return: The retreat pose, in ``reference_T_grasp``'s frame.
        """
        target = reference_T_grasp.reference_frame
        world = target._world
        world_T_grasp = world.transform(
            reference_T_grasp.homogeneous_matrix, world.root
        )
        world_T_lift = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=self.retreat_distance, reference_frame=world.root
        )
        return Pose(
            world.transform((world_T_lift @ world_T_grasp).position, target),
            tool_goal.quaternion,
            reference_frame=target,
        )


# %% behaviour shared by actions


@dataclass(eq=False, repr=False, kw_only=True)
class MovesToolCenterPoint(Action, HasPoseThresholdParameters, ABC):
    """
    An action that moves an arm's tool center point.
    """

    position_threshold: float = 0.005
    """
    Tighter than that of a pose goal, so an approach does not stop short of a small
    object.
    """

    orientation_threshold: float = 0.02
    """
    Looser than that of a pose goal.
    """

    def tool_center_point_goal(
        self,
        target: Pose,
        arm: Arm,
        max_linear_velocity: float | None = None,
        max_angular_velocity: float | None = None,
    ) -> MotionStatechartNode:
        """
        :param target: Where the tool center point should end up.
        :param arm: The arm whose tool center point is moved.
        :param max_linear_velocity: Cap on the speed of the tool center point, in m/s.
        :param max_angular_velocity: Cap on how fast it turns, in rad/s.
        :return: The goal moving the tool center point there, together with whatever
            speed caps were asked for, and the task keeping the base still, see :meth:`~coraplex.robot_plans.actions.base.Action.keep_base_still`.
            A goal that is a parallel of its tasks holds them itself; any other is put in
            a parallel with them.
        """
        root = self.controlled_root
        tip = arm.end_effector.tool_frame
        accompanying: list[MotionStatechartNode] = self._velocity_limits(
            root, tip, max_linear_velocity, max_angular_velocity
        )
        accompanying.extend(self.keep_base_still())
        goal = self._reach_target(target, root, tip)
        if not accompanying:
            return goal
        if isinstance(goal, Parallel):
            for node in accompanying:
                goal.add_node(node)
            return goal
        return Parallel(nodes=[goal, *accompanying], name=type(goal).__name__)

    def _reach_target(
        self,
        target: Pose,
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
    ) -> MotionStatechartNode:
        """
        :return: The task that brings `tip` to `target`.
        """
        return CartesianPose(
            root_link=root,
            tip_link=tip,
            goal=target,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )

    @staticmethod
    def _velocity_limits(
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        max_linear_velocity: float | None,
        max_angular_velocity: float | None,
    ) -> list[MotionStatechartNode]:
        """
        :return: The speed caps that were asked for, if any.
        """
        limits: list[MotionStatechartNode] = []
        if max_linear_velocity is not None:
            limits.append(
                CartesianPositionVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_linear_velocity=max_linear_velocity,
                )
            )
        if max_angular_velocity is not None:
            limits.append(
                CartesianRotationVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_angular_velocity=max_angular_velocity,
                )
            )
        return limits


@dataclass(eq=False, repr=False, kw_only=True)
class OperatesGripper(Action, ABC):
    """
    An action that brings a gripper into a configuration, such as opening or closing
    it.
    """

    gripper_goal_type: type[MovesGripperToConfiguration] = field(
        default=MoveFingers, init=False
    )
    """
    The kind of node bringing the gripper into the configurations this action asks for,
    as set by the plan transformations configuring the action.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ReattachesBranches(Action, ABC):
    """
    An action that moves a branch of the kinematic structure under a new parent in the
    world model, such as an object it picks up or a robot it carries along.
    """

    reattaches_in_world_model: bool = field(default=True, init=False)
    """
    Whether the world model is told that a branch moved under its new parent, rather than
    contact carrying it there.
    """

    def reattach(
        self,
        branch_root: KinematicStructureEntity,
        new_parent: KinematicStructureEntity,
    ) -> list[StatechartNode]:
        """
        :param branch_root: The root of the branch to move.
        :param new_parent: The entity the branch is moved under.
        :return: The node moving `branch_root` under `new_parent`, none unless this
            action :attr:`reattaches_in_world_model`.
        """
        if not self.reattaches_in_world_model:
            return []
        return [MoveBranch(branch_root=branch_root, new_parent=new_parent)]
