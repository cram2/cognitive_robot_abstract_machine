"""
Reusable parameters for actions and motions: the inputs a behaviour is given and the
knobs that tune how it carries them out.

.. note:: This module does not use ``from __future__ import annotations``, so the
    fields a designator inherits from it carry their types as type objects rather than
    as strings that would have to be resolved against the designator's own module.
"""

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from coraplex.datastructures.enums import MovementType
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    CanBeGrasped,
)
from semantic_digital_twin.robots.robot_parts import Arm, Camera, EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Tool,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D


@dataclass(eq=False)
class DesignatorParameterMixin:
    """
    Base of the reusable designator parameters: mixins that add keyword-only fields, and
    the helpers that read them, to an action or motion.
    """


# %% behaviour parameters


@dataclass(eq=False)
class ArmParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that operate one of the robot's arms.
    """

    arm: Arm = field(kw_only=True)
    """
    The arm the behaviour uses.
    """


@dataclass(eq=False)
class GraspCandidateParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that close a gripper on something. The grasp names the object it
    is on, so that is not asked for separately.
    """

    grasp: GraspCandidate = field(kw_only=True)
    """
    The grasp to take hold by.

    One of the object's own
    :meth:`~semantic_digital_twin.grasping.grasp_candidates.HasGraspCandidates.grasp_candidates`.
    """


@dataclass(eq=False)
class GraspableObjectParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that act on a single graspable object.
    """

    object_designator: CanBeGrasped = field(kw_only=True)
    """
    The annotation of the object the behaviour acts on; its :attr:`root` body is used
    where the underlying kinematic body is required.
    """


@dataclass(eq=False)
class HandleParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that grasp and articulate a handle, such as opening or closing a
    container.
    """

    handle: Handle = field(kw_only=True)
    """
    The handle annotation the behaviour operates; its :attr:`root` body is used where the
    underlying kinematic body is required.
    """


@dataclass(eq=False)
class GripperCollisionParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that may permit the gripper to collide with the environment.
    """

    allow_gripper_collision: Optional[bool] = field(default=None, kw_only=True)
    """
    Whether the gripper is allowed to collide during the behaviour.
    """


@dataclass(eq=False)
class PlacementTargetParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that move an object to a destination pose.
    """

    target_location: Pose = field(kw_only=True)
    """
    The destination pose the behaviour moves to.
    """


@dataclass(eq=False)
class NavigationTargetParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that drive the robot's base to a spot on the floor.
    """

    target_location: Pose2D = field(kw_only=True)
    """
    Where the robot's base ends up: its position on the floor and its heading.
    """


@dataclass(eq=False)
class TargetPoseParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that drive an end effector to a target pose.
    """

    target_pose: Pose = field(kw_only=True)
    """
    The pose the end effector reaches.
    """


@dataclass(eq=False)
class LookTargetParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that orient the robot toward a pose.
    """

    target: Pose = field(kw_only=True)
    """
    The pose the behaviour orients toward.
    """


@dataclass(eq=False)
class MovementTypeParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours whose Cartesian motion follows a selectable movement type.
    """

    movement_type: MovementType = field(default=MovementType.CARTESIAN, kw_only=True)
    """
    The type of Cartesian movement the behaviour performs.
    """


@dataclass(eq=False)
class GripperStateParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that set the gripper to an open or closed state.
    """

    motion: GripperState = field(kw_only=True)
    """
    The gripper state the behaviour sets.
    """


@dataclass(eq=False)
class EndEffectorParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that act through a specific end effector.
    """

    end_effector: EndEffector = field(kw_only=True)
    """
    The end effector the behaviour uses.
    """


@dataclass(eq=False)
class CameraParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that point a camera.
    """

    camera: Optional[Camera] = field(default=None, kw_only=True)
    """
    The camera the behaviour points; ``None`` selects the robot's default camera.
    """


@dataclass(eq=False)
class ToolParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that manipulate an object with a held tool.
    """

    tool: Tool = field(kw_only=True)
    """
    The tool the behaviour uses.
    """


@dataclass(eq=False)
class TorsoStateParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that set the torso to a defined state.
    """

    torso_state: TorsoState = field(kw_only=True)
    """
    The torso state the behaviour sets.
    """


@dataclass(eq=False)
class GoalThresholdParameters(DesignatorParameterMixin):
    """
    Mixin for behaviours that count their tool-center-point goal as reached within a
    tolerance, falling back to
    :attr:`~coraplex.datastructures.dataclasses.Context.motion_tolerances` when left unset.

    Meant to be mixed into something carrying a ``context``, which the resolvers read.
    """

    position_threshold: Optional[float] = field(default=None, kw_only=True)
    """
    Distance threshold in meters for goal achievement. ``None`` falls back to
    :attr:`~coraplex.datastructures.dataclasses.MotionToleranceConfig.default_tcp_position_threshold`.
    """

    orientation_threshold: Optional[float] = field(default=None, kw_only=True)
    """
    Rotation threshold in rad for goal achievement. ``None`` falls back to
    :attr:`~coraplex.datastructures.dataclasses.MotionToleranceConfig.tool_orientation_threshold`.
    """

    def resolved_position_threshold(self) -> float:
        """
        :return: :attr:`position_threshold` if set, otherwise the context's default.
        """
        if self.position_threshold is not None:
            return self.position_threshold
        return self.context.motion_tolerances.default_tcp_position_threshold

    def resolved_orientation_threshold(self) -> float:
        """
        :return: :attr:`orientation_threshold` if set, otherwise the context's default.
        """
        if self.orientation_threshold is not None:
            return self.orientation_threshold
        return self.context.motion_tolerances.tool_orientation_threshold


@dataclass(eq=False)
class GraspDetectionThresholdParameter(DesignatorParameterMixin):
    """
    Mixin for behaviours that check whether an object is held between the gripper's
    fingers.
    """

    grasp_detection_threshold: float = field(default=0.9, kw_only=True)
    """
    Minimum fraction of sampled rays between the gripper's fingers that must hit the
    target object for it to count as grasped/held (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """


@dataclass(eq=False)
class MaxJointVelocityParameter(DesignatorParameterMixin):
    """
    Adds an optional joint velocity cap to an action or motion.

    .. note:: Stands on its own rather than joining one of the bundles below, because the
        behaviours that cap a joint speed share no other parameter: one names the arms, the
        other the joints it drives.
    """

    max_joint_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum joint velocity (in rad/s or m/s, per joint), enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
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


@dataclass(eq=False)
class GraspApproachParameters(DesignatorParameterMixin):
    """
    Turns a grasp frame (x-axis along the approach, see
    :class:`~semantic_digital_twin.grasping.grasp_candidates.GraspCandidate`) into the
    tool frame goals that approach it, reach it and withdraw from it.
    """

    approach_clearance: float = field(default=0.1, kw_only=True)
    """
    The gap in meters between the object and the gripper at the pre-grasp pose.
    """

    retreat_distance: float = field(default=0.1, kw_only=True)
    """
    How far in meters the gripper rises when it leaves a grasp or a placed object.
    """

    def grasp_pose_sequence(
        self,
        reference_T_grasp: Pose,
        end_effector: EndEffector,
        grasp: Optional[GraspCandidate] = None,
    ) -> GraspPoseSequence:
        """
        :param reference_T_grasp: The grasp frame to reach; for a release, where the
            object is to be put.
        :param end_effector: The end effector that is to reach it.
        :param grasp: The grasp whose object the pre-grasp pose has to stay outside of;
            ``None`` keeps only :attr:`approach_clearance`.
        :return: The tool frame goals around the grasp.
        """
        tool_goal = end_effector.tool_frame_goal(reference_T_grasp)
        grasp_T_pre_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self._approach_distance(grasp)
        )
        pre_grasp_pose = end_effector.tool_frame_goal(
            (reference_T_grasp.homogeneous_matrix @ grasp_T_pre_grasp).pose
        )
        return GraspPoseSequence(
            pre_grasp=pre_grasp_pose,
            grasp=tool_goal,
            retreat=self._retreat_pose(reference_T_grasp, tool_goal),
        )

    def _approach_distance(self, grasp: Optional[GraspCandidate]) -> float:
        """
        :param grasp: The grasp on the object, or ``None`` when there is no object.
        :return: The distance in meters from the grasp back to the pre-grasp pose: the
            distance to the object's bounding box along the approach, plus
            :attr:`approach_clearance`.
        """
        if grasp is None or not grasp.graspable.root.has_collision():
            return self.approach_clearance
        return self._distance_to_boundary(grasp) + self.approach_clearance

    @staticmethod
    def _distance_to_boundary(grasp: GraspCandidate) -> float:
        """
        The distance the gripper has to retrace before it leaves the body's bounding
        box.

        :param grasp: The grasp on the object.
        :return: The distance in meters, zero when the grasp already lies outside the
            box.
        """
        body = grasp.graspable.root
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()

        grasp_position = grasp.grasp_pose.to_np()[:3, 3]
        # The grasp frame's x-axis is where the gripper comes from, so it retraces -x.
        retrace_direction = -grasp.grasp_pose.to_np()[:3, 0]
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


# %% combined behaviour parameters


@dataclass(eq=False)
class ArmGoalParameters(ArmParameter, GoalThresholdParameters):
    """
    Bundle of the parameters for driving an arm's tool center point to a goal: the arm and
    how close to the goal it has to come.
    """


@dataclass(eq=False)
class GraspParameters(
    GraspCandidateParameter, ArmGoalParameters, GraspApproachParameters
):
    """
    Bundle of the parameters for taking hold of an object: the grasp, which names the
    object, the arm that takes hold by it, how close its tool center point has to come
    to the grasp, and the distances at which the gripper approaches and leaves it.
    """


@dataclass(eq=False)
class HandleOperationParameters(HandleParameter, ArmParameter):
    """
    Bundle of the parameters for articulating a handle with an arm: the handle and the arm.
    """


@dataclass(eq=False)
class EndEffectorPoseParameters(
    EndEffectorParameter,
    TargetPoseParameter,
    GripperCollisionParameter,
    GoalThresholdParameters,
):
    """
    Bundle of the parameters for driving an end effector to a target pose: the end effector,
    the target pose, whether gripper collision is allowed, and how close to the pose it has
    to come.
    """


@dataclass(eq=False)
class CameraTargetParameters(CameraParameter, LookTargetParameter):
    """
    Bundle of the parameters for pointing a camera at a target: the camera and the pose it is
    pointed at.
    """


@dataclass(eq=False)
class GripperActuationParameters(GripperStateParameter, EndEffectorParameter):
    """
    Bundle of the parameters for setting a gripper to an open or closed state: the gripper
    state and the end effector whose gripper is set.
    """


@dataclass(eq=False)
class GripperStallToleranceParameters(GripperActuationParameters):
    """
    Bundle of the parameters for setting a gripper that may stall short of its target: the
    gripper state, the end effector, and how a stall is tolerated.
    """

    finger_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s), enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    stall_minimum_time: Optional[float] = field(default=None, kw_only=True)
    """
    Minimum stall dwell time (in seconds, see
    :attr:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached.minimum_time`)
    to command. Only meaningful when :attr:`tolerate_stall` is True. ``None`` keeps the
    default.
    """

    tolerate_stall: bool = field(default=False, kw_only=True)
    """
    Whether this motion is also considered done once the fingers' velocities settle
    near zero, even without reaching their nominal target position -- checked via a
    separate :class:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached`
    monitor alongside the goal, not by the goal's own observation, since stalling does
    not mean the goal itself was reached.
    """


@dataclass(eq=False)
class CartesianVelocityLimitParameters(MovementTypeParameter):
    """
    Bundle of the parameters for a speed-capped Cartesian movement: the type of movement
    and the speeds the tool center point may not exceed.
    """

    max_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) of the tool center point, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the linear speed unconstrained (other than the robot's own hardware
    limits).
    """

    max_angular_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum angular speed (in rad/s) of the tool center point, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`.
    Only meaningful for :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPose`
    (i.e. when not :attr:`~coraplex.datastructures.enums.MovementType.TRANSLATION`).
    ``None`` leaves the angular speed unconstrained.
    """


# %% per-action tuning


@dataclass(eq=False)
class ReachTuningParameters(GraspDetectionThresholdParameter):
    """
    Tunable approach speeds and grasp sensitivity for reaching towards a target.
    """

    pre_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the initial pre-pose approach, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    final_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the final approach onto the target pose, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass(eq=False)
class PickUpTuningParameters(ReachTuningParameters):
    """
    Tunable grasp speeds and target-object friction for picking an object up.

    Extends :class:`ReachTuningParameters` because a pick-up's reach forwards both
    approach speeds verbatim to the reach it builds, so both fields are literally the
    same value under the same name in both places rather than two
    similarly-named-but-distinct fields.
    """

    grasp_closing_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s) used while closing onto the object, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    lift_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for lifting the object clear of the table after
    grasping, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    grasp_stall_minimum_time: Optional[float] = field(default=None, kw_only=True)
    """
    Minimum stall dwell time (in seconds, see
    :attr:`~coraplex.robot_plans.motions.gripper.MoveGripperMotion.stall_minimum_time`)
    for the CLOSE motion. ``None`` keeps the default.
    """

    object_friction: Optional[float] = field(default=None, kw_only=True)
    """
    Sliding friction coefficient to apply to the target object's geom before this pick,
    overriding the world's default. Not consumed by this action itself -- applying it is
    the caller's responsibility (see
    :meth:`~physics_simulators.mujoco_simulator.MujocoSimulator.set_geom_friction`);
    recorded here for persistence. ``None`` leaves the friction untouched.
    """


@dataclass(eq=False)
class PlaceTuningParameters(GraspDetectionThresholdParameter):
    """
    Tunable transport, placing and release speeds, and grasp sensitivity, for putting an
    object down.
    """

    placing_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the final descent onto the target location,
    enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    transport_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for carrying the held object above the target
    location, before the final descent, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    release_opening_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s) used while opening the gripper to release
    the object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    retract_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for retracting the end effector away from the placed
    object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """
