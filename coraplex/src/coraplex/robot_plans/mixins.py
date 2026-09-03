from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Optional

from coraplex.config.action_conf import ActionConfig
from coraplex.utils import translate_pose_along_local_axis
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class HasMaxJointVelocity:
    """
    Adds an optional joint velocity cap to an action or motion.
    """

    max_joint_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum joint velocity (in rad/s or m/s, per joint), enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """


@dataclass
class HasApproachVelocity:
    """
    Adds an optional pre-approach speed to an action that reaches towards a target
    before its main motion.

    Shared by :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction` and
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`, since a pick-up's
    reach is itself a :class:`ReachAction` and forwards this same value to it.
    """

    pre_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the initial pre-pose approach, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass
class HasGraspDetectionThreshold:
    """
    Adds a grasp-detection sensitivity threshold to an action that checks whether an
    object is held between the gripper's fingers.

    Shared by :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction`,
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction` and
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`.
    """

    grasp_detection_threshold: float = field(default=0.9, kw_only=True)
    """
    Minimum fraction of sampled rays between the gripper's fingers that must hit the
    target object for it to count as grasped/held (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """


@dataclass
class ReachTuningParameters(HasApproachVelocity):
    """
    Tunable approach speeds for :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction`.
    """

    final_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the final approach onto the target pose, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass
class PickUpTuningParameters(ReachTuningParameters):
    """
    Tunable grasp speeds and target-object friction for
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`.

    Extends :class:`ReachTuningParameters` rather than just :class:`HasApproachVelocity`:
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction` forwards both
    ``pre_approach_linear_velocity`` and ``final_approach_linear_velocity`` verbatim to
    the internal :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction` it
    builds, so both fields are literally the same value under the same name in both
    places rather than two similarly-named-but-distinct fields.
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


@dataclass
class PlaceTuningParameters:
    """
    Tunable transport/placing/release speeds for
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`.
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


@dataclass
class GripperStallToleranceParameters:
    """
    Adds an optional finger speed and stall-tolerance to a gripper open/close motion.
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


@dataclass
class CartesianVelocityLimitParameters:
    """
    Adds an optional linear and angular speed cap to a Cartesian tool-center-point
    motion.
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


@dataclass
class HasTcpGoalThresholds:
    """
    Adds optional tool-center-point goal-achievement thresholds to a motion, falling
    back to :attr:`~coraplex.datastructures.dataclasses.Context.motion_tolerances` when
    left unset.

    Meant to be mixed into a :class:`~coraplex.robot_plans.motions.base.BaseMotion`
    subclass, whose ``context`` the resolver methods below rely on.
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


@dataclass
class HasApproachesGraspPoses:
    """
    Turns a grasp frame into the tool center point goals that reach it and withdraw again.

    A grasp pose is a grasp frame as
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspPoses.grasp_poses`
    defines it: its x-axis points the way the gripper travels toward the object. What
    that frame means for a concrete gripper is the end effector's own business (see
    :meth:`~semantic_digital_twin.robots.robot_parts.EndEffector.tool_frame_goal`); what
    is left here is how far ahead of the grasp the approach begins and how far it
    withdraws, which grasping, placing and reaching all share.
    """

    approach_clearance: float = field(
        default=ActionConfig.approach_clearance, kw_only=True
    )
    """
    The gap in meters between the object and the gripper at the pre-grasp pose.
    """

    retreat_distance: float = field(default=ActionConfig.retreat_distance, kw_only=True)
    """
    The height in meters the gripper rises by after closing on the object.
    """

    @staticmethod
    def _grasp_in_body_frame(grasp_pose: Pose, body: Optional[Body]) -> Optional[Pose]:
        """
        The grasp in the frame of the body whose geometry the gripper has to clear.

        A grasp that
        :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspPoses.grasp_poses`
        produced is already written in that body's frame and is passed on unchanged; one
        a caller aimed at the body from somewhere else has to be rewritten first.

        :param grasp_pose: The grasp frame to reach.
        :param body: The body being grasped, or ``None`` when there is none.
        :return: The grasp in ``body``'s frame, or ``None``.
        """
        if body is None:
            return None
        if grasp_pose.reference_frame is body:
            return grasp_pose
        return body._world.transform(grasp_pose.to_homogeneous_matrix(), body).to_pose()

    def grasp_pose_sequence(
        self,
        grasp_pose: Pose,
        end_effector: EndEffector,
        body_T_grasp: Optional[Pose] = None,
        reverse: bool = False,
    ) -> List[Pose]:
        """
        The tool frame goals that reach a grasp pose and withdraw from it.

        The sequence holds three poses: one clear of the object along the approach
        direction, one at the grasp itself, and one raised above it. Reversing it turns
        a grasp into a release.

        :param grasp_pose: The grasp frame to reach.
        :param end_effector: The end effector that is to reach it.
        :param body_T_grasp: The same grasp written in the grasped body's own frame,
            which is what says how much of the body the pre-grasp pose has to clear
            (see :meth:`_grasp_in_body_frame`). It is passed rather than derived because
            a body being placed is still in the gripper, nowhere near the grasp being
            aimed at, so a release passes the grasp it is held by instead. Without it
            only :attr:`approach_clearance` separates the two poses.
        :param reverse: Whether to withdraw from the grasp rather than move onto it.
        :return: The pre-grasp pose, the grasp pose and the retreat pose.
        """
        tool_goal = end_effector.tool_frame_goal(grasp_pose)
        pre_grasp_pose = translate_pose_along_local_axis(
            tool_goal,
            end_effector.front_facing_axis.to_np()[:3].astype(float),
            -self._approach_distance(body_T_grasp),
        )
        sequence = [
            pre_grasp_pose,
            tool_goal,
            self._retreat_pose(grasp_pose, tool_goal),
        ]
        if reverse:
            sequence.reverse()
        return sequence

    def _approach_distance(self, body_T_grasp: Optional[Pose]) -> float:
        """
        How far ahead of the grasp the gripper waits before its final approach.

        The pre-grasp pose has to sit outside the object, so the distance covers
        whatever geometry lies between the grasp and the object's boundary along the
        approach direction, plus :attr:`approach_clearance`. A grasp on the object's own surface,
        such as one on the rim of a bowl, therefore needs barely more than the
        clearance, while a grasp at the object's center needs to clear half of it.

        :param body_T_grasp: The grasp in the grasped body's frame, or ``None`` when
            there is no body to clear.
        :return: The distance in meters.
        """
        body = body_T_grasp.reference_frame if body_T_grasp is not None else None
        if not isinstance(body, Body) or not body.has_collision():
            return self.approach_clearance
        return self._distance_to_boundary(body_T_grasp) + self.approach_clearance

    @staticmethod
    def _distance_to_boundary(body_T_grasp: Pose) -> float:
        """
        The distance the gripper has to retrace before it leaves the body's bounding
        box.

        :param body_T_grasp: The grasp in the grasped body's frame.
        :return: The distance in meters, zero when the grasp already lies outside the
            box.
        """
        body = body_T_grasp.reference_frame
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()

        grasp_position = body_T_grasp.to_np()[:3, 3]
        # The grasp frame's x-axis is where the gripper comes from, so it retraces -x.
        retrace_direction = -body_T_grasp.to_np()[:3, 0]
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

    def _retreat_pose(self, grasp_pose: Pose, tool_goal: Pose) -> Pose:
        """
        The tool frame goal that lifts the object straight up off its support.

        :param grasp_pose: The grasp frame that was reached.
        :param tool_goal: The tool frame goal at the grasp, whose orientation is kept.
        :return: The retreat pose, in ``grasp_pose``'s frame.
        """
        target = grasp_pose.reference_frame
        world = target._world
        world_T_grasp = world.transform(grasp_pose.to_homogeneous_matrix(), world.root)
        grasp_T_retreat = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=self.retreat_distance
        )
        return Pose(
            world.transform((world_T_grasp @ grasp_T_retreat).to_position(), target),
            tool_goal.to_quaternion(),
            reference_frame=target,
        )
