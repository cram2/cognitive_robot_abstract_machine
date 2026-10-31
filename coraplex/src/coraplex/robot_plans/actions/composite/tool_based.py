from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    PouringSide,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)
from coraplex.exceptions import (
    MissingWaypoints,
    WipingTargetMissing,
)
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import (
    Action,
    ActionOfSteps,
    ParallelAction,
    SequentialAction,
    TryAllAction,
)
from coraplex.robot_plans.mixins import MovesToolCenterPoint
from cramph.composites import Parallel, PausedUntilTrue
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.monitors.cartesian_monitors import PositionReached
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionTrajectory,
)
from coraplex.robot_plans.actions.composite.tool_paths import (
    ToolPath,
    ToolPathSegment,
    build_container_path,
    build_cutting_path,
    build_surface_path,
    planar_spiral_xy,
    planar_sweep_x,
)


@dataclass(kw_only=True, eq=False, repr=False)
class FullBodyControlledAction(Action, ABC):
    """
    An action that controls the robot's full body, so the base can support the arm
    motion.
    """

    @property
    def controlled_root(self) -> KinematicStructureEntity:
        """
        :return: The world root for a robot with a mobile base, since driving the base
            while manipulating moves the robot relative to the world; the robot's own
            root otherwise.
        """
        if isinstance(self.robot, HasMobileBase):
            return self.world.root
        return self.robot.root


@dataclass(kw_only=True, eq=False, repr=False)
class ToolMotionAction(
    FullBodyControlledAction, MovesToolCenterPoint, ActionOfSteps, ABC
):
    """
    An action that moves a tool along a sampled tool path while keeping the tool aligned
    with its target.
    """

    arm: Arm
    """
    The arm holding the tool.
    """

    tool: Tool
    """
    The tool that performs the motion.
    """

    pointer_stride: int = 1
    """
    Keep every Nth sampled waypoint for execution.
    """

    maximum_skip_ahead: int = 2
    """
    How many waypoints ahead the tool may already be heading for, so a path is followed
    as a continuous stroke rather than stopping at every point.
    """

    @abstractmethod
    def _build_tool_path(self) -> ToolPath:
        """
        :return: The tool path of this action in its local frame.
        """

    @abstractmethod
    def _path_frame(self) -> HomogeneousTransformationMatrix:
        """
        :return: The frame the tool path is expressed in.
        """

    @property
    @abstractmethod
    def _alignment_target(self) -> Body | Pose | None:
        """
        :return: The target the tool is aligned with during the motion.
        """

    @cached_property
    def _waypoints(self) -> list[Point3]:
        """
        :return: The sampled waypoints of the tool path in the world frame.
        """
        _, points, _ = self._build_tool_path().sample(frame=self._path_frame())
        stride = max(1, int(self.pointer_stride))
        waypoints = [
            Point3(x=point[0], y=point[1], z=point[2], reference_frame=self.world.root)
            for point in points
        ][::stride]
        if not waypoints:
            raise MissingWaypoints(self)
        return waypoints

    @property
    def _alignment_pairs(self) -> list[AlignmentPair]:
        """
        :return: The normal pairs that keep the tool aligned with its target during
            the motion, or an empty list if there is no alignment target.
        """
        target = self._alignment_target
        if target is None:
            return []
        return self.tool.tool_alignment(target)

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the manipulator holding the tool to touch what the
            tool works on, for the whole stroke.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The nodes moving the tool along the sampled waypoints while keeping it
            aligned with its target, see :meth:`_tool_path_steps`.
        """
        return self._tool_path_steps()

    def _tool_path_steps(self) -> list[StatechartNode]:
        """
        :return: The nodes following the sampled waypoints with the tool together,
            holding every alignment the tool asks for while it moves, to be run side by
            side.
        """
        root = self.controlled_root
        tip = self.tool.get_tool_frame()
        alignments = [
            AlignPlanes(
                tip_link=tip,
                root_link=root,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value,
            )
            for pair in self._alignment_pairs
        ]
        trajectory = CartesianPositionTrajectory(
            root_link=root,
            tip_link=tip,
            goal_points=self._waypoints,
            maximum_skip_ahead=self.maximum_skip_ahead,
            weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
            threshold=self.position_threshold,
        )
        return [trajectory, *alignments]


@dataclass(kw_only=True, eq=False, repr=False)
class MixingAction(ToolMotionAction, ParallelAction):
    """
    Mix the contents of a container with a tool.
    """

    container: Body
    """
    The container (e.g., a bowl) whose contents are mixed.
    """

    mix_duration: float = 0.0
    """
    Total mixing time in seconds for a continuous stir loop.

    If not positive, a short spiral pattern is used instead.
    """

    def _build_tool_path(self) -> ToolPath:
        if self.mix_duration > 0.0:
            return build_container_path(
                self.container,
                pattern=MixingPattern.STIR,
                mix_duration=self.mix_duration,
            )
        return build_container_path(self.container, pattern=MixingPattern.SPIRAL)

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        return self.container.global_pose.homogeneous_matrix

    @property
    def _alignment_target(self) -> Body | Pose | None:
        return self.container


@dataclass(kw_only=True, eq=False, repr=False)
class CuttingAction(ToolMotionAction, ParallelAction):
    """
    Cut a food object with a tool.
    """

    object_to_cut: Body
    """
    The object to cut.
    """

    technique: CuttingTechnique = CuttingTechnique.SAW
    """
    The cutting technique to use.
    """

    slice_thickness: float | None = None
    """
    Target slice thickness in meters, controlling the spacing between cut anchors.

    Derived from ``number_of_cuts_on_local_x_axis`` and the object size if None.
    """

    number_of_cuts_on_local_x_axis: int | None = None
    """
    Number of cut passes along the object's local X axis.

    Derived from ``slice_thickness`` and the object size if None.
    """

    slicing_priority: SlicingPriority = SlicingPriority.THICKNESS
    """
    Parameter that is kept when ``slice_thickness`` and
    ``number_of_cuts_on_local_x_axis`` do not both fit the object.
    """

    def _build_tool_path(self) -> ToolPath:
        return build_cutting_path(
            self.object_to_cut,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
            number_of_cuts_on_local_x_axis=self.number_of_cuts_on_local_x_axis,
            slicing_priority=self.slicing_priority,
        )

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        return self.object_to_cut.global_pose.homogeneous_matrix

    @property
    def _alignment_target(self) -> Body | Pose | None:
        return self.object_to_cut


@dataclass(kw_only=True, eq=False, repr=False)
class WipingAction(ToolMotionAction, TryAllAction):
    """
    Wipe a surface or a patch around a target pose with a tool.
    """

    surface: Body | None = None
    """
    The surface body to wipe.

    If None, ``target_pose`` is used instead.
    """

    target_pose: Pose | None = None
    """
    Center pose of the wiping patch.

    Only used if ``surface`` is None.
    """

    technique: WipingTechnique = WipingTechnique.WIPE
    """
    The wiping technique to use.
    """

    length: float = 0.20
    """
    Sweep length in meters for the spreading motion.
    """

    cycles: float = 1.0
    """
    Number of sweep cycles for the spreading motion.
    """

    final_waypoint_success_tolerance: float = 0.08
    """
    Accept a stalled motion as successful if the tool ended up within this distance in
    meters of the final waypoint.
    """

    stall_timeout: timedelta = timedelta(seconds=2)
    """
    How long the tool may make no progress along its path before the wipe is checked for
    having reached the final waypoint.

    .. note:: Has to be shorter than the stall timeout of the executor, which otherwise
        aborts the plan first.
    """

    def __post_init__(self):
        """
        :raises WipingTargetMissing: If neither a surface nor a target pose is given.
        """
        super().__post_init__()
        if self.surface is None and self.target_pose is None:
            raise WipingTargetMissing(self)

    def _build_tool_path(self) -> ToolPath:
        if self.surface is not None:
            return build_surface_path(self.surface, technique=self.technique)
        if self.technique is WipingTechnique.SPREAD:
            return ToolPath(
                [
                    ToolPathSegment(
                        kind=ToolPathSegmentKind.SWEEP,
                        duration=2.0,
                        local_curve=lambda tau: planar_sweep_x(
                            tau,
                            length=float(self.length),
                            cycles=max(1.0, float(self.cycles)),
                        ),
                    )
                ]
            )
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SPIRAL,
                    duration=2.0,
                    local_curve=lambda tau: planar_spiral_xy(
                        tau, r0=0.00, r1=0.12, cycles=2.5
                    ),
                )
            ]
        )

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        if self.surface is not None:
            return self.surface.global_pose.homogeneous_matrix
        if self.target_pose.reference_frame is None:
            self.target_pose.reference_frame = self.world.root
        return self.target_pose.homogeneous_matrix

    @property
    def _alignment_target(self) -> Body | Pose | None:
        if self.surface is not None:
            return self.surface
        return self.target_pose

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The goal moving the tool along the sampled waypoints, and as an
            alternative to it, the tool having reached the final waypoint once that
            motion stalled, since the last stretch of a wipe often stalls against the
            surface.
        """
        tool_path = Parallel(nodes=self._tool_path_steps())
        return [
            tool_path,
            PausedUntilTrue(
                monitor=Stalled(monitored_node=tool_path, timeout=self.stall_timeout),
                monitored_node=PositionReached(
                    name=f"{self.name}/final waypoint reached",
                    root_link=self.world.root,
                    tip_link=self.tool.root,
                    goal_point=self._waypoints[-1],
                    threshold=self.final_waypoint_success_tolerance,
                ),
            ),
        ]


@dataclass(kw_only=True, eq=False, repr=False)
class PouringAction(FullBodyControlledAction, MovesToolCenterPoint, SequentialAction):
    """
    Pour from a held source container into a target container by tilting the source next
    to the target's rim.
    """

    target_container: Body
    """
    The container that is poured into.
    """

    source_container: Tool
    """
    The held container that is poured from.
    """

    arm: Arm
    """
    The arm holding the source container.
    """

    tilt_angle: float = 1.85
    """
    Tilt angle in radians applied to the source container while pouring.
    """

    pour_side: PouringSide | None = None
    """
    Robot-relative side of the target container to pour from.

    Defaults to the side of the pouring arm, so one-arm robots can still use either
    side's pouring geometry.
    """

    pour_side_offset: float = 0.0
    """
    Extra lateral offset in meters of the pour point from the target container's center.
    """

    pour_approach_offset: float = 0.0
    """
    Extra offset in meters away from the target container along the approach direction.
    """

    pour_height: float = 0.13
    """
    TCP height in meters above the target container for the pre-pour pose.
    """

    def _effective_pour_side(self) -> PouringSide:
        """
        :return: The requested pour side, or the side of the pouring arm if none was
            requested.
        """
        if self.pour_side is not None:
            return self.pour_side
        if self.arm is self.robot.get_right_arm_if_specified():
            return PouringSide.RIGHT
        return PouringSide.LEFT

    def _mouth_height_above_tool_frame(self) -> float:
        """
        :return: Height in meters of the source container's opening above the arm's
            tool frame, measured along the tool frame's z axis.
        """
        tool_frame = self.arm.end_effector.tool_frame
        tool_frame_T_source = self.world.compute_forward_kinematics_np(
            tool_frame, self.source_container.root
        )
        bounding_box = (
            self.source_container.root.visual.as_bounding_box_collection_in_frame(
                self.source_container.root
            ).bounding_box()
        )
        mouth_in_source = np.array(
            [
                0.5 * (bounding_box.min_x + bounding_box.max_x),
                0.5 * (bounding_box.min_y + bounding_box.max_y),
                bounding_box.max_z,
                1.0,
            ]
        )
        return float((tool_frame_T_source @ mouth_in_source)[2])

    def _approach_direction(
        self, target_pose: Pose, robot_pose: Pose
    ) -> tuple[float, float]:
        """
        :return: The XY unit vector from the robot toward the target container,
            snapped to the target's nearest local axis so the pour never aims at a
            corner.
        """
        approach_x = float(target_pose.x) - float(robot_pose.x)
        approach_y = float(target_pose.y) - float(robot_pose.y)
        approach_norm = math.hypot(approach_x, approach_y)
        if approach_norm < 1e-6:
            approach_x, approach_y = 1.0, 0.0
        else:
            approach_x /= approach_norm
            approach_y /= approach_norm

        target_quaternion = [float(value) for value in target_pose.quaternion.to_np()]
        target_rotation = Rotation.from_quat(target_quaternion)
        target_x_axis = target_rotation.apply([1, 0, 0])
        target_y_axis = target_rotation.apply([0, 1, 0])
        approach_vector = np.array([approach_x, approach_y, 0.0])
        candidates = [target_x_axis, -target_x_axis, target_y_axis, -target_y_axis]
        alignments = [np.dot(approach_vector, candidate) for candidate in candidates]
        best = candidates[int(np.argmax(alignments))]

        snapped_norm = math.hypot(float(best[0]), float(best[1]))
        if snapped_norm <= 1e-6:
            return approach_x, approach_y
        return float(best[0]) / snapped_norm, float(best[1]) / snapped_norm

    def _pour_poses(self) -> tuple[Pose, Pose]:
        """
        :return: The pre-pour pose next to the target container's rim and the tilted
            pouring pose.
        """
        pour_side = self._effective_pour_side()
        target_pose = self.target_container.global_pose
        robot_pose = self.robot.root.global_pose

        approach_x, approach_y = self._approach_direction(target_pose, robot_pose)
        robot_right_x = approach_y
        robot_right_y = -approach_x
        side_sign = 1.0 if pour_side == PouringSide.RIGHT else -1.0

        side_offset = float(self.pour_side_offset) + math.sin(self.tilt_angle) * max(
            self._mouth_height_above_tool_frame(), 0.0
        )
        approach_offset = float(self.pour_approach_offset)

        pour_x = (
            float(target_pose.x)
            + side_sign * robot_right_x * side_offset
            - approach_x * approach_offset
        )
        pour_y = (
            float(target_pose.y)
            + side_sign * robot_right_y * side_offset
            - approach_y * approach_offset
        )
        pour_z = float(target_pose.z) + float(self.pour_height)

        yaw_to_target = math.atan2(
            float(target_pose.y) - pour_y, float(target_pose.x) - pour_x
        )
        base_rotation = Rotation.from_euler("z", yaw_to_target)
        if pour_side == PouringSide.LEFT:
            base_rotation = Rotation.from_euler("z", math.pi) * base_rotation

        signed_tilt_angle = (
            self.tilt_angle if pour_side == PouringSide.RIGHT else -self.tilt_angle
        )
        tilted_rotation = base_rotation * Rotation.from_euler("y", signed_tilt_angle)

        pre_pour_pose = self._pose_from_rotation(pour_x, pour_y, pour_z, base_rotation)
        pour_pose = self._pose_from_rotation(pour_x, pour_y, pour_z, tilted_rotation)
        return pre_pour_pose, pour_pose

    def _pose_from_rotation(
        self, x: float, y: float, z: float, rotation: Rotation
    ) -> Pose:
        """
        :param x: X position of the pose in the world frame.
        :param y: Y position of the pose in the world frame.
        :param z: Z position of the pose in the world frame.
        :param rotation: Orientation of the pose.
        :return: The pose in the world frame.
        """
        quat_x, quat_y, quat_z, quat_w = rotation.as_quat()
        return Pose.from_xyz_quaternion(
            pos_x=x,
            pos_y=y,
            pos_z=z,
            quat_x=quat_x,
            quat_y=quat_y,
            quat_z=quat_z,
            quat_w=quat_w,
            reference_frame=self.world.root,
        )

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the manipulator holding the source container to touch
            the rim it pours over.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The goals moving the source container to the pre-pour pose and then
            tilting it into the pouring pose.
        """
        pre_pour_pose, pour_pose = self._pour_poses()
        return [
            self.tool_center_point_goal(
                pre_pour_pose,
                self.arm,
            ),
            self.tool_center_point_goal(
                pour_pose,
                self.arm,
            ),
        ]
