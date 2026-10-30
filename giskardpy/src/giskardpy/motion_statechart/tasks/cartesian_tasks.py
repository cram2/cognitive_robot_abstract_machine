from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from functools import cached_property

import numpy as np
from typing_extensions import ClassVar

import krrood.symbolic_math.symbolic_math as sm
from cramph.composites import Parallel
from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from giskardpy.motion_statechart.binding_policy import (
    GoalBindingPolicy,
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.error_signals import (
    joint_position_and_velocity_variables,
    time_derivative_from_joint_motion,
)
from giskardpy.motion_statechart.exceptions import GoalPointsReferenceFrameMismatchError
from giskardpy.motion_statechart.graph_node import (
    MotionNodeArtifacts,
    MotionStatechartNode,
    DebugExpression,
)
from giskardpy.motion_statechart.graph_node import Task, ConvergingTask
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import (
    VariableParameters,
    CompiledFunction,
    FloatVariable,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, SpatialType
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianTask(ConvergingTask, ABC):
    """
    Base class for all cartesian tasks.
    Offers goal binding policy functionality to subclasses.
    .. note:: subclasses describe their goal by implementing `build_artifacts`.
    """

    root_link: KinematicStructureEntity
    """Base link of the kinematic chain."""

    tip_link: KinematicStructureEntity
    """End link that should reach the goal position."""

    binding_policy: GoalBindingPolicy = GoalBindingPolicy.Bind_on_start
    """Describes when the goal is computed. See GoalBindingPolicy for more information."""

    root_T_goal_reference_frame: HomogeneousTransformationMatrix = field(init=False)
    """Transformation matrix from root to goal_reference_frame link."""

    _forward_kinematics_binding: ForwardKinematicsBinding = field(init=False)
    """Binding for the goal pose."""

    GOAL_COLOR: ClassVar[Color] = Color.from_rgba(
        red=0.0, green=1.0, blue=0.0, opacity=1.0
    )
    """The color of the goal debug expression marker (green)."""

    CURRENT_COLOR: ClassVar[Color] = Color.from_rgba(
        red=1.0, green=0.0, blue=0.0, opacity=1.0
    )
    """The color of the current debug expression marker (red)."""

    def set_up(self) -> None:
        """
        Bind the goal reference frame, which the error is described against.
        """
        super().set_up()
        self._forward_kinematics_binding = ForwardKinematicsBinding(
            name=PrefixedName("root_T_goal_ref", str(self.name)),
            root=self.root_link,
            tip=self.goal_reference_frame,
            float_variable_data=self.context.float_variable_data,
        )
        self._forward_kinematics_binding.bind(self.context.world)
        self.root_T_goal_reference_frame = self._forward_kinematics_binding.root_T_tip

    def on_start(self) -> None:
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._forward_kinematics_binding.bind(self.context.world)

    @property
    @abstractmethod
    def goal_reference_frame(self) -> KinematicStructureEntity:
        """
        :return: Reference frame for the goal.
        """

    def add_goal_and_current_debug_expressions(
        self,
        artifacts: MotionNodeArtifacts,
        goal: SpatialType,
        current: SpatialType,
    ) -> None:
        """
        Register a goal and a current spatial expression for visualization.

        The expressions are named ``<task name>/goal`` and ``<task name>/current``
        and colored green and red respectively, so they can be told apart in RViz.

        :param artifacts: The node artifacts the debug expressions are appended to.
        :param goal: The spatial expression describing the desired state.
        :param current: The spatial expression describing the current state.
        """
        artifacts.debug_expressions.append(
            DebugExpression(f"{self.name}/goal", goal, color=self.GOAL_COLOR)
        )
        artifacts.debug_expressions.append(
            DebugExpression(f"{self.name}/current", current, color=self.CURRENT_COLOR)
        )


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPosition(CartesianTask):
    """
    Move a tip link to a goal position in 3D space.

    This task controls only the position (x, y, z) of the tip link, not its orientation.

    .. warning:: This task does not constrain orientation.
    """

    default_reference_velocity: ClassVar[float] = 0.2

    goal_point: Point3
    """Target 3D point to reach."""
    threshold: float = 0.01
    """Distance threshold for goal achievement in meters."""

    reference_velocity: float = field(
        default_factory=lambda: CartesianPosition.default_reference_velocity,
    )
    """Reference velocity for normalization in m/s."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints for reaching the goal position.

        :return: The artifacts of this task, whose error is the distance between the tip and the goal point.
        """
        artifacts = MotionNodeArtifacts()
        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point

        # Get current tip position in root frame
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position

        # Add constraints to move tip towards goal
        artifacts.geometry.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=root_P_current,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_P_goal, current=root_P_current
        )

        artifacts.error = root_P_goal.euclidean_distance(root_P_current)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceAlongAxis(ConvergingTask):
    """
    Move two points, each fixed on a link of its own, until they stand a given distance
    apart along an axis, leaving them free in every other direction. Both links may
    move.
    """

    root_link: KinematicStructureEntity
    """
    The frame :attr:`axis` is fixed in.
    """

    reference_point: Point3
    """
    The point the distance is measured from, in the frame of the link it is fixed on.
    """

    tip_point: Point3
    """
    The point the distance is measured to, in the frame of the link it is fixed on.
    """

    axis: Vector3
    """
    The unit direction the distance is measured along, in :attr:`root_link`.
    """

    distance: float
    """
    How far, in metres, :attr:`tip_point` is to stand from :attr:`reference_point`
    along :attr:`axis`; negative where it is to stand against the axis.
    """

    reference_velocity: float = 0.2
    """
    Reference velocity for normalization in m/s.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The artifacts of this task, whose error is how far the distance along
            :attr:`axis` lies from :attr:`distance`.
        """
        artifacts = MotionNodeArtifacts()
        root_P_reference, root_P_tip = (
            self.context.world.compose_forward_kinematics_expression(
                self.root_link, point.reference_frame
            )
            @ point
            for point in (self.reference_point, self.tip_point)
        )
        current_distance = (root_P_tip - root_P_reference) @ self.axis
        artifacts.constraints.add_equality_constraint(
            reference_velocity=self.reference_velocity,
            equality_bound=self.distance - current_distance,
            quadratic_weight=self.weight,
            task_expression=current_distance,
        )
        artifacts.error = sm.abs(self.distance - current_distance)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPositionTrajectory(CartesianTask):
    """
    Move a tip link to a goal position along a trajectory.
    .. warning:: the trajectory is assumed to be dense and smooth.
    """

    goal_points: list[Point3]
    """Target 3D point to reach."""
    _goal_points_np: np.ndarray = field(init=False, repr=False)
    """Goal points in numpy format."""
    maximum_skip_ahead: int | None = None
    """
    This limits how many points can be skipped in very dense trajectories.
    Setting this number is required if your trajectory contains loops.
    """

    threshold: float = 0.01
    """
    Distance threshold for goal achievement in meters.
    """

    look_ahead_distance: float = 0.01
    """
    Distance from the current position to tracking target.
    Increasing this value can increase the tracking velocity, but might reduce tracking accuracy.
    """

    reference_velocity: float | None = field(
        default_factory=lambda: CartesianPosition.default_reference_velocity,
    )
    """Reference velocity for normalization in m/s."""

    goal_reference_frame_P_current_target_point: Point3 = field(init=False, repr=False)
    """Symbolic expression representing the current target point in the goal reference frame."""

    remaining_distance: FloatVariable = field(init=False, repr=False)
    """Distance left to travel along the trajectory, rewritten every control cycle."""

    current_index: int = 0
    """Current index in the goal points array."""

    _compiled_goal_reference_frame_P_tip: CompiledFunction = field(
        init=False, repr=False
    )
    """Compiled function representing the goal reference frame position in the tip frame."""

    @cached_property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        reference_frame = self.goal_points[0].reference_frame
        for point in self.goal_points[1:]:
            if point.reference_frame != reference_frame:
                raise GoalPointsReferenceFrameMismatchError(
                    node=self,
                    reference_frame_a=point.reference_frame,
                    reference_frame_b=reference_frame,
                )
        return reference_frame

    def _goal_points_to_np(self):
        self._goal_points_np = np.array(
            [point.to_np()[:-1] for point in self.goal_points]
        )

    def set_up(self) -> None:
        """
        Register the target point on the trajectory and the distance left to travel.
        """
        super().set_up()
        self._goal_points_to_np()
        self._init_goal_reference_frame_P_current_target_point(
            self.context.float_variable_data
        )
        self._init_remaining_distance(self.context.float_variable_data)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that pull the tip along the trajectory.

        :return: The artifacts of this task, whose error is the distance still to travel along the trajectory, which only
            :meth:`on_tick` can compute because it depends on how far along the
            trajectory the tip already is.
        """
        artifacts = MotionNodeArtifacts()
        root_P_goal = (
            self.root_T_goal_reference_frame
            @ self.goal_reference_frame_P_current_target_point
        )

        # Get current tip position in root frame
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position

        # Add constraints to move tip towards goal
        artifacts.geometry.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=root_P_current,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_P_goal, current=root_P_current
        )

        self.compile_current_point_on_tick()
        artifacts.error = self.remaining_distance
        return artifacts

    def _init_remaining_distance(self, float_variable_data: FloatVariableData) -> None:
        """
        Create the variable holding the distance left to travel along the trajectory and
        seed it with the full trajectory length.

        :param float_variable_data: The data the variable is registered with.
        """
        self.remaining_distance = FloatVariable(
            str(PrefixedName("remaining_distance", str(self.name)))
        )
        float_variable_data.register_expression(self.remaining_distance)
        float_variable_data.set_value(
            self.remaining_distance,
            self._distance_left_from(0, self._goal_points_np[0]),
        )

    def _distance_left_from(
        self, index: int, goal_reference_frame_P_tip_np: np.ndarray
    ) -> float:
        """
        :param index: Index of the trajectory point the tip is currently at.
        :param goal_reference_frame_P_tip_np: Current tip position in the goal reference
            frame.
        :return: Distance from the tip to the trajectory point at `index`, plus the
            length of the trajectory remaining after it.
        """
        remaining_points = self._goal_points_np[index:]
        if len(remaining_points) > 1:
            segment_lengths = np.linalg.norm(np.diff(remaining_points, axis=0), axis=1)
        else:
            segment_lengths = np.zeros(0)
        distance_to_path = np.linalg.norm(
            remaining_points[0] - goal_reference_frame_P_tip_np
        )
        return float(distance_to_path + segment_lengths.sum())

    def compile_current_point_on_tick(self) -> None:
        """
        Computing the current point relative to the goal reference frame is expensive, this method turns it into
        a compiled expression.

        .. warning:: Every build binds a new compiled function to the float variable data, and
            :meth:`~krrood.symbolic_math.float_variable_data.FloatVariableData.bind_argument`
            never releases the one bound by the previous build. A statechart that builds its nodes again
            after a change of the world structure therefore keeps one stale compiled function per rebuild.

        """
        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        goal_reference_frame_T_tip = (
            self.root_T_goal_reference_frame.inverse() @ root_T_tip
        )
        goal_reference_frame_P_tip = goal_reference_frame_T_tip.position[:-1]
        self._compiled_goal_reference_frame_P_tip = goal_reference_frame_P_tip.compile(
            parameters=VariableParameters.from_lists(
                self.context.world.state.position_float_variables,
                self.context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_goal_reference_frame_P_tip.bind_args_to_memory_view(
            0, self.context.world.state.positions
        )
        self.context.float_variable_data.bind_argument(
            self._compiled_goal_reference_frame_P_tip, 1
        )

    def _update_trajectory_index(self, goal_reference_frame_P_tip_np: np.ndarray):
        """
        Search for the closest point in the trajectory to the current position, without going backwards.
        :param goal_reference_frame_P_tip_np: the current position in the goal reference frame as a 3d numpy array.
        """
        if self.maximum_skip_ahead is None:
            remaining_points = self._goal_points_np[self.current_index :]
        else:
            remaining_points = self._goal_points_np[
                self.current_index : self.current_index + self.maximum_skip_ahead
            ]
        distances = np.linalg.norm(
            remaining_points - goal_reference_frame_P_tip_np, axis=1
        )
        local_closest_index = np.argmin(distances)
        self.current_index += local_closest_index

    def _compute_target_point(
        self, goal_reference_frame_P_tip_np: np.ndarray
    ) -> np.ndarray:
        """
        Computes a target point at a fixed distance away from the current position projected onto the trajectory.
        This ensures a constant velocity and pulls the tip onto the trajectory.
        :param goal_reference_frame_P_tip_np: the current position in the goal reference frame as a 3d numpy array.
        """
        if self.current_index >= len(self._goal_points_np) - 1:
            # If we've reached the end of the trajectory, return the last point
            return self._goal_points_np[-1]

        p_current = self._goal_points_np[self.current_index]
        p_next = self._goal_points_np[self.current_index + 1]

        # Tangent vector
        tangent = p_next - p_current
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm <= 1e-6:
            # /0 safeguard
            return p_current
        unit_tangent = tangent / tangent_norm

        # Project current position onto the segment (p_current, p_next)
        # This is the "closest point" on the line segment
        v = goal_reference_frame_P_tip_np - p_current
        projection_dist = np.dot(v, unit_tangent)
        projection_dist = np.clip(projection_dist, 0, tangent_norm)
        p_projected = p_current + projection_dist * unit_tangent

        # Aim for a point 'threshold' distance away from the PROJECTED point
        # This ensures that the target point is always 'threshold' away along the path,
        # which creates a vector that pulls the robot back to the path AND forward.
        return p_projected + unit_tangent * self.look_ahead_distance

    def on_tick(self) -> ObservationStateValues | None:
        """
        Update the target point on the trajectory and the distance left to travel.

        The observation follows from that distance, so it is one control cycle behind.
        """
        goal_reference_frame_P_tip_np = (
            self._compiled_goal_reference_frame_P_tip.evaluate()
        )
        self._update_trajectory_index(goal_reference_frame_P_tip_np)
        target_point = self._compute_target_point(goal_reference_frame_P_tip_np)
        self.context.float_variable_data.set_value(
            self.goal_reference_frame_P_current_target_point, target_point
        )
        self.context.float_variable_data.set_value(
            self.remaining_distance,
            self._distance_left_from(self.current_index, goal_reference_frame_P_tip_np),
        )
        return None

    def _init_goal_reference_frame_P_current_target_point(
        self, float_variable_data: FloatVariableData
    ):
        """
        Initialize the symbolic expression representing the current target point in the goal reference frame.
        :param float_variable_data: The FloatVariableData instance to register the expression with.
        """
        self.goal_reference_frame_P_current_target_point = Point3.create_with_variables(
            "goal_reference_frame_P_current_target_point"
        )
        self.goal_reference_frame_P_current_target_point.reference_frame = (
            self.goal_reference_frame
        )
        float_variable_data.register_expression(
            self.goal_reference_frame_P_current_target_point
        )
        float_variable_data.set_value(
            self.goal_reference_frame_P_current_target_point,
            self.goal_points[self.current_index].to_np()[:-1],
        )


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPositionStraight(CartesianTask):
    """
    Move a tip link to a goal position along a straight line.

    Unlike CartesianPosition, this task constrains the tip link to move in a straight
    line towards the goal, useful for tasks requiring linear trajectories.
    """

    goal_point: Point3
    """Target 3D point to reach."""

    reference_velocity: float = CartesianPosition.default_reference_velocity
    """Reference velocity for movement in m/s."""

    threshold: float = 0.01
    """Distance threshold for goal achievement in meters."""

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value, kw_only=True
    )
    """
    Cost of falling behind on the line; leaving the line costs
    :attr:`LATERAL_WEIGHT_FACTOR` times as much.
    """

    _line_start_binding: ForwardKinematicsBinding = field(init=False)
    """Binding for the tip pose the line starts at."""

    LATERAL_WEIGHT_FACTOR: ClassVar[float] = 2
    """How much more expensive leaving the line is than falling behind on it."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def set_up(self) -> None:
        """
        Bind the pose the line starts at, which the constraints are described against.
        """
        super().set_up()
        self._line_start_binding = ForwardKinematicsBinding(
            name=PrefixedName("root_T_line_start", str(self.name)),
            root=self.root_link,
            tip=self.tip_link,
            float_variable_data=self.context.float_variable_data,
        )
        self._line_start_binding.bind(self.context.world)

    def on_start(self) -> None:
        """
        Start the line at the pose the tip has now.

        A line starts where the tip is when the motion begins, no matter how much earlier
        the statechart was built, and independently of :attr:`binding_policy`, which only
        decides when the goal is computed.
        """
        super().on_start()
        self._line_start_binding.bind(self.context.world)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints for reaching the goal along a straight line.

        The tip is driven along the line from the pose the motion starts at to the goal,
        and back onto that line whenever it strays.

        :return: The artifacts of this task, whose error is the distance between the tip and the goal point.
        """
        artifacts = MotionNodeArtifacts()
        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point
        root_P_line_start = self._line_start_binding.root_T_tip.position
        root_P_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position

        root_V_line = root_P_goal - root_P_line_start
        # scale normalizes in place, so the length has to be read before it
        line_length = root_V_line.norm()
        root_V_line.scale(1)
        root_R_line = RotationMatrix.from_x_axis(root_V_line)
        line_V_travelled = root_R_line.inverse() @ (root_P_tip - root_P_line_start)

        lateral_weight = self.weight * self.LATERAL_WEIGHT_FACTOR
        artifacts.geometry.add_position_constraint(
            expression_current=line_V_travelled[0],
            expression_goal=line_length,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
            name="line/progress",
        )
        artifacts.geometry.add_position_constraint(
            expression_current=line_V_travelled[1],
            expression_goal=0,
            reference_velocity=self.reference_velocity,
            quadratic_weight=lateral_weight,
            name="line/y",
        )
        artifacts.geometry.add_position_constraint(
            expression_current=line_V_travelled[2],
            expression_goal=0,
            reference_velocity=self.reference_velocity,
            quadratic_weight=lateral_weight,
            name="line/z",
        )

        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_P_goal, current=root_P_tip
        )

        artifacts.error = root_P_goal.euclidean_distance(root_P_tip)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianOrientation(CartesianTask):
    """
    Rotate a tip link to match a goal orientation.

    This task controls only the orientation (roll, pitch, yaw) of the tip link,
    not its position.

    .. warning:: This task does not constrain position.
    """

    default_reference_velocity: ClassVar[float] = 0.2

    goal_orientation: RotationMatrix
    """Target rotation matrix to match."""
    threshold: float = 0.01
    """Rotation error threshold for goal achievement in radians."""

    reference_velocity: float = field(
        default_factory=lambda: CartesianOrientation.default_reference_velocity,
    )
    """Reference angular velocity for normalization in rad/s."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_orientation.reference_frame

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints for reaching the goal orientation.

        :return: The artifacts of this task, whose error is the angle between the tip orientation and the goal orientation.
        """
        artifacts = MotionNodeArtifacts()
        root_R_goal = self.root_T_goal_reference_frame @ self.goal_orientation

        # Get current tip orientation in root frame
        root_T_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_R_current = root_T_current.rotation_matrix

        # Add constraints to rotate tip towards goal
        artifacts.geometry.add_rotation_goal_constraints(
            frame_R_current=root_R_current,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_R_goal, current=root_R_current
        )

        artifacts.error = sm.abs(root_R_current.rotational_distance(root_R_goal))
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPose(Parallel):
    """
    This goal will use the kinematic chain between root and tip link to move tip_link into the 6D goal_pose.

    Position and orientation are separate tasks, because an error in meters and an error
    in radians cannot be compared against one threshold. This goal is the parallel
    holding both, together with any node it is handed, such as a speed cap or a
    collision allowance accompanying the motion.
    """

    root_link: KinematicStructureEntity | None = None
    """Base link of the kinematic chain. Defaults to the root of the world."""

    tip_link: KinematicStructureEntity
    """End link that should reach the goal pose."""

    goal_pose: Pose
    """The goal pose."""

    reference_linear_velocity: float = CartesianPosition.default_reference_velocity
    """Unit: m/s. This is used for normalization, for real limits use CartesianVelocityLimit."""

    reference_angular_velocity: float = CartesianOrientation.default_reference_velocity
    """Unit: rad/s. This is used for normalization, for real limits use CartesianVelocityLimit."""

    translation_threshold: float = 0.01
    """If the position error falls below this threshold (in meters), that half of the goal is achieved."""

    orientation_threshold: float = 0.01
    """
    If the orientation error falls below this threshold (in rad), that half of the goal
    is achieved.

    ..note:: A physically tracked arm settles with a residual orientation error, so a
        rotation tolerance as tight as a typical translation tolerance in meters may
        never be reached -- set this independently of :attr:`translation_threshold`.
    """

    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """Task priority relative to other tasks."""

    binding_policy: GoalBindingPolicy = GoalBindingPolicy.Bind_on_start
    """Describes when the goal is computed. See GoalBindingPolicy for more information."""

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The position and the orientation task, followed by the nodes this goal was handed.
    """

    def expand(self) -> None:
        """
        Create the position and the orientation task, before the nodes this goal was
        handed.
        """
        if self.root_link is None:
            self.root_link = self.context.world.root
        for position, task in enumerate(self.create_tasks()):
            self._insert_at(position, task)

    def create_tasks(self) -> list[MotionStatechartNode]:
        """
        :return: The position and the orientation task.
        """
        return [
            CartesianPosition(
                name=f"{self.name}/position",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.goal_pose.position,
                reference_velocity=self.reference_linear_velocity,
                threshold=self.translation_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
            CartesianOrientation(
                name=f"{self.name}/orientation",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_orientation=self.goal_pose.rotation_matrix,
                reference_velocity=self.reference_angular_velocity,
                threshold=self.orientation_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
        ]


@dataclass(eq=False, repr=False, kw_only=True)
class HoldOrientation(Task):
    """
    Keeps a link in the orientation it had when this task started, leaving it free to
    move.

    An invariant rather than a goal, so it does not report progress. It observes
    whether the link is still in that orientation.
    """

    root_link: KinematicStructureEntity
    """Base link of the kinematic chain."""

    tip_link: KinematicStructureEntity
    """The link kept as it is."""

    reference_angular_velocity: float = CartesianOrientation.default_reference_velocity
    """Unit: rad/s. Normalizes the rotational constraints against the others."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Task priority relative to other tasks.

    Above collision avoidance by default, because a link is held against whatever would
    otherwise move it, and buying clearance by moving it is the motion this prevents.
    """

    orientation_threshold: float = 0.01
    """Unit: rad. The link counts as held while it is rotated at most this far from its pose."""

    _pose_to_keep: ForwardKinematicsBinding = field(init=False, repr=False)
    """The pose of the link bound when this task starts."""

    def set_up(self) -> None:
        """
        Bind the pose the link is held at.
        """
        super().set_up()
        self._pose_to_keep = ForwardKinematicsBinding(
            name=PrefixedName("pose_to_keep", str(self.name)),
            root=self.root_link,
            tip=self.tip_link,
            float_variable_data=self.context.float_variable_data,
        )
        self._pose_to_keep.bind(self.context.world)

    def on_start(self) -> None:
        self._pose_to_keep.bind(self.context.world)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Constrain the chain to the orientation bound when this task started.

        :return: The artifacts of this task, which observes whether the link is within
            :attr:`orientation_threshold` of that orientation.
        """
        artifacts = MotionNodeArtifacts()
        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).rotation_matrix
        root_R_tip_when_started = self._pose_to_keep.root_T_tip.rotation_matrix
        artifacts.geometry.add_rotation_goal_constraints(
            frame_R_goal=root_R_tip_when_started,
            frame_R_current=root_R_tip,
            reference_velocity=self.reference_angular_velocity,
            quadratic_weight=self.weight,
            name="orientation",
        )
        artifacts.observation = (
            sm.abs(root_R_tip.rotational_distance(root_R_tip_when_started))
            <= self.orientation_threshold
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class HoldPose(HoldOrientation):
    """
    Keeps a link at the pose it had when this task started.

    .. note:: Unlike :class:`CartesianPose`, this constrains the chain between
        :attr:`root_link` and :attr:`tip_link` without naming a pose, because the pose
        to keep is the one the link is already at.
    """

    reference_linear_velocity: float = CartesianPosition.default_reference_velocity
    """Unit: m/s. Normalizes the translational constraints against the others."""

    translation_threshold: float = 0.01
    """Unit: m. The link counts as held while it is at most this far from its pose."""

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Differs from :meth:`HoldOrientation.build_artifacts` in also constraining the
        position, and observing whether the link is within
        :attr:`translation_threshold` of it.
        """
        artifacts = super().build_artifacts()
        root_P_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position
        root_P_tip_when_started = self._pose_to_keep.root_T_tip.position
        artifacts.geometry.add_point_goal_constraints(
            frame_P_goal=root_P_tip_when_started,
            frame_P_current=root_P_tip,
            reference_velocity=self.reference_linear_velocity,
            quadratic_weight=self.weight,
            name="position",
        )
        artifacts.observation = sm.logic_and(
            artifacts.observation,
            root_P_tip_when_started.euclidean_distance(root_P_tip)
            <= self.translation_threshold,
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPositionVelocityLimit(Task):
    """
    Limit the Cartesian (translational) velocity of a tip link relative to a root link.

    This goal enforces a strict cap on the linear speed of the frame defined by
    the kinematic transform from `root_link` to `tip_link`. Enforcement is performed
    by adding constraints to the optimizer and by providing an observation expression
    that evaluates whether the current translational speed is within the limit.

    .. warning::
       Strict Cartesian velocity limits require as many constraints as the prediction
       horizon size, making the optimization problem more complex. This can impact
       solve time especially at high control frequencies. If computation time is critical,
       consider using larger limits or reducing the prediction horizon.
    """

    root_link: KinematicStructureEntity
    """
    Root link of the kinematic chain. 
    Defines the reference frame from which the tip's motion is measured.
    """
    tip_link: KinematicStructureEntity
    """
    Tip link of the kinematic chain. 
    The translational velocity of this link (expressed in the root link frame) is constrained.
    """
    max_linear_velocity: float = 0.1
    """
    Maximum allowed linear speed of the tip in meters per second (m/s).
    Default: 0.1 m/s. The enforcement ensures the Euclidean norm of the
    tip-frame translational velocity does not exceed this value.
    """
    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Optimization weight determining how strongly the linear velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        root_P_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position
        artifacts.geometry.add_translational_velocity_limit(
            frame_P_current=root_P_tip,
            max_velocity=self.max_linear_velocity,
            quadratic_weight=self.weight,
        )

        position_variables, velocity_variables = joint_position_and_velocity_variables(
            root_P_tip
        )
        root_P_tip_dot = root_P_tip.total_derivative(
            position_variables, velocity_variables
        )

        artifacts.observation = root_P_tip_dot.norm() <= self.max_linear_velocity

        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianRotationVelocityLimit(Task):
    """
    Represents a Cartesian rotational velocity limit task within a kinematic chain.

    This task constrains the angular velocity of a specified tip link relative
    to a root link to not exceed a maximum allowed angular velocity. It uses
    optimization weights to prioritize its enforcement in solving problems
    involving kinematic motion. The task calculates and enforces constraints
    based on the rotation matrix between the root and tip links, ensuring
    compliance with the defined angular velocity limits.

    .. warning::
       Strict Cartesian velocity limits require as many constraints as the prediction
       horizon size, making the optimization problem more complex. This can impact
       solve time especially at high control frequencies. If computation time is critical,
       consider using larger limits or reducing the prediction horizon.
    """

    root_link: KinematicStructureEntity
    """Root link of the kinematic chain. Defines the reference frame from which the tip's motion is measured."""
    tip_link: KinematicStructureEntity
    """Tip link of the kinematic chain. The translational velocity of this link (expressed in the root link frame) is constrained."""
    max_angular_velocity: float = 0.4
    """Maximum allowed angular speed. Interpreted in radians per second (rad/s).
    The enforcement ensures the magnitude of the instantaneous
    rotation rate does not exceed this threshold."""
    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """Optimization weight determining how strongly the rotational velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur."""

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()

        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).rotation_matrix

        artifacts.geometry.add_rotational_velocity_limit(
            frame_R_current=root_R_tip,
            max_velocity=self.max_angular_velocity,
            quadratic_weight=self.weight,
        )

        angle = root_R_tip.axis_angle.angle
        angle_dot = time_derivative_from_joint_motion(angle)

        artifacts.observation = sm.abs(angle_dot) <= self.max_angular_velocity

        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianVelocityLimit(Parallel):
    """
    Combines both linear and angular velocity limits for a kinematic chain.

    This task enforces strict caps on both the linear and angular velocities of
    a tip link relative to a root link by combining CartesianPositionVelocityLimit
    and CartesianRotationVelocityLimit tasks in parallel.

    .. warning::
       Strict Cartesian velocity limits require as many constraints as the prediction
       horizon size, making the optimization problem more complex. This can impact
       solve time especially at high control frequencies. If computation time is critical,
       consider using larger limits or reducing the prediction horizon.
    """

    root_link: KinematicStructureEntity
    """Root link of the kinematic chain. Defines the reference frame from which the tip's motion is measured."""
    tip_link: KinematicStructureEntity
    """Tip link of the kinematic chain. Both translational and rotational velocities of this link (expressed in the root link frame) are constrained."""
    max_linear_velocity: float = 0.1
    """Maximum allowed linear speed of the tip in meters per second (m/s).
    Default: 0.1 m/s. The enforcement ensures the Euclidean norm of the
    tip-frame translational velocity does not exceed this value."""
    max_angular_velocity: float = 0.4
    """Maximum allowed angular speed. Interpreted in radians per second (rad/s).
    Default: 0.5 rad/s. The enforcement ensures the magnitude of the instantaneous
    rotation rate does not exceed this threshold."""
    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """Optimization weight determining how strongly both velocity
    limits are enforced. Higher weights give these constraints soft priority
    over lower weighted constraints when conflicts occur."""

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The linear and the angular velocity limit.
    """

    def expand(self) -> None:
        """
        Create the linear and the angular velocity limit, before the nodes this goal
        was handed.
        """
        limits = [
            CartesianPositionVelocityLimit(
                root_link=self.root_link,
                tip_link=self.tip_link,
                max_linear_velocity=self.max_linear_velocity,
                weight=self.weight,
            ),
            CartesianRotationVelocityLimit(
                root_link=self.root_link,
                tip_link=self.tip_link,
                max_angular_velocity=self.max_angular_velocity,
                weight=self.weight,
            ),
        ]
        for position, limit in enumerate(limits):
            self._insert_at(position, limit)
