from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from functools import cached_property
from typing import Optional, ClassVar

import numpy as np
from line_profiler.explicit_profiler import profile
from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.exceptions import NodeInitializationError
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import VariableParameters, CompiledFunction
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Vector3,
    Point3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.binding_policy import (
    GoalBindingPolicy,
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import (
    NodeArtifacts,
    DebugExpression,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.graph_node import Task


@dataclass(eq=False, repr=False)
class CartesianTask(Task, ABC):
    """
    Base class for all cartesian tasks.
    Offers goal binding policy functionality to subclasses.
    .. note:: make sure to call `super().build(context)` in subclasses, if you override it.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Base link of the kinematic chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """End link that should reach the goal position."""

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_on_start, kw_only=True
    )
    """Describes when the goal is computed. See GoalBindingPolicy for more information."""

    root_T_goal_reference_frame: HomogeneousTransformationMatrix = field(
        kw_only=True, init=False
    )
    """Transformation matrix from root to goal_reference_frame link."""

    _fk_binding: ForwardKinematicsBinding = field(kw_only=True, init=False)
    """Binding for the goal pose."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self._fk_binding = ForwardKinematicsBinding(
            name=PrefixedName("root_T_goal_ref", str(self.name)),
            root=self.root_link,
            tip=self.goal_reference_frame,
            float_variable_data=context.float_variable_data,
        )
        self._fk_binding.bind(context.world)
        self.root_T_goal_reference_frame = self._fk_binding.root_T_tip

        return artifacts

    def on_start(self, context: MotionStatechartContext):
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._fk_binding.bind(context.world)

    @property
    @abstractmethod
    def goal_reference_frame(self) -> KinematicStructureEntity:
        """
        :return: Reference frame for the goal.
        """


@dataclass(eq=False, repr=False)
class CartesianPosition(CartesianTask):
    """
    Move a tip link to a goal position in 3D space.

    This task controls only the position (x, y, z) of the tip link, not its orientation.

    .. warning:: This task does not constrain orientation.
    """

    default_reference_velocity: ClassVar[float] = 0.2

    goal_point: Point3 = field(kw_only=True)
    """Target 3D point to reach."""
    threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold for goal achievement in meters."""

    reference_velocity: Optional[float] = field(
        default_factory=lambda: CartesianPosition.default_reference_velocity,
        kw_only=True,
    )
    """Reference velocity for normalization in m/s."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build motion constraints for reaching the goal position.

        :param context: Provides access to world model and kinematic expressions.
        :return: NodeArtifacts containing constraints and observation conditions.
        """
        artifacts = super().build(context)

        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point

        # Get current tip position in root frame
        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()

        # Add constraints to move tip towards goal
        artifacts.constraints.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=root_P_current,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        # Success condition: distance below threshold
        distance_to_goal = root_P_goal.euclidean_distance(root_P_current)
        artifacts.observation = distance_to_goal < self.threshold

        return artifacts


@dataclass(eq=False, repr=False)
class CartesianPositionTrajectory(CartesianTask):
    """
    Move a tip link to a goal position along a trajectory.
    .. warning:: the trajectory is assumed to be dense and smooth.
    """

    goal_points: list[Point3] = field(kw_only=True)
    """Target 3D point to reach."""
    _goal_points_np: np.ndarray = field(init=False, repr=False)
    """Goal points in numpy format."""
    maximum_skip_ahead: int | None = field(default=None, kw_only=True)
    """
    This limits how many points can be skipped in very dense trajectories.
    Setting this number is required if your trajectory contains loops.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance threshold for goal achievement in meters.
    """

    look_ahead_distance: float = field(default=0.01, kw_only=True)
    """
    Distance from the current position to tracking target.
    Increasing this value can increase the tracking velocity, but might reduce tracking accuracy.
    """

    reference_velocity: Optional[float] = field(
        default_factory=lambda: CartesianPosition.default_reference_velocity,
        kw_only=True,
    )
    """Reference velocity for normalization in m/s."""

    goal_reference_frame_P_current_target_point: Point3 = field(init=False, repr=False)
    """Symbolic expression representing the current target point in the goal reference frame."""

    current_index: int = field(default=0, kw_only=True)
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
                raise NodeInitializationError(
                    self,
                    f"All goal points must have the same reference frame, but got {point.reference_frame} and {reference_frame}.",
                )
        return reference_frame

    def _goal_points_to_np(self):
        self._goal_points_np = np.array(
            [point.to_np()[:-1] for point in self.goal_points]
        )

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._goal_points_to_np()
        artifacts = super().build(context)
        self._init_goal_reference_frame_P_current_target_point(
            context.float_variable_data
        )

        root_P_goal = (
            self.root_T_goal_reference_frame
            @ self.goal_reference_frame_P_current_target_point
        )

        # Get current tip position in root frame
        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()

        # Add constraints to move tip towards goal
        artifacts.constraints.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=root_P_current,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        self.compile_current_point_on_tick(context)
        return artifacts

    def compile_current_point_on_tick(self, context: MotionStatechartContext):
        """
        Computing the current point relative to the goal reference frame is expensive, this method turns it into
        a compiled expression.
        :param context: the current context, needed for the world reference.
        """
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        goal_reference_frame_T_tip = (
            self.root_T_goal_reference_frame.inverse() @ root_T_tip
        )
        goal_reference_frame_P_tip = goal_reference_frame_T_tip.to_position()[:-1]
        self._compiled_goal_reference_frame_P_tip = goal_reference_frame_P_tip.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_goal_reference_frame_P_tip.bind_args_to_memory_view(
            0, context.world.state.positions
        )
        self._compiled_goal_reference_frame_P_tip.bind_args_to_memory_view(
            1, context.float_variable_data.data
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

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Update the target point on the trajectory and return true if we have reached the end of the trajectory.
        """
        goal_reference_frame_P_tip_np = (
            self._compiled_goal_reference_frame_P_tip.evaluate()
        )
        self._update_trajectory_index(goal_reference_frame_P_tip_np)
        target_point = self._compute_target_point(goal_reference_frame_P_tip_np)
        context.float_variable_data.set_value(
            self.goal_reference_frame_P_current_target_point, target_point
        )
        if (
            np.linalg.norm(target_point - goal_reference_frame_P_tip_np)
            < self.threshold
            and self.current_index == len(self._goal_points_np) - 1
        ):
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

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


@dataclass(eq=False, repr=False)
class CartesianPositionStraight(CartesianTask):
    """
    Move a tip link to a goal position along a straight line.

    Unlike CartesianPosition, this task constrains the tip link to move in a straight
    line towards the goal, useful for tasks requiring linear trajectories.
    """

    goal_point: Point3 = field(kw_only=True)
    """Target 3D point to reach."""

    reference_velocity: float = field(
        default=CartesianPosition.default_reference_velocity, kw_only=True
    )
    """Reference velocity for movement in m/s."""

    threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold for goal achievement in meters."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build motion constraints for reaching the goal along a straight line.

        Creates a virtual coordinate frame aligned with the straight-line path and
        constrains motion to stay on that line.

        :param context: Provides access to world model and kinematic expressions.
        :return: NodeArtifacts containing constraints and observation conditions.
        """
        artifacts = super().build(context)

        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point

        # Get current tip position and transformations
        root_P_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        tip_T_root = context.world.compose_forward_kinematics_expression(
            self.tip_link, self.root_link
        )
        tip_P_goal = tip_T_root.dot(root_P_goal)

        # Create coordinate frame aligned with straight-line path
        # x-axis points from current position towards goal
        tip_V_error = Vector3.from_iterable(tip_P_goal)
        trans_error = tip_V_error.norm()
        tip_V_intermediate_error = tip_V_error.safe_division(trans_error)

        # Create orthogonal y and z axes
        tip_V_intermediate_y = Vector3.from_iterable(np.random.random((3,)))
        tip_V_intermediate_y.scale(1)
        y = tip_V_intermediate_error.cross(tip_V_intermediate_y)
        z = tip_V_intermediate_error.cross(y)
        tip_R_aligned = RotationMatrix.from_vectors(
            x=tip_V_intermediate_error, y=-z, z=y
        )

        # Transform tip kinematics into aligned frame
        tip_T_root_evaluated = context.world.compute_forward_kinematics(
            self.tip_link, self.root_link
        )
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        aligned_T_tip = tip_R_aligned.inverse() @ tip_T_root_evaluated @ root_T_tip

        expr_p = aligned_T_tip.to_position()
        dist = (root_P_goal - root_P_tip).norm()

        # Constrain motion: x-axis moves towards goal, y and z stay at zero
        for i, (name, bound, weight_mult) in enumerate(
            [
                ("line/x", dist, 1),
                ("line/y", 0, 2),
                ("line/z", 0, 2),
            ]
        ):
            artifacts.constraints.add_equality_constraint(
                name=name,
                reference_velocity=self.reference_velocity,
                equality_bound=bound,
                quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA * weight_mult,
                task_expression=expr_p[i],
            )

        artifacts.observation = dist < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class CartesianOrientation(CartesianTask):
    """
    Rotate a tip link to match a goal orientation.

    This task controls only the orientation (roll, pitch, yaw) of the tip link,
    not its position.

    .. warning:: This task does not constrain position.
    """

    default_reference_velocity: ClassVar[float] = 0.2

    goal_orientation: RotationMatrix = field(kw_only=True)
    """Target rotation matrix to match."""
    threshold: float = field(default=0.01, kw_only=True)
    """Rotation error threshold for goal achievement in radians."""

    reference_velocity: float = field(
        default_factory=lambda: CartesianOrientation.default_reference_velocity,
        kw_only=True,
    )
    """Reference angular velocity for normalization in rad/s."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_orientation.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build motion constraints for reaching the goal orientation.

        :param context: Provides access to world model and kinematic expressions.
        :return: NodeArtifacts containing constraints and observation conditions.
        """
        artifacts = super().build(context)

        root_R_goal = self.root_T_goal_reference_frame @ self.goal_orientation

        # Get current tip orientation in root frame
        root_T_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_R_current = root_T_current.to_rotation_matrix()

        # Add constraints to rotate tip towards goal
        artifacts.constraints.add_rotation_goal_constraints(
            frame_R_current=root_R_current,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        # Success condition: rotation error below threshold
        rotation_error = root_R_current.rotational_error(root_R_goal)
        artifacts.observation = sm.abs(rotation_error) < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class CartesianPose(CartesianTask):
    """
    This goal will use the kinematic chain between root and tip link to move tip_link into the 6D goal_pose.
    """

    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """The goal pose."""

    reference_linear_velocity: float = field(
        default=CartesianPosition.default_reference_velocity, kw_only=True
    )
    """Unit: m/s. This is used for normalization, for real limits use CartesianVelocityLimit."""

    reference_angular_velocity: float = field(
        default=CartesianOrientation.default_reference_velocity, kw_only=True
    )
    """Unit: rad/s. This is used for normalization, for real limits use CartesianVelocityLimit."""

    threshold: float = field(default=0.01, kw_only=True)
    """If the error falls below this threshold, the goal is achieved. This is used for both position and orientation. Units are m and rad."""

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_pose.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)

        # Use world root if no root link specified
        if self.root_link is None:
            self.root_link = context.world.root

        # Extract position and orientation from goal pose
        goal_reference_frame_R_goal_orientation = self.goal_pose.to_rotation_matrix()
        goal_reference_frame_P_goal_position = self.goal_pose.to_position()

        # Transform goal into root frame
        root_P_goal = (
            self.root_T_goal_reference_frame @ goal_reference_frame_P_goal_position
        )
        root_R_goal = (
            self.root_T_goal_reference_frame @ goal_reference_frame_R_goal_orientation
        )

        # Get current tip pose in root frame
        root_T_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_current = root_T_current.to_position()

        # Add position constraints
        artifacts.constraints.add_point_goal_constraints(
            name="position",
            frame_P_goal=root_P_goal,
            frame_P_current=root_P_current,
            reference_velocity=self.reference_linear_velocity,
            quadratic_weight=self.weight,
        )

        distance_to_goal = root_P_goal.euclidean_distance(root_P_current)

        # Get current orientation
        root_R_current = root_T_current.to_rotation_matrix()

        # Add orientation constraints
        artifacts.constraints.add_rotation_goal_constraints(
            name="rotation",
            frame_R_current=root_R_current,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_angular_velocity,
            quadratic_weight=self.weight,
        )

        rotation_error = root_R_current.rotational_error(root_R_goal)
        artifacts.observation = sm.logic_and(
            sm.abs(rotation_error) < self.threshold,
            distance_to_goal < self.threshold,
        )

        return artifacts


@dataclass(eq=False, repr=False)
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

    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    Root link of the kinematic chain. 
    Defines the reference frame from which the tip's motion is measured.
    """
    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    Tip link of the kinematic chain. 
    The translational velocity of this link (expressed in the root link frame) is constrained.
    """
    max_linear_velocity: float = field(default=0.1, kw_only=True)
    """
    Maximum allowed linear speed of the tip in meters per second (m/s).
    Default: 0.1 m/s. The enforcement ensures the Euclidean norm of the
    tip-frame translational velocity does not exceed this value.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """
    Optimization weight determining how strongly the linear velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        root_P_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        artifacts.constraints.add_translational_velocity_limit(
            frame_P_current=root_P_tip,
            max_velocity=self.max_linear_velocity,
            quadratic_weight=self.weight,
        )

        position_variables: List[PositionVariable] = root_P_tip.free_variables()
        velocity_variables = [p.dof.variables.velocity for p in position_variables]
        root_P_tip_dot = root_P_tip.total_derivative(
            position_variables, velocity_variables
        )

        artifacts.observation = root_P_tip_dot.norm() <= self.max_linear_velocity

        return artifacts


@dataclass(eq=False, repr=False)
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

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain. Defines the reference frame from which the tip's motion is measured."""
    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Tip link of the kinematic chain. The translational velocity of this link (expressed in the root link frame) is constrained."""
    max_angular_velocity: float = field(default=0.4, kw_only=True)
    """Maximum allowed angular speed. Interpreted in radians per second (rad/s).
    The enforcement ensures the magnitude of the instantaneous
    rotation rate does not exceed this threshold."""
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """Optimization weight determining how strongly the rotational velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_R_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()

        artifacts.constraints.add_rotational_velocity_limit(
            frame_R_current=root_R_tip,
            max_velocity=self.max_angular_velocity,
            quadratic_weight=self.weight,
        )

        _, angle = root_R_tip.to_axis_angle()
        angle_variables: List[PositionVariable] = angle.free_variables()
        angle_velocities = [v.dof.variables.velocity for v in angle_variables]
        angle_dot = angle.total_derivative(angle_variables, angle_velocities)

        artifacts.observation = sm.abs(angle_dot) <= self.max_angular_velocity

        return artifacts


@dataclass(eq=False, repr=False)
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

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain. Defines the reference frame from which the tip's motion is measured."""
    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Tip link of the kinematic chain. Both translational and rotational velocities of this link (expressed in the root link frame) are constrained."""
    max_linear_velocity: float = field(default=0.1, kw_only=True)
    """Maximum allowed linear speed of the tip in meters per second (m/s).
    Default: 0.1 m/s. The enforcement ensures the Euclidean norm of the
    tip-frame translational velocity does not exceed this value."""
    max_angular_velocity: float = field(default=0.4, kw_only=True)
    """Maximum allowed angular speed. Interpreted in radians per second (rad/s).
    Default: 0.5 rad/s. The enforcement ensures the magnitude of the instantaneous
    rotation rate does not exceed this threshold."""
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """Optimization weight determining how strongly both velocity
    limits are enforced. Higher weights give these constraints soft priority
    over lower weighted constraints when conflicts occur."""
    nodes: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """List of motion nodes that run in parallel and enforce the velocity limits.
    Contains a CartesianPositionVelocityLimit and CartesianRotationVelocityLimit node 
    by default. Populated in __post_init__()."""

    def __post_init__(self):
        super().__post_init__()

        translational = CartesianPositionVelocityLimit(
            root_link=self.root_link,
            tip_link=self.tip_link,
            max_linear_velocity=self.max_linear_velocity,
            weight=self.weight,
        )
        rotational = CartesianRotationVelocityLimit(
            root_link=self.root_link,
            tip_link=self.tip_link,
            max_angular_velocity=self.max_angular_velocity,
            weight=self.weight,
        )
        self.nodes.append(translational)
        self.nodes.append(rotational)


@dataclass(eq=False, repr=False)
class CartesianPositionVelocityTarget(Task):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    x_vel: float = field(kw_only=True)
    y_vel: float = field(kw_only=True)
    z_vel: float = field(kw_only=True)
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default DefaultWeights.WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        r_P_c = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        self.add_velocity_eq_constraint_vector(
            velocity_goals=sm.Vector([self.x_vel, self.y_vel, self.z_vel]),
            task_expression=r_P_c,
            reference_velocities=[
                CartesianPosition.default_reference_velocity,
                CartesianPosition.default_reference_velocity,
                CartesianPosition.default_reference_velocity,
            ],
            names=[
                f"{self.name}/x",
                f"{self.name}/y",
                f"{self.name}/z",
            ],
            weights=[self.weight] * 3,
        )
