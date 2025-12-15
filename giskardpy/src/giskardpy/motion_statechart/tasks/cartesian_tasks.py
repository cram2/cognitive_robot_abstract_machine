from dataclasses import field, dataclass
from typing import Optional, ClassVar

import numpy as np
from typing_extensions import List

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart import auxilary_variable_manager
from giskardpy.motion_statechart.binding_policy import (
    GoalBindingPolicy,
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.context import BuildContext, ExecutionContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import (
    NodeArtifacts,
    DebugExpression,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class CartesianPosition(Task):
    """
    This Task will use the kinematic chain between root and tip link to move tip_link into goal_point.
    .. warning:: This task does not constrain orientation.
    """

    default_reference_velocity: ClassVar[float] = 0.2
    root_link: KinematicStructureEntity = field(kw_only=True)
    tip_link: KinematicStructureEntity = field(kw_only=True)
    goal_point: cas.Point3 = field(kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    reference_velocity: Optional[float] = field(
        default_factory=lambda: CartesianPosition.default_reference_velocity,
        kw_only=True,
    )
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    absolute: bool = field(default=False, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_P_goal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        )

        r_P_c = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=r_P_c,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )

        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)
        artifacts.observation = distance_to_goal < self.threshold
        return artifacts


@dataclass
class CartesianPositionStraight(Task):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    goal_point: cas.Point3 = field(kw_only=True)
    threshold: float = 0.01
    reference_velocity: Optional[float] = CartesianPosition.default_reference_velocity
    absolute: bool = False
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        Same as CartesianPosition, but tries to move the tip_link in a straight line to the goal_point.
        """
        if self.absolute:
            root_P_goal = context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        else:
            root_T_x = context.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_point.reference_frame
            )
            root_P_goal = root_T_x.dot(self.goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        root_P_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        t_T_r = context.world.compose_forward_kinematics_expression(
            self.tip_link, self.root_link
        )
        tip_P_goal = t_T_r.dot(root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_V_error = cas.Vector3.from_iterable(tip_P_goal)
        trans_error = tip_V_error.norm()
        # x-axis
        tip_V_intermediate_error = tip_V_error.safe_division(trans_error)
        # y- and z-axis
        tip_V_intermediate_y = cas.Vector3.from_iterable(np.random.random((3,)))
        tip_V_intermediate_y.scale(1)
        y = tip_V_intermediate_error.cross(tip_V_intermediate_y)
        z = tip_V_intermediate_error.cross(y)
        t_R_a = cas.RotationMatrix.from_vectors(x=tip_V_intermediate_error, y=-z, z=y)

        # Apply rotation matrix on the fk of the tip link
        tip_T_root = context.world.compute_forward_kinematics(
            self.tip_link, self.root_link
        )
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        a_T_t = t_R_a.inverse() @ tip_T_root @ root_T_tip

        expr_p = a_T_t.to_position()
        dist = (root_P_goal - root_P_tip).norm()

        self.add_equality_constraint_vector(
            reference_velocities=[self.reference_velocity] * 3,
            equality_bounds=[dist, 0, 0],
            weights=[
                DefaultWeights.WEIGHT_ABOVE_CA,
                DefaultWeights.WEIGHT_ABOVE_CA * 2,
                DefaultWeights.WEIGHT_ABOVE_CA * 2,
            ],
            task_expression=expr_p[:3],
            names=["line/x", "line/y", "line/z"],
        )
        self.observation_expression = dist < self.threshold


@dataclass
class CartesianOrientation(Task):
    """
    This Task will use the kinematic chain between root and tip link to move tip_link into goal_orientation.
    .. warning:: This task does not constrain position.
    """

    default_reference_velocity: ClassVar[float] = 0.2
    root_link: KinematicStructureEntity = field(kw_only=True)
    tip_link: KinematicStructureEntity = field(kw_only=True)
    goal_orientation: cas.RotationMatrix = field(kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    reference_velocity: float = field(
        default_factory=lambda: CartesianOrientation.default_reference_velocity,
        kw_only=True,
    )
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    absolute: bool = field(default=False, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        # if self.absolute:
        root_R_goal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_orientation
        )
        # else:
        #     root_T_x = context.world.compose_forward_kinematics_expression(
        #         self.root_link, self.goal_orientation.reference_frame
        #     )
        #     root_R_goal = root_T_x.dot(self.goal_orientation)
        #     root_R_goal = self.update_expression_on_starting(root_R_goal)

        r_T_c = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        r_R_c = r_T_c.to_rotation_matrix()

        artifacts.constraints.add_rotation_goal_constraints(
            frame_R_current=r_R_c,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )

        rotation_error = r_R_c.rotational_error(root_R_goal)
        artifacts.observation = cas.abs(rotation_error) < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class CartesianPose(Task):
    """
    This goal will use the kinematic chain between root and tip link to move tip_link into the 6D goal_pose.
    """

    root_link: Optional[KinematicStructureEntity] = field(kw_only=True, default=None)
    """Name of the root link of the kin chain"""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Name of the tip link of the kin chain"""

    goal_pose: cas.TransformationMatrix = field(kw_only=True)
    """The goal pose"""

    reference_linear_velocity: float = field(
        default=CartesianPosition.default_reference_velocity, kw_only=True
    )
    """
    Unit: m/s
    This is used for normalization, for real limits use CartesianVelocityLimit.
    """

    reference_angular_velocity: float = field(
        default=CartesianOrientation.default_reference_velocity, kw_only=True
    )
    """
    Unit: rad/s
    This is used for normalization, for real limits use CartesianVelocityLimit.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    If the error falls below this threshold, the goal is achieved.
    This is used for both position and orientation.
    Units are m and rad.
    """

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_on_start, kw_only=True
    )
    """Describes when the goal is computed. See GoalBindingPolicy for more information."""
    _fk_binding: ForwardKinematicsBinding = field(kw_only=True, init=False)

    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        if self.root_link is None:
            self.root_link = context.world.root

        self._fk_binding = ForwardKinematicsBinding(
            name=PrefixedName("root_T_ref", str(self.name)),
            root=self.root_link,
            tip=self.goal_pose.reference_frame,
            build_context=context,
        )

        goal_orientation = self.goal_pose.to_rotation_matrix()
        goal_point = self.goal_pose.to_position()

        root_P_goal = self._fk_binding.root_T_tip @ goal_point
        root_R_goal = self._fk_binding.root_T_tip @ goal_orientation

        r_T_c = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        r_P_c = r_T_c.to_position()
        artifacts.constraints.add_point_goal_constraints(
            name="position",
            frame_P_goal=root_P_goal,
            frame_P_current=r_P_c,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)

        r_R_c = r_T_c.to_rotation_matrix()

        artifacts.constraints.add_rotation_goal_constraints(
            name="rotation",
            frame_R_current=r_R_c,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_angular_velocity,
            weight=self.weight,
        )

        artifacts.debug_expressions.append(
            DebugExpression(
                "current_pose",
                expression=cas.TransformationMatrix(reference_frame=self.tip_link),
            )
        )
        artifacts.debug_expressions.append(
            DebugExpression(
                "goal_pose",
                expression=self._fk_binding.root_T_tip @ self.goal_pose,
            )
        )

        rotation_error = r_R_c.rotational_error(root_R_goal)
        artifacts.observation = cas.logic_and(
            cas.abs(rotation_error) < self.threshold,
            distance_to_goal < self.threshold,
        )

        return artifacts

    def on_start(self, context: ExecutionContext):
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._fk_binding.bind(context.world)


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
    """Root link of the kinematic chain. Defines the reference frame from which the tip's motion is measured."""
    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Tip link of the kinematic chain. The translational velocity of this link (expressed in the root link frame) is constrained."""
    max_linear_velocity: float = field(default=0.1, kw_only=True)
    """Maximum allowed linear speed of the tip in meters per second (m/s).
    Default: 0.1 m/s. The enforcement ensures the Euclidean norm of the
    tip-frame translational velocity does not exceed this value."""
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """Optimization weight determining how strongly the linear velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur."""

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        root_P_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        artifacts.constraints.add_translational_velocity_limit(
            frame_P_current=root_P_tip,
            max_velocity=self.max_linear_velocity,
            weight=self.weight,
        )

        position_variables: List[PositionVariable] = root_P_tip.free_variables()
        velocity_variables = [p.dof.variables.velocity for p in position_variables]
        root_P_tip_dot = cas.Expression(root_P_tip).total_derivative(
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
    max_angular_velocity: float = field(default=0.5, kw_only=True)
    """Maximum allowed angular speed. Interpreted in radians per second (rad/s).
    The enforcement ensures the magnitude of the instantaneous
    rotation rate does not exceed this threshold."""
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """Optimization weight determining how strongly the rotational velocity
    limit is enforced. Higher weights give this constraint soft priority
    over lower weighted constraints when conflicts occur."""

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_R_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()

        artifacts.constraints.add_rotational_velocity_limit(
            frame_R_current=root_R_tip,
            max_velocity=self.max_angular_velocity,
            weight=self.weight,
        )

        _, angle = root_R_tip.to_axis_angle()
        angle_variables: List[PositionVariable] = angle.free_variables()
        angle_velocities = [v.dof.variables.velocity for v in angle_variables]
        angle_dot = cas.Expression(angle).total_derivative(
            angle_variables, angle_velocities
        )

        artifacts.observation = cas.abs(angle_dot) <= self.max_angular_velocity

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
    max_angular_velocity: float = field(default=0.5, kw_only=True)
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


@dataclass
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
            velocity_goals=cas.Expression([self.x_vel, self.y_vel, self.z_vel]),
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


@dataclass
class JustinTorsoLimitCart(Task):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    forward_distance: float = field(kw_only=True)
    backward_distance: float = field(kw_only=True)
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        torso_root_T_torso_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        torso_root_V_up = cas.Vector3(0, 0, 1)
        torso_root_V_up.reference_frame = self.root_link
        torso_root_V_up.vis_frame = self.root_link

        torso_root_V_left = cas.Vector3(0, 1, 0)
        torso_root_V_left.reference_frame = self.root_link
        torso_root_V_left.vis_frame = self.root_link

        torso_root_P_torso_tip = torso_root_T_torso_tip.to_position()

        nearest, distance = torso_root_P_torso_tip.project_to_plane(
            frame_V_plane_vector1=torso_root_V_left,
            frame_V_plane_vector2=torso_root_V_up,
        )
        # distance = cas.distance_point_to_line(torso_root_P_torso_tip, cas.Point3((0, 0, 0)), torso_root_V_up)

        # god_map.context.add_debug_expression(f'{self.name}/torso_root_V_up',
        #                                                       expression=torso_root_V_up)
        # god_map.context.add_debug_expression(f'{self.name}/torso_root_P_torso_tip',
        #                                                       expression=torso_root_P_torso_tip)

        self.add_inequality_constraint(
            reference_velocity=CartesianPosition.default_reference_velocity,
            lower_error=-self.backward_distance - distance,
            upper_error=self.forward_distance - distance,
            weight=self.weight,
            task_expression=distance,
            name=f"{self.name}/distance",
        )
        self.observation_expression = distance <= self.forward_distance
