from dataclasses import field, dataclass
from itertools import combinations

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar, FloatVariable
from semantic_digital_twin.collision_checking.collision_groups import CollisionGroup
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import (
    Vector3,
    Point3,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)
from ..context import MotionStatechartContext
from ..data_types import DefaultWeights
from ..graph_node import Goal, MotionStatechartNode, NodeArtifacts
from ..graph_node import Task
from ...qp.qp_controller_config import QPControllerConfig


@dataclass(eq=False, repr=False)
class _CollisionAvoidanceTask(Task):
    """
    Superclass with helper methods for collision avoidance tasks.
    """

    def create_upper_slack(
        self,
        context: MotionStatechartContext,
        buffer_zone_expr: sm.Scalar,
        violated_distance: sm.Scalar,
        distance_expression: sm.Scalar,
        max_velocity: float = 0.2,
    ) -> sm.Scalar:
        """
        Creates upper slack limits for collision avoidance tasks that is just large enough to reach the violated distance.
        :param context: The motion statechart context.
        :param buffer_zone_expr: The buffer zone expression for the collision avoidance task.
        :param violated_distance: The violated distance for the collision avoidance task.
        :param distance_expression: The distance expression for the collision avoidance task.
        :param max_velocity: The maximum velocity for the collision avoidance task.
        :return: The upper slack limit for the collision avoidance task.
        """
        distance_to_buffer_zone = buffer_zone_expr - distance_expression
        qp_limits_for_lower_error_bound = (
            max_velocity
            * context.qp_controller_config.mpc_dt
            * self.compute_control_horizon(context.qp_controller_config)
        )

        hard_threshold = sm.min(violated_distance, buffer_zone_expr / 2)

        lower_limit_limited = sm.limit(
            distance_to_buffer_zone,
            -qp_limits_for_lower_error_bound,
            qp_limits_for_lower_error_bound,
        )

        upper_slack = sm.if_greater(
            distance_expression,
            hard_threshold,
            lower_limit_limited + sm.max(0, distance_expression - hard_threshold),
            lower_limit_limited - 1e-4,
        )
        # undo factor in A
        upper_slack /= context.qp_controller_config.mpc_dt

        upper_slack = sm.if_greater(
            distance_expression,
            50,  # assuming that distance of unchecked closest points is 100
            sm.Scalar(1e4),
            sm.max(0, upper_slack),
        )
        return upper_slack

    def compute_control_horizon(
        self, qp_controller_config: QPControllerConfig
    ) -> float:
        """
        Computes the control horizon for the QP controller.
        """
        control_horizon = qp_controller_config.prediction_horizon - (
            qp_controller_config.max_derivative - 1
        )
        return max(1, control_horizon)


@dataclass(eq=False, repr=False)
class _ExternalCollisionAvoidanceNode(_CollisionAvoidanceTask):
    """
    Avoids external collisions between a collision group and its collision_index-closest object in the environment.
    Moves `root_T_tip @ tip_P_contact` in `root_T_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    .. warning: Can result in insolvable QPs if multiple of these constraints are violated.
    """

    collision_group: CollisionGroup = field(kw_only=True)
    """
    The collision group avoiding external collisions.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    collision_index: int = field(default=0, kw_only=True)
    """
    The index of the closest object in the collision group.
    e.g. of collision_index=1 it will avoid the 2. closest contact.
    """
    external_collision_manager: ExternalCollisionVariableManager = field(kw_only=True)
    """
    Reference to the external collision variable manager shared by other external collision avoidance nodes.
    """

    @property
    def root_V_contact_normal(self) -> Vector3:
        return self.external_collision_manager.get_root_V_contact_normal_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def group_a_P_point_on_a(self) -> Point3:
        return self.external_collision_manager.get_group_a_P_point_on_a_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def contact_distance(self):
        return self.external_collision_manager.get_contact_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def buffer_zone_distance(self):
        return self.external_collision_manager.get_buffer_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def violated_distance(self):
        return self.external_collision_manager.get_violated_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def has_collision_data(self) -> Scalar:
        return self.root_V_contact_normal.norm() == 0


@dataclass(eq=False, repr=False)
class _ExternalCollisionHasData(_ExternalCollisionAvoidanceNode):
    """
    Monitors whether data was computed for the external collision avoidance task.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = self.has_collision_data

        return artifacts


@dataclass(eq=False, repr=False)
class _ExternalCollisionAvoidanceTask(_ExternalCollisionAvoidanceNode):
    """
    Avoids external collisions between a collision group and its collision_index-closest object in the environment.
    Moves `root_T_tip @ tip_P_contact` in `root_T_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    .. warning: Can result in insolvable QPs if multiple of these constraints are violated.
    """

    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """

    @property
    def tip(self) -> KinematicStructureEntity:
        return self.collision_group.root

    def create_weight(self, context: MotionStatechartContext) -> sm.Scalar:
        """
        Creates a weight expression for this task which is scaled by the number of external collisions.
        """
        max_avoided_bodies = self.collision_group.get_max_avoided_bodies(
            context.collision_manager
        )
        number_of_external_collisions = 0
        for index in range(max_avoided_bodies):
            has_collision_data = (
                self.external_collision_manager.get_root_V_contact_normal_symbol(
                    self.collision_group, index
                ).norm()
            )
            is_active = has_collision_data > 0
            number_of_external_collisions += is_active
        weight = sm.Scalar(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(sm.min(number_of_external_collisions, max_avoided_bodies))
        return weight

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_T_group_a = context.world.compose_forward_kinematics_expression(
            context.world.root, self.tip
        )

        root_V_point_on_a = (root_T_group_a @ self.group_a_P_point_on_a).to_vector3()

        # the position distance is not accurate, but the derivative is still correct
        a_projected_on_normal = self.root_V_contact_normal @ root_V_point_on_a

        lower_limit = self.buffer_zone_distance - self.contact_distance

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=self.create_weight(context),
            task_expression=a_projected_on_normal,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=self.create_upper_slack(
                context=context,
                max_velocity=self.max_velocity,
                buffer_zone_expr=self.buffer_zone_distance,
                violated_distance=self.violated_distance,
                distance_expression=sm.Scalar(self.contact_distance),
            ),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidance(Goal):
    """
    A goal combining an ExternalCollisionDistanceMonitor and an ExternalCollisionAvoidanceTask.
    One pair will be added for all collision groups of the robot.
    The task will only be active if the monitor detects that a collision is close.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot for which the collision avoidance goal is defined.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    external_collision_manager: ExternalCollisionVariableManager = field(init=False)
    """
    Reference to the external collision variable manager shared by other external collision avoidance nodes.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.external_collision_manager = context.external_collision_manager

        for body in self.robot.bodies_with_collision:
            if context.collision_manager.get_max_avoided_bodies(body):
                self.external_collision_manager.register_group_of_body(body)

        for group in self.external_collision_manager.registered_groups:
            max_avoided_bodies = group.get_max_avoided_bodies(context.collision_manager)
            for index in range(max_avoided_bodies):
                distance_monitor = _ExternalCollisionHasData(
                    name=f"{self.name}/monitor({group.root.name.name, index})",
                    collision_group=group,
                    collision_index=index,
                    external_collision_manager=self.external_collision_manager,
                )
                self.add_node(distance_monitor)

                task = _ExternalCollisionAvoidanceTask(
                    name=f"{self.name}/task({group.root.name.name, index})",
                    collision_group=group,
                    max_velocity=self.max_velocity,
                    collision_index=index,
                    external_collision_manager=self.external_collision_manager,
                )
                self.add_node(task)
                task.pause_condition = distance_monitor.observation_variable


@dataclass(eq=False, repr=False)
class ExternalCollisionDistanceMonitor(MotionStatechartNode):
    """
    Monitors the distance to the closest external object for a specific collision group of a body.
    Turns True if the distance falls below a given threshold.

    .. note:: the input bodies are only used to look up the collision groups.
    """

    body: Body = field(kw_only=True)
    """The robot body to monitor."""
    threshold: float = field(kw_only=True)
    """Distance threshold in meters."""
    collision_index: int = field(default=0, kw_only=True)
    """Index of the closest collision (0 = closest, 1 = second closest, etc.)."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        # 1. Access the shared external collision manager
        # This automatically registers the manager with the CollisionManager
        manager = context.external_collision_manager

        # 2. Register the body to ensure the manager tracks its collisions
        manager.register_group_of_body(self.body)
        group = manager.get_collision_group(self.body)

        # 3. Retrieve the symbolic variable for the contact distance
        distance_symbol = manager.get_contact_distance_symbol(
            group=group, idx=self.collision_index
        )

        # 4. Return an observation artifact
        # The node's observation_variable will be True when distance < threshold
        return NodeArtifacts(observation=distance_symbol < self.threshold)


@dataclass(eq=False, repr=False)
class _SelfCollisionAvoidanceNode(_CollisionAvoidanceTask):
    """
    Avoids self collisions between two collision groups.
    Moves `group_a_P_point_on_a @ group_b_P_point_on_b` in `group_a_T_group_b_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    """

    collision_group_a: CollisionGroup = field(kw_only=True)
    """
    The first collision group to avoid self collisions with.
    """
    collision_group_b: CollisionGroup = field(kw_only=True)
    """
    The second collision group to avoid self collisions with.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    self_collision_manager: SelfCollisionVariableManager = field(kw_only=True)
    """
    Reference to the self collision variable manager shared by other self collision avoidance nodes.
    """

    @property
    def group_a_P_point_on_a(self) -> Point3:
        return self.self_collision_manager.get_group_a_P_point_on_a_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def group_b_P_point_on_b(self) -> Point3:
        return self.self_collision_manager.get_group_b_P_point_on_b_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def group_b_V_contact_normal(self) -> Vector3:
        return self.self_collision_manager.get_group_b_V_contact_normal_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def contact_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_contact_distance_symbol(
            self.collision_group_a,
            self.collision_group_b,
        )

    @property
    def buffer_zone_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_buffer_distance_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def violated_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_violated_distance_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def has_collision_data(self) -> Scalar:
        return self.group_b_V_contact_normal.norm() == 0


@dataclass(eq=False, repr=False)
class _SelfCollisionHasData(_SelfCollisionAvoidanceNode):
    """
    Monitors whether data was computed for the self collision avoidance task.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = self.has_collision_data

        return artifacts


@dataclass(eq=False, repr=False)
class _SelfCollisionAvoidanceTask(_SelfCollisionAvoidanceNode):
    """
    Avoids self collisions between two collision groups.
    Moves `group_a_P_point_on_a @ group_b_P_point_on_b` in `group_a_T_group_b_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    """

    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        group_b_T_group_a = context.world.compose_forward_kinematics_expression(
            self.collision_group_b.root, self.collision_group_a.root
        )

        group_b_P_point_on_a = group_b_T_group_a @ self.group_a_P_point_on_a

        group_b_V_point_on_b_to_point_on_a = (
            group_b_P_point_on_a - self.group_b_P_point_on_b
        )

        a_projected_on_normal = (
            self.group_b_V_contact_normal @ group_b_V_point_on_b_to_point_on_a
        )

        lower_limit = self.buffer_zone_distance - self.contact_distance

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            task_expression=a_projected_on_normal,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=self.create_upper_slack(
                context=context,
                max_velocity=self.max_velocity,
                buffer_zone_expr=self.buffer_zone_distance,
                violated_distance=self.violated_distance,
                distance_expression=sm.Scalar(self.contact_distance),
            ),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidance(Goal):
    """
    A goal combining a SelfCollisionDistanceMonitor and a SelfCollisionAvoidanceTask.
    One pair will be added for all collision groups of the robot.
    The task will only be active if the monitor detects that a collision is close.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot for which the collision avoidance goal is defined.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    self_collision_manager: SelfCollisionVariableManager = field(init=False)
    """
    Reference to the self collision variable manager shared by other self collision avoidance nodes.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.self_collision_manager = context.self_collision_manager
        context.collision_manager.update_collision_matrix()

        for group_a, group_b in combinations(
            self.self_collision_manager.collision_groups, 2
        ):
            if (
                group_a.root not in self.robot.kinematic_structure_entities
                or group_b.root not in self.robot.kinematic_structure_entities
            ):
                # this is no self collision
                continue
            if not self.self_collision_manager.is_any_collision_checked(
                group_a, group_b
            ):
                continue
            self.self_collision_manager.register_groups_of_body_combination(
                group_a.root, group_b.root
            )
            (group_a, group_b) = self.self_collision_manager.body_pair_to_group_pair(
                group_a.root, group_b.root
            )

            distance_monitor = _SelfCollisionHasData(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/monitor",
                collision_group_a=group_a,
                collision_group_b=group_b,
                self_collision_manager=self.self_collision_manager,
            )
            self.add_node(distance_monitor)

            task = _SelfCollisionAvoidanceTask(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/task",
                collision_group_a=group_a,
                collision_group_b=group_b,
                max_velocity=self.max_velocity,
                self_collision_manager=self.self_collision_manager,
            )
            self.add_node(task)
            task.pause_condition = distance_monitor.observation_variable


@dataclass(eq=False, repr=False)
class SelfCollisionDistanceMonitor(MotionStatechartNode):
    """
    Monitors the distance to the closest external object for the group of a body.
    Turns True if the distance falls below a given threshold.
    .. note:: the input bodies are only used to look up the collision groups.
    """

    body_a: Body = field(kw_only=True)
    """First robot body to monitor."""
    body_b: Body = field(kw_only=True)
    """Second robot body to monitor."""
    threshold: float = field(kw_only=True)
    """Distance threshold in meters."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        manager = context.self_collision_manager

        manager.register_groups_of_body_combination(self.body_a, self.body_b)
        (group_a, group_b) = manager.body_pair_to_group_pair(self.body_a, self.body_b)

        distance_symbol = manager.get_contact_distance_symbol(group_a, group_b)

        return NodeArtifacts(observation=distance_symbol < self.threshold)
