from collections import defaultdict
from dataclasses import field, dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.graph_node import (
    Task,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import Scalar, FloatVariable
from semantic_digital_twin.collision_checking.collision_groups import CollisionGroup
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import (
    Vector3,
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class CollisionAvoidanceTask(Task):
    def create_upper_slack(
        self,
        context: BuildContext,
        lower_limit: sm.Scalar,
        buffer_zone_expr: sm.Scalar,
        violated_distance: sm.Scalar,
        distance_expression: sm.Scalar,
    ) -> sm.Scalar:
        qp_limits_for_lba = (
            self.max_velocity
            * context.qp_controller_config.mpc_dt
            * self.compute_control_horizon(context.qp_controller_config)
        )

        hard_threshold = sm.min(violated_distance, buffer_zone_expr / 2)

        lower_limit_limited = sm.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
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
        control_horizon = qp_controller_config.prediction_horizon - (
            qp_controller_config.max_derivative - 1
        )
        return max(1, control_horizon)


@dataclass(eq=False, repr=False)
class ExternalCollisionDistanceMonitor(MotionStatechartNode):
    collision_group: CollisionGroup = field(kw_only=True)
    collision_index: int = field(default=0, kw_only=True)
    external_collision_manager: ExternalCollisionVariableManager = field(kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = (
            self.external_collision_manager.get_contact_distance_symbol(
                self.collision_group.root, self.collision_index
            )
            > Scalar(50)
        )

        return artifacts


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidanceTask(CollisionAvoidanceTask):
    """
    Moves root_T_tip @ tip_P_contact in root_T_contact_normal direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    Can result in insolvable QPs if multiple of these constraints are violated.
    """

    collision_group: CollisionGroup = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    collision_index: int = field(default=0, kw_only=True)
    external_collision_manager: ExternalCollisionVariableManager = field(kw_only=True)

    @property
    def tip(self) -> KinematicStructureEntity:
        return self.collision_group.root

    def create_weight(self, context: BuildContext) -> sm.Scalar:
        """
        Creates a weight for this task which is scaled by the number of external collisions.
        :return:
        """
        max_avoided_bodies = self.collision_group.get_max_avoided_bodies(
            context.collision_manager
        )
        number_of_external_collisions = 0
        for index in range(max_avoided_bodies):
            distance_variable = (
                self.external_collision_manager.get_contact_distance_symbol(
                    self.collision_group.root, index
                )
            )
            is_active = distance_variable < 50
            number_of_external_collisions += is_active
        weight = sm.Scalar(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(sm.min(number_of_external_collisions, max_avoided_bodies))
        return weight

    @property
    def root_V_contact_normal(self) -> Vector3:
        return self.external_collision_manager.get_root_V_contact_normal_symbol(
            self.tip, self.collision_index
        )

    @property
    def group_a_P_point_on_a(self) -> Point3:
        return self.external_collision_manager.get_group_a_P_point_on_a_symbol(
            self.tip, self.collision_index
        )

    @property
    def contact_distance(self):
        return self.external_collision_manager.get_contact_distance_symbol(
            self.tip, self.collision_index
        )

    @property
    def buffer_zone_distance(self):
        return self.external_collision_manager.get_buffer_distance_symbol(
            self.tip, self.collision_index
        )

    @property
    def violated_distance(self):
        return self.external_collision_manager.get_violated_distance_symbol(
            self.tip, self.collision_index
        )

    def build(self, context: BuildContext) -> NodeArtifacts:
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
                lower_limit=lower_limit,
                buffer_zone_expr=self.buffer_zone_distance,
                violated_distance=self.violated_distance,
                distance_expression=sm.Scalar(self.contact_distance),
            ),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidance(Goal):
    """
    Avoidance collision between all direct children of a connection and the environment.
    """

    robot: AbstractRobot = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)

    def expand(self, context: BuildContext) -> None:
        external_collision_manager = ExternalCollisionVariableManager(
            context.float_variable_data
        )
        context.collision_manager.add_collision_consumer(external_collision_manager)
        for body in self.robot.bodies_with_collision:
            if context.collision_manager.get_max_avoided_bodies(body):
                external_collision_manager.register_body(body)

        for group in external_collision_manager.active_groups:
            max_avoided_bodies = group.get_max_avoided_bodies(context.collision_manager)
            for index in range(max_avoided_bodies):
                distance_monitor = ExternalCollisionDistanceMonitor(
                    name=f"{self.name}/monitor({group.root.name.name, index})",
                    collision_group=group,
                    collision_index=index,
                    external_collision_manager=external_collision_manager,
                )
                self.add_node(distance_monitor)

                task = ExternalCollisionAvoidanceTask(
                    name=f"{self.name}/task({group.root.name.name, index})",
                    collision_group=group,
                    max_velocity=self.max_velocity,
                    collision_index=index,
                    external_collision_manager=external_collision_manager,
                )
                self.add_node(task)
                task.pause_condition = distance_monitor.observation_variable


@dataclass(eq=False, repr=False)
class SelfCollisionDistanceMonitor(MotionStatechartNode):
    collision_group_a: CollisionGroup = field(kw_only=True)
    collision_group_b: CollisionGroup = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    collision_index: int = field(default=0, kw_only=True)
    self_collision_manager: SelfCollisionVariableManager = field(kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = self.self_collision_manager.get_contact_distance_symbol(
            self.collision_group_a,
            self.collision_group_b,
        ) > Scalar(50)

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidanceTask(CollisionAvoidanceTask):
    collision_group_a: CollisionGroup = field(kw_only=True)
    collision_group_b: CollisionGroup = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    self_collision_manager: SelfCollisionVariableManager = field(kw_only=True)

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

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        group_b_T_group_a = context.world.compose_forward_kinematics_expression(
            self.collision_group_a.root, self.collision_group_b.root
        )

        group_b_P_point_on_a = group_b_T_group_a @ self.group_a_P_point_on_a

        group_b_V_point_on_b_to_point_on_a = (
            self.group_b_P_point_on_b - group_b_P_point_on_a
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
                lower_limit=lower_limit,
                buffer_zone_expr=self.buffer_zone_distance,
                violated_distance=self.violated_distance,
                distance_expression=sm.Scalar(self.contact_distance),
            ),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidance(Goal):
    robot: AbstractRobot = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)

    def expand(self, context: BuildContext) -> None:
        self_collision_manager = SelfCollisionVariableManager(
            context.float_variable_data
        )
        context.collision_manager.add_collision_consumer(self_collision_manager)
        for group_a, group_b in combinations(
            self_collision_manager.collision_groups, 2
        ):
            self_collision_manager.register_body_combination(group_a.root, group_b.root)
            (group_a, group_b) = self_collision_manager.body_pair_to_group_pair(
                group_a.root, group_b.root
            )

            distance_monitor = SelfCollisionDistanceMonitor(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/monitor",
                collision_group_a=group_a,
                collision_group_b=group_b,
                self_collision_manager=self_collision_manager,
            )
            self.add_node(distance_monitor)

            task = SelfCollisionAvoidanceTask(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/task",
                collision_group_a=group_a,
                collision_group_b=group_b,
                max_velocity=self.max_velocity,
                self_collision_manager=self_collision_manager,
            )
            self.add_node(task)
            task.pause_condition = distance_monitor.observation_variable


# use cases
# avoid all
# allow all
# avoid all then allow something
# avoid only something


@dataclass(eq=False, repr=False)
class CollisionAvoidance(Goal):
    collision_rules: List[CollisionRule] = field(default_factory=list)

    def expand(self, context: BuildContext) -> None:
        context.collision_manager.temporary_rules.extend(self.collision_rules)
        context.collision_manager.add_collision_consumer(
            ExternalCollisionVariableManager()
        )
        context.collision_manager.update_collision_matrix()

        if (
            not self.collision_rules
            or not self.collision_rules[-1].is_allow_all_collision()
        ):
            self.add_external_collision_avoidance_constraints(context)
        if (
            not self.collision_rules
            or not self.collision_rules[-1].is_allow_all_collision()
        ):
            self.add_self_collision_avoidance_constraints(context)
        collision_matrix = (
            context.collision_expression_manager.matrix_manager.compute_collision_matrix()
        )
        context.collision_expression_manager.set_collision_matrix(collision_matrix)

    def add_external_collision_avoidance_constraints(self, context: BuildContext):
        robot: AbstractRobot
        # thresholds = god_map.collision_scene.matrix_manager.external_thresholds
        for robot in context.world.get_semantic_annotations_by_type(AbstractRobot):
            if robot.drive:
                connection_list = robot.controlled_connections.union({robot.drive})
            else:
                connection_list = robot.controlled_connections
            for connection in connection_list:
                bodies = context.world.get_direct_child_bodies_with_collision(
                    connection
                )
                if not bodies:
                    continue
                max_avoided_bodies = 0
                for body in bodies:
                    max_avoided_bodies = max(
                        max_avoided_bodies,
                        body.get_collision_config().max_avoided_bodies,
                    )
                for index in range(max_avoided_bodies):
                    self.add_node(
                        node := ExternalCollisionAvoidance(
                            name=PrefixedName(
                                f"{connection.name}/{index}", str(self.name)
                            ),
                            connection=connection,
                            index=index,
                            max_avoided_bodies=max_avoided_bodies,
                        )
                    )
                    node.plot_specs.visible = False

        # get_middleware().loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    def add_self_collision_avoidance_constraints(self, context: BuildContext):
        counter: Dict[Tuple[Body, Body], float] = defaultdict(float)
        num_constr = 0
        robot: AbstractRobot
        # collect bodies from the same connection to the main body pair
        for robot in context.world.get_semantic_annotations_by_type(AbstractRobot):
            for body_a_original in robot.bodies_with_collision:
                for body_b_original in robot.bodies_with_collision:
                    if (
                        (
                            body_a_original,
                            body_b_original,
                        )
                        in context.world._collision_pair_manager.disabled_collision_pairs
                        or (
                            body_b_original,
                            body_a_original,
                        )
                        in context.world._collision_pair_manager.disabled_collision_pairs
                    ):
                        continue
                    body_a, body_b = (
                        context.world.compute_chain_reduced_to_controlled_connections(
                            body_a_original, body_b_original
                        )
                    )
                    if body_b.id < body_a.id:
                        body_a, body_b = body_b, body_a
                    counter[body_a, body_b] = max(
                        [
                            counter[body_a, body_b],
                            body_a_original.get_collision_config().buffer_zone_distance
                            or 0,
                            body_b_original.get_collision_config().buffer_zone_distance
                            or 0,
                        ]
                    )

        for link_a, link_b in counter:
            # num_of_constraints = min(1, counter[link_a, link_b])
            # for i in range(num_of_constraints):
            #     number_of_repeller = min(link_a.collision_config.max_avoided_bodies,
            #                              link_b.collision_config.max_avoided_bodies)
            ca_goal = SelfCollisionAvoidance(
                body_a=link_a,
                body_b=link_b,
                name=PrefixedName(f"{link_a.name}/{link_b.name}", str(self.name)),
                index=0,
                max_avoided_bodies=1,
                buffer_zone_distance=counter[link_a, link_b],
            )
            ca_goal.plot_specs.visible = False
            self.add_node(ca_goal)
            num_constr += 1
        get_middleware().loginfo(
            f"Adding {num_constr} self collision avoidance constraints."
        )
