from collections import defaultdict
from dataclasses import field, dataclass
from typing import Dict, List, Tuple

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.context import BuildContext, ContextExtension
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
from krrood.symbolic_math.symbolic_math import Scalar
from giskardpy.motion_statechart.variable_managers.collision_expressions import (
    ExternalCollisionVariableManager,
)
from semantic_digital_twin.collision_checking.collision_manager import CollisionGroup
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


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
class ExternalCollisionAvoidanceTask(Task):
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

    def compute_control_horizon(
        self, qp_controller_config: QPControllerConfig
    ) -> float:
        control_horizon = qp_controller_config.prediction_horizon - (
            qp_controller_config.max_derivative - 1
        )
        return max(1, control_horizon)

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

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_V_contact_normal = (
            self.external_collision_manager.get_group1_V_contact_normal_symbol(
                self.tip, self.collision_index
            )
        )
        tip_P_contact = self.external_collision_manager.get_group1_P_point_on_a_symbol(
            self.tip, self.collision_index
        )
        distance_expression = (
            self.external_collision_manager.get_contact_distance_symbol(
                self.tip, self.collision_index
            )
        )

        buffer_zone_expr = self.external_collision_manager.get_buffer_distance_symbol(
            self.tip, self.collision_index
        )
        violated_distance = (
            self.external_collision_manager.get_violated_distance_symbol(
                self.tip, self.collision_index
            )
        )

        map_T_a = context.world.compose_forward_kinematics_expression(
            context.world.root, self.tip
        )

        map_V_pa = (map_T_a @ tip_P_contact).to_vector3()

        # the position distance is not accurate, but the derivative is still correct
        dist = root_V_contact_normal @ map_V_pa

        lower_limit = buffer_zone_expr - distance_expression

        artifacts.constraints.add_inequality_constraint(
            name=self.name,
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=self.create_weight(context),
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=self.create_upper_slack(
                context=context,
                lower_limit=lower_limit,
                buffer_zone_expr=buffer_zone_expr,
                violated_distance=violated_distance,
                distance_expression=sm.Scalar(distance_expression),
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
                    name=f"{self.name}/monitor{index}",
                    collision_group=group,
                    collision_index=index,
                    external_collision_manager=external_collision_manager,
                )
                self.add_node(distance_monitor)

                task = ExternalCollisionAvoidanceTask(
                    name=f"{self.name}/task{index}",
                    collision_group=group,
                    max_velocity=self.max_velocity,
                    collision_index=index,
                    external_collision_manager=external_collision_manager,
                )
                self.add_node(task)
                task.pause_condition = distance_monitor.observation_variable


@dataclass(eq=False, repr=False)
class SelfCollisionDistanceMonitor(MotionStatechartNode):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    collision_index: int = field(default=0, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = (
            context.collision_expression_manager.self_contact_distance_symbol(
                self.body_a, self.body_b, self.collision_index
            )
            > 50
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidanceTask(Task):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    collision_index: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self.root = context.world.root
        self.control_horizon = context.qp_controller_config.prediction_horizon - (
            context.qp_controller_config.max_derivative - 1
        )
        self.control_horizon = max(1, self.control_horizon)
        # buffer_zone_distance = max(self.body_a.collision_config.buffer_zone_distance,
        #                            self.body_b.collision_config.buffer_zone_distance)
        violated_distance = max(
            self.body_a.get_collision_config().violated_distance,
            self.body_b.get_collision_config().violated_distance,
        )
        violated_distance = sm.min(violated_distance, self.buffer_zone_distance / 2)
        actual_distance = (
            context.collision_expression_manager.self_contact_distance_symbol(
                self.body_a, self.body_b, self.collision_index
            )
        )
        number_of_self_collisions = (
            context.collision_expression_manager.self_number_of_collisions_symbol(
                self.body_a, self.body_b
            )
        )
        sample_period = context.qp_controller_config.mpc_dt

        b_T_a = context.world._forward_kinematic_manager.compose_expression(
            self.body_b, self.body_a
        )
        b_P_pb = context.collision_expression_manager.self_new_b_P_pb_symbol(
            self.body_a, self.body_b, self.collision_index
        )
        pb_T_b = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=b_P_pb
        ).inverse()
        a_P_pa = context.collision_expression_manager.self_new_a_P_pa_symbol(
            self.body_a, self.body_b, self.collision_index
        )

        pb_V_n = context.collision_expression_manager.self_new_b_V_n_symbol(
            self.body_a, self.body_b, self.collision_index
        )

        pb_V_pa = Vector3.from_iterable(pb_T_b @ b_T_a @ a_P_pa)

        dist = pb_V_n @ pb_V_pa

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.buffer_zone_distance - actual_distance

        lower_limit_limited = sm.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
        )

        upper_slack = sm.if_greater(
            actual_distance,
            violated_distance,
            lower_limit_limited + sm.max(0, actual_distance - violated_distance),
            lower_limit_limited,
        )

        # undo factor in A
        upper_slack /= sample_period

        upper_slack = sm.if_greater(
            actual_distance,
            50,  # assuming that distance of unchecked closest points is 100
            sm.Scalar(1e4),
            sm.max(0, upper_slack),
        )

        weight = sm.Scalar(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(sm.min(number_of_self_collisions, self.max_avoided_bodies))

        artifacts.constraints.add_inequality_constraint(
            name=self.name,
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=weight,
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=upper_slack,
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidance(Goal):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    index: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(kw_only=True)

    def expand(self, context: BuildContext) -> None:
        distance_monitor = SelfCollisionDistanceMonitor(
            name=PrefixedName("collision distance", str(self.name)),
            body_a=self.body_a,
            body_b=self.body_b,
            index=self.collision_index,
        )
        self.add_node(distance_monitor)

        task = SelfCollisionAvoidanceTask(
            name=PrefixedName(f"task", str(self.name)),
            body_a=self.body_a,
            body_b=self.body_b,
            max_velocity=self.max_velocity,
            index=self.collision_index,
            max_avoided_bodies=self.max_avoided_bodies,
            buffer_zone_distance=self.buffer_zone_distance,
        )
        self.add_node(task)

        task.pause_condition = distance_monitor.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        context.collision_expression_manager.monitor_link_for_self(
            self.body_a, self.body_b, self.collision_index
        )
        return NodeArtifacts()


# use cases
# avoid all
# allow all
# avoid all then allow something
# avoid only something


@dataclass(eq=False, repr=False)
class CollisionAvoidance(Goal):
    collision_rules: List[CollisionRule] = field(default_factory=list)

    def expand(self, context: BuildContext) -> None:
        context.collision_manager.normal_priority_rules.extend(self.collision_rules)
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
