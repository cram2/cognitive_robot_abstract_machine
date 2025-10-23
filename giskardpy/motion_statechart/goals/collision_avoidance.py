from collections import defaultdict
from dataclasses import field
from typing import Dict, Optional, List, Tuple

from line_profiler import profile

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_matrix_manager import CollisionViewRequest
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.motion_statechart.tasks.task import (
    WEIGHT_ABOVE_CA,
    WEIGHT_COLLISION_AVOIDANCE,
    Task,
)
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import Body


@validated_dataclass
class ExternalCA(Goal):
    name: Optional[str] = field(kw_only=True, default=None)
    name_prefix: Optional[str] = field(kw_only=True, default=None)
    connection: ActiveConnection = field(kw_only=True)
    main_body: Body = field(init=False)
    max_velocity: float = field(default=0.2, kw_only=True)
    world: World = field(kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(init=False)
    violated_distance: float = field(init=False)

    def __post_init__(self):
        """
        Don't use me
        """
        self._plot = False
        self.name = f"{self.name_prefix}/{self.__class__.__name__}/{self.connection.name}/{self.idx}"
        self.main_body = self.connection.child
        self.control_horizon = god_map.qp_controller.config.prediction_horizon - (
            god_map.qp_controller.config.max_derivative - 1
        )
        self.control_horizon = max(1, self.control_horizon)
        # threshold = copy.copy(self.robot.collision_config.external_avoidance_threshold[self.main_body])
        # for body in self.robot.get_directly_child_bodies_with_collision(self.connection):
        #     threshold.soft_threshold =

        self.root = self.world.root
        a_P_pa = self.get_closest_point_on_a_in_a()
        map_V_n = self.map_V_n_symbol()
        actual_distance = self.get_actual_distance()
        sample_period = god_map.qp_controller.config.mpc_dt
        number_of_external_collisions = self.get_number_of_external_collisions()

        map_T_a = self.world.compose_forward_kinematics_expression(
            self.root, self.main_body
        )

        map_P_pa = cas.Vector3.from_iterable(map_T_a.dot(a_P_pa))

        # the position distance is not accurate, but the derivative is still correct
        dist = map_V_n @ map_P_pa

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        # soft_threshold = 0
        actual_link_b_hash = self.get_link_b_hash()
        direct_children = self.world.get_direct_child_bodies_with_collision(
            self.connection
        )

        self.buffer_zone_distance = max(
            b.get_collision_config().buffer_zone_distance
            for b in direct_children
            if b.get_collision_config().buffer_zone_distance is not None
        )
        self.violated_distance = max(
            b.get_collision_config().violated_distance for b in direct_children
        )
        b_result_cases = []
        for body in self.world.bodies_with_enabled_collision:
            if body.get_collision_config().buffer_zone_distance is None:
                continue
            if body.get_collision_config().disabled:
                continue
            if (
                body.get_collision_config().buffer_zone_distance
                > self.buffer_zone_distance
            ):
                b_result_cases.append(
                    (body.__hash__(), body.get_collision_config().buffer_zone_distance)
                )

        if len(b_result_cases) > 0:
            buffer_zone_expr = cas.if_eq_cases(
                a=actual_link_b_hash,
                b_result_cases=b_result_cases,
                else_result=cas.Expression(self.buffer_zone_distance),
            )
        else:
            buffer_zone_expr = self.buffer_zone_distance

        hard_threshold = cas.min(self.violated_distance, buffer_zone_expr / 2)
        lower_limit = buffer_zone_expr - actual_distance

        lower_limit_limited = cas.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
        )

        self.actual_distance = actual_distance
        self.hard_threshold = hard_threshold
        upper_slack = cas.if_greater(
            actual_distance,
            hard_threshold,
            lower_limit_limited + cas.max(0, actual_distance - hard_threshold),
            lower_limit_limited - 1e-4,
        )
        # undo factor in A
        upper_slack /= sample_period

        upper_slack = cas.if_greater(
            actual_distance,
            50,  # assuming that distance of unchecked closest points is 100
            cas.Expression(1e4),
            cas.max(0, upper_slack),
        )

        # if 'r_wrist_roll_link' in self.link_name:
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/actual_distance', actual_distance)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/hard_threshold', hard_threshold)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/soft_threshold', soft_threshold)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/upper_slack', upper_slack)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/lower_limit', lower_limit)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/qp_limits_for_lba', qp_limits_for_lba)
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/actual_distance > hard_threshold', cas.greater(actual_distance, hard_threshold))
        #     god_map.debug_expression_manager.add_debug_expression(f'{self.name}/soft_threshold - hard_threshold', soft_threshold - hard_threshold)

        # weight = cas.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        weight = cas.Expression(data=WEIGHT_COLLISION_AVOIDANCE).safe_division(
            cas.min(number_of_external_collisions, self.max_avoided_bodies)
        )
        distance_monitor = Monitor(name=f"collision distance {self.name}", _plot=False)
        distance_monitor.observation_expression = actual_distance > 50
        self.add_monitor(distance_monitor)
        task = Task(name=self.name + "/task", _plot=False)
        self.add_task(task)
        task.plot = False
        task.pause_condition = distance_monitor
        task.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=weight,
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=upper_slack,
        )

    def map_V_n_symbol(self):
        return god_map.collision_scene.external_map_V_n_symbol(self.main_body, self.idx)

    def get_closest_point_on_a_in_a(self):
        return god_map.collision_scene.external_new_a_P_pa_symbol(
            self.main_body, self.idx
        )

    def get_actual_distance(self):
        return god_map.collision_scene.external_contact_distance_symbol(
            self.main_body, self.idx
        )

    def get_link_b_hash(self):
        return god_map.collision_scene.external_link_b_hash_symbol(
            self.main_body, self.idx
        )

    def get_number_of_external_collisions(self):
        return god_map.collision_scene.external_number_of_collisions_symbol(
            self.main_body
        )


@validated_dataclass
class SelfCA(Goal):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    name: Optional[str] = field(kw_only=True, default=None)
    name_prefix: Optional[str] = field(kw_only=True, default=None)
    max_velocity: float = field(default=0.2, kw_only=True)
    world: World = field(kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(kw_only=True)

    def __post_init__(self):
        self._plot = False
        self.name = f"{self.name_prefix}/{self.__class__.__name__}/{self.body_a.name}/{self.body_b.name}/{self.idx}"
        self.root = self.world.root
        self.control_horizon = god_map.qp_controller.config.prediction_horizon - (
            god_map.qp_controller.config.max_derivative - 1
        )
        self.control_horizon = max(1, self.control_horizon)
        # buffer_zone_distance = max(self.body_a.collision_config.buffer_zone_distance,
        #                            self.body_b.collision_config.buffer_zone_distance)
        violated_distance = max(
            self.body_a.get_collision_config().violated_distance,
            self.body_b.get_collision_config().violated_distance,
        )
        violated_distance = cas.min(violated_distance, self.buffer_zone_distance / 2)
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        sample_period = god_map.qp_controller.config.mpc_dt

        b_T_a = god_map.world.compose_forward_kinematics_expression(
            self.body_b, self.body_a
        )
        pb_T_b = self.get_b_T_pb().inverse()
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_V_pa = cas.Vector3.from_iterable(pb_T_b @ b_T_a @ a_P_pa)

        dist = pb_V_n @ pb_V_pa

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.buffer_zone_distance - actual_distance

        lower_limit_limited = cas.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
        )

        upper_slack = cas.if_greater(
            actual_distance,
            violated_distance,
            lower_limit_limited + cas.max(0, actual_distance - violated_distance),
            lower_limit_limited,
        )

        # undo factor in A
        upper_slack /= sample_period

        upper_slack = cas.if_greater(
            actual_distance,
            50,  # assuming that distance of unchecked closest points is 100
            cas.Expression(1e4),
            cas.max(0, upper_slack),
        )

        weight = cas.Expression(data=WEIGHT_COLLISION_AVOIDANCE).safe_division(
            cas.min(number_of_self_collisions, self.max_avoided_bodies)
        )
        distance_monitor = Monitor(name=f"collision distance {self.name}", _plot=False)
        distance_monitor.observation_expression = actual_distance > 50
        self.add_monitor(distance_monitor)
        task = Task(name=self.name + "/task", _plot=False)
        self.add_task(task)
        task.plot = False
        task.pause_condition = distance_monitor
        task.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=weight,
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=upper_slack,
        )

    def get_contact_normal_in_b(self):
        return god_map.collision_scene.self_new_b_V_n_symbol(
            self.body_a, self.body_b, self.idx
        )

    def get_position_on_a_in_a(self):
        return god_map.collision_scene.self_new_a_P_pa_symbol(
            self.body_a, self.body_b, self.idx
        )

    def get_b_T_pb(self) -> cas.TransformationMatrix:
        p = god_map.collision_scene.self_new_b_P_pb_symbol(
            self.body_a, self.body_b, self.idx
        )
        return cas.TransformationMatrix.from_xyz_rpy(x=p.x, y=p.y, z=p.z)

    def get_actual_distance(self):
        return god_map.collision_scene.self_contact_distance_symbol(
            self.body_a, self.body_b, self.idx
        )

    def get_number_of_self_collisions(self):
        return god_map.collision_scene.self_number_of_collisions_symbol(
            self.body_a, self.body_b
        )


class CollisionAvoidanceHint(Goal):
    def __init__(
        self,
        tip_link: PrefixedName,
        avoidance_hint: cas.Vector3,
        object_link_name: PrefixedName,
        max_linear_velocity: float = 0.1,
        root_link: Optional[PrefixedName] = None,
        max_threshold: float = 0.05,
        spring_threshold: Optional[float] = None,
        weight: float = WEIGHT_ABOVE_CA,
        name: Optional[str] = None,
    ):
        """
        This goal pushes the link_name in the direction of avoidance_hint, if it is closer than spring_threshold
        to body_b/link_b.
        :param tip_link: str, name of the robot link, has to have a collision body
        :param avoidance_hint: Vector3Stamped as json, direction in which the robot link will get pushed
        :param object_link_name: str, name of the link of the environment object. e.g. fridge handle
        :param max_linear_velocity: float, m/s, default 0.1
        :param root_link: str, default robot root, name of the root link for the kinematic chain
        :param max_threshold: float, default 0.05, distance at which the force has reached weight
        :param spring_threshold: float, default max_threshold, need to be >= than max_threshold weight increases from
                                        sprint_threshold to max_threshold linearly, to smooth motions
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        self.link_name = god_map.world.search_for_link_name(tip_link)
        self.link_b = god_map.world.search_for_link_name(object_link_name)
        if name is None:
            name = f"{self.__class__.__name__}/{self.link_name}/{self.link_b}"
        super().__init__(name=name)
        self.key = (self.link_name, self.link_b)
        self.link_b_hash = self.link_b.__hash__()
        if root_link is None:
            self.root_link = god_map.world.root_link_name
        else:
            self.root_link = root_link

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        god_map.collision_scene.add_collision_check(
            god_map.world.links[self.link_name].name,
            god_map.world.links[self.link_b].name,
            spring_threshold,
        )

        self.avoidance_hint = god_map.world.transform(
            target_frame=self.root_link, spatial_object=avoidance_hint
        )
        self.avoidance_hint.scale(1)

        self.max_velocity = max_linear_velocity
        self.threshold = max_threshold
        self.threshold2 = spring_threshold
        self.weight = weight
        actual_distance = self.get_actual_distance()
        max_velocity = self.max_velocity
        max_threshold = self.threshold
        spring_threshold = self.threshold2
        link_b_hash = self.get_link_b_hash()
        actual_distance_capped = cas.max(actual_distance, 0)

        root_T_a = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.link_name
        )

        spring_error = spring_threshold - actual_distance_capped
        spring_error = cas.max(spring_error, 0)

        spring_weight = cas.if_eq(
            spring_threshold,
            max_threshold,
            cas.Expression(0),
            weight * (spring_error / (spring_threshold - max_threshold)) ** 2,
        )

        weight = cas.if_less_eq(
            actual_distance, max_threshold, cas.Expression(weight), spring_weight
        )
        weight = cas.if_eq(link_b_hash, self.link_b_hash, weight, cas.Expression(0))

        root_V_avoidance_hint = cas.Vector3(self.avoidance_hint)

        # penetration_distance = threshold - actual_distance_capped

        root_P_a = root_T_a.to_position()
        expr = root_V_avoidance_hint.dot(root_P_a)

        # self.add_debug_expr('dist', actual_distance)
        task = Task(name="avoidance_hint")
        self.add_task(task)
        task.add_equality_constraint(
            reference_velocity=max_velocity,
            equality_bound=max_velocity,
            weight=weight,
            task_expression=expr,
        )
        self.observation_expression = cas.TrinaryUnknown

    def get_actual_distance(self):
        return god_map.collision_scene.external_contact_distance_symbol(
            body=self.key[0], body_b=self.key[1]
        )

    def get_link_b_hash(self):
        return god_map.collision_scene.external_link_b_hash_symbol(
            body=self.key[0], body_b=self.key[1]
        )


# use cases
# avoid all
# allow all
# avoid all then allow something
# avoid only something


@validated_dataclass
class CollisionAvoidance(Goal):
    collision_entries: List[CollisionViewRequest] = field(default_factory=list)

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__
        god_map.collision_scene.matrix_manager.parse_collision_requests(
            self.collision_entries
        )
        self.collision_entries = (
            god_map.collision_scene.matrix_manager.collision_requests
        )
        if (
            not self.collision_entries
            or not self.collision_entries[-1].is_allow_all_collision()
        ):
            self.add_external_collision_avoidance_constraints()
        if not self.collision_entries or (
            not self.collision_entries[-1].is_allow_all_collision()
            and not self.collision_entries[-1].is_allow_all_self_collision()
        ):
            self.add_self_collision_avoidance_constraints()
        # if not cas.is_true_symbol(start_condition):
        #     payload_monitor = CollisionMatrixUpdater(name=f'{self.name}/update collision matrix',
        #                                              start_condition=start_condition,
        #                                              new_collision_matrix=self.collision_matrix)
        #     god_map.motion_statechart_manager.add_monitor(payload_monitor)
        # else:
        collision_matrix = (
            god_map.collision_scene.matrix_manager.compute_collision_matrix()
        )
        god_map.collision_scene.set_collision_matrix(collision_matrix)

    def _task_sanity_check(self):
        pass

    @profile
    def add_external_collision_avoidance_constraints(self):
        robot: AbstractRobot
        # thresholds = god_map.collision_scene.matrix_manager.external_thresholds
        for robot in god_map.world.get_views_by_type(AbstractRobot):
            for connection in robot.controlled_connections:
                if connection.frozen_for_collision_avoidance:
                    continue
                bodies = god_map.world.get_direct_child_bodies_with_collision(
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
                for idx in range(max_avoided_bodies):
                    self.add_goal(
                        ExternalCA(
                            connection=connection,
                            world=god_map.world,
                            name_prefix=self.name,
                            idx=idx,
                            max_avoided_bodies=max_avoided_bodies,
                        )
                    )

        # get_middleware().loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter: Dict[Tuple[Body, Body], float] = defaultdict(float)
        num_constr = 0
        robot: AbstractRobot
        # collect bodies from the same connection to the main body pair
        for robot in god_map.world.get_views_by_type(AbstractRobot):
            for body_a_original in robot.bodies_with_enabled_collision:
                for body_b_original in robot.bodies_with_enabled_collision:
                    if (
                        body_a_original,
                        body_b_original,
                    ) in god_map.world.disabled_collision_pairs or (
                        body_b_original,
                        body_a_original,
                    ) in god_map.world.disabled_collision_pairs:
                        continue
                    body_a, body_b = (
                        god_map.world.compute_chain_reduced_to_controlled_joints(
                            body_a_original, body_b_original
                        )
                    )
                    if body_b.name < body_a.name:
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
            ca_goal = SelfCA(
                body_a=link_a,
                body_b=link_b,
                world=god_map.world,
                name_prefix=self.name,
                idx=0,
                max_avoided_bodies=1,
                buffer_zone_distance=counter[link_a, link_b],
            )
            self.add_goal(ca_goal)
            num_constr += 1
        get_middleware().loginfo(
            f"Adding {num_constr} self collision avoidance constraints."
        )
