from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import product, combinations_with_replacement, chain
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict, Callable

import numpy as np
from line_profiler import profile

from giskardpy.data_types.key_default_dict import KeyDefaultDict
from giskardpy.god_map import god_map
from giskardpy.qp.free_variable import FreeVariable
from semantic_digital_twin.collision_checking.collision_detector import CollisionCheck
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class CollisionViewRequest:
    AVOID_COLLISION: int = field(default=0, init=False)
    ALLOW_COLLISION: int = field(default=1, init=False)

    type_: Optional[int] = None
    distance: Optional[float] = None
    view1: Optional[AbstractRobot] = None
    view2: Optional[AbstractRobot] = None

    def __post_init__(self):
        if self.type_ is None:
            self.type_ = self.AVOID_COLLISION
        if self.view1 is None and self.view2 is not None:
            self.view1, self.view2 = self.view2, self.view1
        if self.distance is not None and self.distance < 0:
            raise ValueError(f"Distance must be positive or None, got {self.distance}")
        if self.type_ not in [self.AVOID_COLLISION, self.ALLOW_COLLISION]:
            raise ValueError(f"Unknown type {self.type_}")

    def is_distance_set(self) -> bool:
        return self.distance is not None

    def any_view1(self):
        return self.view1 is None

    def any_view2(self):
        return self.view2 is None

    def is_avoid_collision(self) -> bool:
        return self.type_ == self.AVOID_COLLISION

    def is_allow_collision(self) -> bool:
        return self.type_ == self.ALLOW_COLLISION

    def is_avoid_all_self_collision(self) -> bool:
        return self.is_avoid_collision() and self.view1 == self.view2

    def is_allow_all_self_collision(self) -> bool:
        return self.is_allow_collision() and self.view1 == self.view2

    def is_avoid_all_collision(self) -> bool:
        return self.is_avoid_collision() and self.any_view1() and self.any_view2()

    def is_allow_all_collision(self) -> bool:
        return self.is_allow_collision() and self.any_view1() and self.any_view2()


class DisableCollisionReason(Enum):
    Unknown = -1
    Never = 1
    Adjacent = 2
    Default = 3
    AlmostAlways = 4


@dataclass
class CollisionMatrixManager:
    """
    Handles all matrix related operations for multiple robots.
    """

    world: World
    robots: Set[AbstractRobot]

    added_checks: Set[CollisionCheck] = field(default_factory=set)

    collision_requests: List[CollisionViewRequest] = field(default_factory=list)
    """
    Motion goal specific requests for collision avoidance checks.
    Can overwrite the thresholds and add additional disabled bodies/pairs.
    """

    # external_thresholds: Dict[Body, CollisionAvoidanceThreshold] = field(init=False)
    # self_thresholds: Dict[Tuple[Body, Body], CollisionAvoidanceThreshold] = field(init=False)
    # """
    # Thresholds used for collision avoidance checks.
    # They already include collision requests
    # """

    # disabled_bodies: Set[Body] = field(default_factory=set)
    # disabled_pairs: Set[Tuple[Body, Body]] = field(default_factory=set)
    # """
    # Bodies disables for collision avoidance checks.
    # Include collision configs for the robot and collision requests
    # """

    # def __post_init__(self):
    #     self.compute_thresholds()
    #     self.combine_collision_configs()

    # def compute_thresholds(self):
    #     def get_external_threshold(body: Body) -> CollisionAvoidanceThreshold:
    #         for robot in self.robots:
    #             # if body is part of the robot, use robot threshold
    #             if body in robot.bodies_with_collisions:
    #                 return robot.collision_config.external_avoidance_threshold[body]
    #         raise KeyError(f'Could not find collision avoidance threshold for body "{body.name}". '
    #                        f'Body is not part of any robot.')
    #
    #     self.external_thresholds = KeyDefaultDict(get_external_threshold)
    #
    #     def get_self_threshold(key: Tuple[Body, Body]) -> CollisionAvoidanceThreshold:
    #         body_a, body_b = key
    #         soft_a = body_a._collision_config.buffer_zone_distance
    #         soft_b = body_b._collision_config.buffer_zone_distance
    #         if soft_a is None and soft_b is None:
    #             raise ValueError(f'Could not find collision avoidance threshold for body pair '
    #                              f'"{body_a.name}", "{body_b.name}". '
    #                              f'Neither body has a soft threshold set.')
    #         if soft_a is None:
    #             soft = soft_b
    #         elif soft_b is None:
    #             soft = soft_a
    #         else:
    #             soft = max(soft_a, soft_b)
    #         return CollisionAvoidanceThreshold(
    #             soft_threshold=soft,
    #             hard_threshold=max(body_a._collision_config.violated_distance,
    #                                body_b._collision_config.violated_distance))
    #
    #     self.self_thresholds = KeyDefaultDict(get_self_threshold)

    # def combine_collision_configs(self):
    #     for robot in self.robots:
    #         self.disabled_pairs.update(robot.collision_config.disabled_pairs)
    #         self.disabled_bodies.update(robot.collision_config.disabled_bodies)

    def compute_collision_matrix(self) -> Set[CollisionCheck]:
        """
        Parses the collision requrests and (temporary) collision configs in the world
        to create a set of collision checks.
        """
        collision_matrix: Set[CollisionCheck] = set()
        for collision_request in self.collision_requests:
            if collision_request.any_view1():
                view_1_bodies = self.world.bodies_with_enabled_collision
            else:
                view_1_bodies = collision_request.view1.bodies_with_enabled_collision
            if collision_request.any_view2():
                view2_bodies = self.world.bodies_with_enabled_collision
            else:
                view2_bodies = collision_request.view2.bodies_with_enabled_collision
            disabled_pairs = self.world.disabled_collision_pairs
            for body1 in view_1_bodies:
                for body2 in view2_bodies:
                    collision_check = CollisionCheck(
                        body_a=body1, body_b=body2, distance=0, _world=self.world
                    )
                    (robot_body, env_body) = collision_check.bodies()
                    if (robot_body, env_body) in disabled_pairs:
                        continue
                    if collision_request.distance is None:
                        distance = max(
                            robot_body.get_collision_config().buffer_zone_distance
                            or 0.0,
                            env_body.get_collision_config().buffer_zone_distance or 0.0,
                        )
                    else:
                        distance = collision_request.distance
                    if not collision_request.is_allow_collision():
                        collision_check.distance = distance
                        collision_check._validate()
                    if collision_request.is_allow_collision():
                        if collision_check in collision_matrix:
                            collision_matrix.remove(collision_check)
                    if collision_request.is_avoid_collision():
                        if collision_request.is_distance_set():
                            collision_matrix.add(collision_check)
                        else:
                            collision_matrix.add(collision_check)
        return collision_matrix

    # def create_default_thresholds(self):
    #     max_distances = {}
    #     for robot_name in self.robot_names:
    #         collision_avoidance_config = god_map.collision_scene.collision_avoidance_configs[robot_name]
    #         external_distances = collision_avoidance_config.external_collision_avoidance
    #         self_distances = collision_avoidance_config.self_collision_avoidance
    #
    #         # override max distances based on external distances dict
    #         for robot in god_map.collision_scene.robots:
    #             for body in robot.bodies_with_collisions:
    #                 try:
    #                     controlled_parent_joint = god_map.world.get_controlled_parent_joint_of_link(body)
    #                 except KeyError as e:
    #                     continue  # this happens when the root link of a robot has a collision model
    #                 distance = external_distances[controlled_parent_joint].buffer_zone_distance
    #                 for child_link_name in god_map.world.get_directly_controlled_child_links_with_collisions(
    #                         controlled_parent_joint):
    #                     max_distances[child_link_name] = distance
    #
    #         for link_name in self_distances:
    #             distance = self_distances[link_name].buffer_zone_distance
    #             if link_name in max_distances:
    #                 max_distances[link_name] = max(distance, max_distances[link_name])
    #             else:
    #                 max_distances[link_name] = distance
    #
    #     return max_distances

    def add_collision_check(self, body_a: Body, body_b: Body, distance: float):
        """
        Tell Giskard to check this collision, even if it got disabled through other means such as allow_all_collisions.
        """
        check = CollisionCheck.create_and_validate(
            body_a=body_a, body_b=body_b, distance=distance, world=god_map.world
        )
        if check in self.added_checks:
            raise ValueError(f"Collision check {check} already added")
        self.added_checks.add(check)

    def parse_collision_requests(
        self, collision_goals: List[CollisionViewRequest]
    ) -> None:
        """
        Resolve an incoming list of collision goals into collision checks.
        1. remove redundancy
        2. remove entries where view1 or view2 are none
        :param collision_goals:
        :return:
        """
        for i, collision_goal in enumerate(reversed(collision_goals)):
            if collision_goal.is_avoid_all_collision():
                # remove everything before the avoid all
                collision_goals = collision_goals[len(collision_goals) - i - 1 :]
                break
            if collision_goal.is_allow_all_collision():
                # remove everything before the allow all, including the allow all
                collision_goals = collision_goals[len(collision_goals) - i :]
                break
        else:
            # put an avoid all at the front
            collision_goal = CollisionViewRequest()
            collision_goal.type_ = CollisionViewRequest.AVOID_COLLISION
            collision_goal.distance = None
            collision_goals.insert(0, collision_goal)

        self.collision_requests = list(collision_goals)

    def apply_world_model_updates(self) -> None:
        # self.update_self_collision_matrices_for_attached_bodies()
        # self.disable_non_robot_collisions()
        # self.disable_env_with_unmovable_bodies()
        pass

    # def update_self_collision_matrices_for_attached_bodies(self):
    #     for robot in self.robots:
    #         attached_links = [link for link in robot.bodies_with_collisions
    #                           if (link, link) not in robot.collision_config.disabled_pairs]
    #         body_combinations = set(product(attached_links, robot.bodies_with_collisions))
    #         scm = SelfCollisionMatrix(collision_config=robot.collision_config)
    #         if body_combinations:
    #             disabled_pairs = scm.compute_self_collision_matrix(body_combinations=body_combinations)
    #             robot.collision_config.disabled_pairs.update(disabled_pairs)

    def get_non_robot_bodies(self) -> Set[Body]:
        return set(self.world.bodies_with_enabled_collision).difference(
            self.get_robot_bodies()
        )

    def get_robot_bodies(self) -> Set[Body]:
        robot_bodies = set()
        for robot in self.robots:
            robot_bodies.update(robot.bodies_with_enabled_collision)
        return robot_bodies

    # def disable_env_with_unmovable_bodies(self) -> None:
    #     for robot in self.robots:
    #         for body_a in robot.unmovable_bodies_with_collision:
    #             for body_b in self.get_non_robot_bodies():
    #                 body_a, body_b = sort_bodies(body_a, body_b)
    #                 self.disabled_pairs.add((body_a, body_b))

    # def disable_non
    #     for group in god_map.world.groups.values():
    #         if group.name not in self.robot_names:
    #             for link_a, link_b in set(combinations_with_replacement(group.link_names_with_collisions, 2)):
    #                 key = god_map.world.sort_links(link_a, link_b)
    #                 self.self_collision_matrix[key] = DisableCollisionReason.Unknown
