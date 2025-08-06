from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from itertools import product, combinations_with_replacement
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict, Callable, TYPE_CHECKING

import numpy as np
from line_profiler import profile
from lxml import etree

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import UnknownGroupException, UnknownLinkException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_data_synchronizer import CollisionDataSynchronizer
from giskardpy.model.collision_detector import CollisionDetector
from giskardpy.model.collision_matrix_manager import CollisionMatrixManager
from giskardpy.qp.free_variable import FreeVariable
from semantic_world.connections import ActiveConnection
from semantic_world.robots import AbstractRobot
from semantic_world.spatial_types.derivatives import Derivatives
from semantic_world.spatial_types.symbol_manager import symbol_manager
from semantic_world.utils import copy_lru_cache
from semantic_world.world_entity import Body, Connection

np.random.seed(1337)


class CollisionCheckerLib(Enum):
    none = -1
    bpb = 1


@dataclass
class CollisionAvoidanceThresholds:
    number_of_repeller: int = 1
    soft_threshold: float = 0.05
    hard_threshold: float = 0.0
    max_velocity: float = 0.2

    @classmethod
    def init_50mm(cls):
        return cls(soft_threshold=0.05, hard_threshold=0.0)

    @classmethod
    def init_100mm(cls):
        return cls(soft_threshold=0.1, hard_threshold=0.0)

    @classmethod
    def init_25mm(cls):
        return cls(soft_threshold=0.025, hard_threshold=0.0)


@dataclass
class CollisionAvoidanceGroupThresholds:
    robot: AbstractRobot
    external_collision_avoidance: DefaultDict[Connection, CollisionAvoidanceThresholds] = field(
        default_factory=lambda: defaultdict(CollisionAvoidanceThresholds))
    self_collision_avoidance: DefaultDict[Body, CollisionAvoidanceThresholds] = field(
        default_factory=lambda: defaultdict(CollisionAvoidanceThresholds))

    def max_num_of_repeller(self):
        external_distances = self.external_collision_avoidance
        self_distances = self.self_collision_avoidance
        default_distance = max(external_distances.default_factory().number_of_repeller,
                               self_distances.default_factory().number_of_repeller)
        for value in external_distances.values():
            default_distance = max(default_distance, value.number_of_repeller)
        for value in self_distances.values():
            default_distance = max(default_distance, value.number_of_repeller)
        return default_distance


@dataclass
class CollisionWorldSynchronizer:
    collision_detector: CollisionDetector = None
    matrix_manager: CollisionMatrixManager = field(default_factory=CollisionMatrixManager)
    data_synchronizer: CollisionDataSynchronizer = field(default_factory=CollisionDataSynchronizer)

    external_monitored_links: Dict[Body, int] = field(default_factory=dict)
    self_monitored_links: Dict[Body, int] = field(default_factory=dict)
    world_model_version: int = -1

    # config: CollisionAvoidanceConfig
    # self_collision_matrices: List[SelfCollisionMatrix] = field(default_factory=list)
    # collision_avoidance_configs: List[CollisionAvoidanceGroupThresholds] = field(
    #     default_factory=CollisionAvoidanceGroupThresholds)
    # collision_checker_id: CollisionCheckerLib = CollisionCheckerLib.none
    # _fixed_joints: Tuple[Connection, ...] = field(default_factory=tuple)
    #
    # closest_points: Collisions = field(default_factory=Collisions)
    # external_monitored_links: Dict[Body, int] = field(default_factory=dict)
    # self_monitored_links: Dict[Tuple[Body, Body], int] = field(default_factory=dict)
    # world_version: int = -1
    # collision_matrix: dict = field(default_factory=dict)
    # external_collision_data: np.ndarray = field(default_factory=lambda: np.zeros(1))
    # self_collision_data: np.ndarray = field(default_factory=lambda: np.zeros(1))

    # def clear_collision_matrix(self):
    #     self.collision_matrix = {}

    # @property
    # def disabled_links(self) -> Set[Body]:
    #     disabled_links = set()
    #     for matrix in self.self_collision_matrices:
    #         disabled_links.update(matrix.disabled_bodies)
    #     return disabled_links

    # @property
    # def robots(self) -> List[AbstractRobot]:
    #     return [view for view in god_map.world.views if isinstance(view, AbstractRobot)]
    #
    # @property
    # def robot_names(self):
    #     return [r.name for r in self.robots]

    # def blacklist_inter_group_collisions(self) -> None:
    #     for group_a_name, group_b_name in combinations(god_map.world.minimal_group_names, 2):
    #         one_group_is_robot = group_a_name in self.robot_names or group_b_name in self.robot_names
    #         if one_group_is_robot:
    #             if group_a_name in self.robot_names:
    #                 robot_group = god_map.world.groups[group_a_name]
    #                 other_group = god_map.world.groups[group_b_name]
    #             else:
    #                 robot_group = god_map.world.groups[group_b_name]
    #                 other_group = god_map.world.groups[group_a_name]
    #             unmovable_links = robot_group.get_unmovable_links()
    #             if len(unmovable_links) > 0:  # ignore collisions between unmovable links of the robot and the env
    #                 for link_a, link_b in product(unmovable_links,
    #                                               other_group.link_names_with_collisions):
    #                     self.self_collision_matrix[
    #                         god_map.world.sort_links(link_a, link_b)] = DisableCollisionReason.Unknown
    #             continue
    #         # disable all collisions of groups that aren't a robot
    #         group_a: AbstractRobot = god_map.world.get_view_by_name(group_a_name)
    #         group_b: AbstractRobot = god_map.world.get_view_by_name(group_b_name)
    #         for link_a, link_b in product(group_a.bodies_with_collisions, group_b.bodies_with_collisions):
    #             self.self_collision_matrix[god_map.world.sort_links(link_a, link_b)] = DisableCollisionReason.Unknown
    #     # disable non actuated groups
    #     for group in god_map.world.groups.values():
    #         if group.name not in self.robot_names:
    #             for link_a, link_b in set(combinations_with_replacement(group.link_names_with_collisions, 2)):
    #                 key = god_map.world.sort_links(link_a, link_b)
    #                 self.self_collision_matrix[key] = DisableCollisionReason.Unknown

    def sync(self):
        if self.has_world_model_changed():
            self.collision_detector.sync_world_model()
            self.matrix_manager.apply_world_model_updates()
        self.collision_detector.sync_world_state()

    def has_world_model_changed(self) -> bool:
        if self.world_model_version != god_map.world._model_version:
            self.world_model_version = god_map.world._model_version
            return True
        return False

    # def add_added_checks(self):
    #     try:
    #         added_checks = god_map.added_collision_checks
    #         god_map.added_collision_checks = {}
    #     except AttributeError:
    #         # no collision checks added
    #         added_checks = {}
    #     for key, distance in added_checks.items():
    #         if key in self.collision_matrix:
    #             self.collision_matrix[key] = max(distance, self.collision_matrix[key])
    #         else:
    #             self.collision_matrix[key] = distance

    # def reset_cache(self):
    #     """
    #     Called when model changes.
    #     :return:
    #     """
    #     self.external_monitored_links = {}
    #     self.self_monitored_links = {}

    # def get_map_T_geometry(self, body: Body, collision_id: int = 0) -> np.ndarray:
    #     return god_map.world.compute_fk_with_collision_offset_np(god_map.world.root_link_name, body, collision_id)

    # %% external collision symbols
    def monitor_link_for_external(self, body: Body, idx: int):
        self.external_monitored_links[body] = max(idx, self.external_monitored_links.get(body, 0))

    def get_external_collision_symbol(self) -> List[cas.Symbol]:
        symbols = []
        for body, max_idx in self.external_monitored_links.items():
            for idx in range(max_idx + 1):
                symbols.append(self.external_link_b_hash_symbol(body, idx))

                v = self.external_map_V_n_symbol(body, idx)
                symbols.extend([v.x.free_symbols()[0], v.y.free_symbols()[0], v.z.free_symbols()[0]])

                symbols.append(self.external_contact_distance_symbol(body, idx))

                p = self.external_new_a_P_pa_symbol(body, idx)
                symbols.extend([p.x.free_symbols()[0], p.y.free_symbols()[0], p.z.free_symbols()[0]])

            symbols.append(self.external_number_of_collisions_symbol(body))
        if len(symbols) != self.external_collision_data.shape[0]:
            self.data_synchronizer.external_collision_data = np.zeros(len(symbols), dtype=float)
        return symbols

    def external_map_V_n_symbol(self, body: Body, idx: int) -> cas.Vector3:
        provider = lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[i].map_V_n
        return symbol_manager.register_vector3(name=f'closest_point({body.name})[{idx}].map_V_n',
                                               provider=provider)

    def external_new_a_P_pa_symbol(self, body: Body, idx: int) -> cas.Point3:
        provider = lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[i].new_a_P_pa
        return symbol_manager.register_point3(name=f'closest_point({body.name})[{idx}].new_a_P_pa',
                                              provider=provider)

    def external_contact_distance_symbol(self,
                                         body: Body,
                                         idx: Optional[int] = None,
                                         body_b: Optional[Body] = None) -> cas.Symbol:
        if body_b is None:
            assert idx is not None
            provider = lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[i].contact_distance
            return symbol_manager.register_symbol_provider(name=f'closest_point({body.name})[{idx}].contact_distance',
                                                           provider=provider)
        assert body_b is not None
        provider = lambda l1=body, l2=body_b: (
            self.closest_points.get_external_collisions_long_key(l1, l2).contact_distance)
        return symbol_manager.register_symbol_provider(
            name=f'closest_point({body.name}, {body_b.name}).contact_distance',
            provider=provider)

    def external_link_b_hash_symbol(self,
                                    body: Body,
                                    idx: Optional[int] = None,
                                    body_b: Optional[Body] = None) -> cas.Symbol:
        if body_b is None:
            assert idx is not None
            provider = lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[i].link_b_hash
            return symbol_manager.register_symbol_provider(name=f'closest_point({body.name})[{idx}].link_b_hash',
                                                           provider=provider)
        assert body_b is not None
        provider = lambda l1=body, l2=body_b: (
            self.closest_points.get_external_collisions_long_key(l1, l2).link_b_hash)
        return symbol_manager.register_symbol_provider(
            name=f'closest_point({body.name}, {body_b.name}).link_b_hash',
            provider=provider
        )

    def external_number_of_collisions_symbol(self, body: Body) -> cas.Symbol:
        provider = lambda n=body: self.closest_points.get_number_of_external_collisions(n)
        return symbol_manager.register_symbol_provider(name=f'len(closest_point({body.name}))',
                                                       provider=provider)

    # %% self collision symbols
    def monitor_link_for_self(self, link_a: Body, link_b: Body, idx: int):
        self.self_monitored_links[link_a, link_b] = max(idx, self.self_monitored_links.get((link_a, link_b), 0))

    def get_self_collision_symbol(self) -> List[cas.Symbol]:
        symbols = []
        for (link_a, link_b), max_idx in self.self_monitored_links.items():
            for idx in range(max_idx + 1):
                symbols.append(self.self_contact_distance_symbol(link_a, link_b, idx))

                p = self.self_new_a_P_pa_symbol(link_a, link_b, idx)
                symbols.extend([p.x.free_symbols()[0], p.y.free_symbols()[0], p.z.free_symbols()[0]])

                v = self.self_new_b_V_n_symbol(link_a, link_b, idx)
                symbols.extend([v.x.free_symbols()[0], v.y.free_symbols()[0], v.z.free_symbols()[0]])

                p = self.self_new_b_P_pb_symbol(link_a, link_b, idx)
                symbols.extend([p.x.free_symbols()[0], p.y.free_symbols()[0], p.z.free_symbols()[0]])

            symbols.append(self.self_number_of_collisions_symbol(link_a, link_b))
        if len(symbols) != self.self_collision_data.shape[0]:
            self.data_synchronizer.self_collision_data = np.zeros(len(symbols), dtype=float)
        return symbols

    def self_new_b_V_n_symbol(self, link_a: Body, link_b: Body, idx: int) -> cas.Vector3:
        provider = lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(a, b)[i].new_b_V_n
        return symbol_manager.register_vector3(name=f'closest_point({link_a.name}, {link_b.name})[{idx}].new_b_V_n',
                                               provider=provider)

    def self_new_a_P_pa_symbol(self, link_a: Body, link_b: Body, idx: int) -> cas.Point3:
        provider = lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(a, b)[i].new_a_P_pa
        return symbol_manager.register_point3(name=f'closest_point({link_a.name}, {link_b.name}).new_a_P_pa',
                                              provider=provider)

    def self_new_b_P_pb_symbol(self, link_a: Body, link_b: Body, idx: int) -> cas.Point3:
        provider = lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(a, b)[i].new_b_P_pb
        p = symbol_manager.register_point3(name=f'closest_point({link_a.name}, {link_b.name}).new_b_P_pb',
                                           provider=provider)
        return p

    def self_contact_distance_symbol(self, link_a: Body, link_b: Body, idx: int) -> cas.Symbol:
        provider = lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(a, b)[i].contact_distance
        return symbol_manager.register_symbol_provider(
            name=f'closest_point({link_a.name}, {link_b.name}).contact_distance',
            provider=provider)

    def self_number_of_collisions_symbol(self, link_a: Body, link_b: Body) -> cas.Symbol:
        provider = lambda a=link_a, b=link_b: self.closest_points.get_number_of_self_collisions(a, b)
        return symbol_manager.register_symbol_provider(name=f'len(closest_point({link_a.name}, {link_b.name}))',
                                                       provider=provider)
