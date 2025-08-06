from __future__ import annotations

import abc
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
from giskardpy.model.collision_matrix_manager import CollisionCheck
from giskardpy.qp.free_variable import FreeVariable
from semantic_world.connections import ActiveConnection
from semantic_world.robots import AbstractRobot
from semantic_world.spatial_types.derivatives import Derivatives
from semantic_world.spatial_types.symbol_manager import symbol_manager
from semantic_world.utils import copy_lru_cache
from semantic_world.world_entity import Body, Connection


@dataclass
class Collision:
    contact_distance: float
    link_a: Body = field(default=None)
    link_b: Body = field(default=None)
    original_link_a: Body = field(init=False)
    original_link_b: Body = field(init=False)
    map_P_pa: np.ndarray = field(default=None)
    map_P_pb: np.ndarray = field(default=None)
    map_V_n_input: np.ndarray = field(default=None)
    a_P_pa: np.ndarray = field(default=None)
    b_P_pb: np.ndarray = field(default=None)
    data: np.ndarray = field(init=False)
    is_external: bool = False

    _hash_idx: int = 0
    _map_V_n_idx: int = 1
    _map_V_n_slice: slice = slice(1, 4)

    _contact_distance_idx: int = 4
    _new_a_P_pa_idx: int = 5
    _new_a_P_pa_slice: slice = slice(5, 8)

    _new_b_V_n_idx: int = 8
    _new_b_V_n_slice: slice = slice(8, 11)
    _new_b_P_pb_idx: int = 11
    _new_b_P_pb_slice: slice = slice(11, 14)

    _self_data_slice: slice = slice(4, 14)
    _external_data_slice: slice = slice(0, 8)

    @profile
    def __post_init__(self):
        self.original_link_a = self.link_a
        self.original_link_b = self.link_b

        self.data = np.array([
            self.link_b.__hash__(),  # hash
            0, 0, 1,  # map_V_n

            self.contact_distance,
            0, 0, 0,  # new_a_P_pa

            0, 0, 1,  # new_b_V_n
            0, 0, 0,  # new_b_P_pb
        ],
            dtype=float)
        if self.map_V_n_input is not None:
            self.map_V_n = self.map_V_n_input

    @property
    def external_data(self) -> np.ndarray:
        return self.data[:self._new_b_V_n_idx]

    @property
    def self_data(self) -> np.ndarray:
        return self.data[self._self_data_slice]

    @property
    def external_and_self_data(self) -> np.ndarray:
        return self.data[self._external_data_slice]

    @property
    def contact_distance(self) -> float:
        return self.data[self._contact_distance_idx]

    @contact_distance.setter
    def contact_distance(self, value: float):
        self.data[self._contact_distance_idx] = value

    @property
    def link_b_hash(self) -> float:
        return self.data[self._hash_idx]

    @property
    def map_V_n(self) -> np.ndarray:
        a = self.data[self._map_V_n_slice]
        return np.array([a[0], a[1], a[2], 0])

    @map_V_n.setter
    def map_V_n(self, value: np.ndarray):
        self.data[self._map_V_n_slice] = value[:3]

    @property
    def new_a_P_pa(self):
        a = self.data[self._new_a_P_pa_slice]
        return np.array([a[0], a[1], a[2], 1])

    @new_a_P_pa.setter
    def new_a_P_pa(self, value: np.ndarray):
        self.data[self._new_a_P_pa_slice] = value[:3]

    @property
    def new_b_P_pb(self):
        a = self.data[self._new_b_P_pb_slice]
        return np.array([a[0], a[1], a[2], 1])

    @new_b_P_pb.setter
    def new_b_P_pb(self, value: np.ndarray):
        self.data[self._new_b_P_pb_slice] = value[:3]

    @property
    def new_b_V_n(self):
        a = self.data[self._new_b_V_n_slice]
        return np.array([a[0], a[1], a[2], 0])

    @new_b_V_n.setter
    def new_b_V_n(self, value: np.ndarray):
        self.data[self._new_b_V_n_slice] = value[:3]

    def __str__(self):
        return f'{self.original_link_a}|-|{self.original_link_b}: {self.contact_distance}'

    def __repr__(self):
        return str(self)

    def reverse(self):
        return Collision(link_a=self.original_link_b,
                         link_b=self.original_link_a,
                         map_P_pa=self.map_P_pb,
                         map_P_pb=self.map_P_pa,
                         map_V_n_input=-self.map_V_n,
                         a_P_pa=self.b_P_pb,
                         b_P_pb=self.a_P_pa,
                         contact_distance=self.contact_distance)


@dataclass
class SortedCollisionResults:
    data: List[Collision] = field(default_factory=list)
    default_result: Collision = Collision(contact_distance=100)

    def _sort(self, x: Collision):
        return x.contact_distance

    def add(self, element: Collision):
        self.data.append(element)
        self.data = list(sorted(self.data, key=self._sort))

    def __getitem__(self, item: int) -> Collision:
        try:
            return self.data[item]
        except (KeyError, IndexError) as e:
            return SortedCollisionResults.default_result


@dataclass
class Collisions:
    collision_list_size: int
    all_collisions: Set[Collision]
    fixed_joints: List[Connection] = field(default_factory=list)
    self_collisions: Dict[Tuple[Body, Body], SortedCollisionResults] = field(
        default_factory=lambda: defaultdict(SortedCollisionResults))
    external_collisions: Dict[Body, SortedCollisionResults] = field(
        default_factory=lambda: defaultdict(SortedCollisionResults))
    external_collision_long_key: Dict[Tuple[Body, Body], Collision] = field(
        default_factory=lambda: defaultdict(lambda: SortedCollisionResults.default_result))
    all_collisions: Set[Collision] = field(default_factory=set)
    number_of_self_collisions: Dict[Tuple[Body, Body], int] = field(default_factory=lambda: defaultdict(int))
    number_of_external_collisions: Dict[Body, int] = field(default_factory=lambda: defaultdict(int))

    @profile
    def __post_init__(self):
        self.fixed_joints = god_map.collision_scene.fixed_joints

    def get_robot_from_self_collision(self, collision):
        link_a, link_b = collision.link_a, collision.link_b
        for robot in god_map.collision_scene.robots:
            if link_a in robot.link_names_as_set and link_b in robot.link_names_as_set:
                return robot

    @profile
    def add(self, collision: Collision):
        robot = self.get_robot_from_self_collision(collision)
        collision.is_external = robot is None
        if collision.is_external:
            collision = self.transform_external_collision(collision)
            key = collision.link_a
            self.external_collisions[key].add(collision)
            self.number_of_external_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_external_collisions[key] + 1)
            key_long = (collision.original_link_a, collision.original_link_b)
            if key_long not in self.external_collision_long_key:
                self.external_collision_long_key[key_long] = collision
            else:
                self.external_collision_long_key[key_long] = min(collision, self.external_collision_long_key[key_long],
                                                                 key=lambda x: x.contact_distance)
        else:
            collision = self.transform_self_collision(collision, robot)
            key = collision.link_a, collision.link_b
            self.self_collisions[key].add(collision)
            try:
                self.number_of_self_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_self_collisions[key] + 1)
            except Exception as e:
                pass
        self.all_collisions.add(collision)

    @profile
    def transform_self_collision(self, collision: Collision, robot: AbstractRobot) -> Collision:
        link_a = collision.original_link_a
        link_b = collision.original_link_b
        new_link_a, new_link_b = god_map.world.compute_chain_reduced_to_controlled_joints(link_a, link_b,
                                                                                          self.fixed_joints)
        if not god_map.world.link_order(new_link_a, new_link_b):
            collision = collision.reverse()
            new_link_a, new_link_b = new_link_b, new_link_a
        collision.link_a = new_link_a
        collision.link_b = new_link_b

        new_b_T_r = god_map.world.compute_fk_np(new_link_b, robot.root)
        root_T_map = god_map.world.compute_fk_np(robot.root, god_map.world.root_link_name)
        new_b_T_map = new_b_T_r @ root_T_map
        collision.new_b_V_n = new_b_T_map @ collision.map_V_n

        if collision.map_P_pa is not None:
            new_a_T_r = god_map.world.compute_fk_np(new_link_a, robot.root)
            collision.new_a_P_pa = new_a_T_r @ root_T_map @ collision.map_P_pa
            collision.new_b_P_pb = new_b_T_map @ collision.map_P_pb
        else:
            new_a_T_a = god_map.world.compute_fk_np(new_link_a, collision.original_link_a)
            collision.new_a_P_pa = new_a_T_a @ collision.a_P_pa
            new_b_T_b = god_map.world.compute_fk_np(new_link_b, collision.original_link_b)
            collision.new_b_P_pb = new_b_T_b @ collision.b_P_pb
        return collision

    @profile
    def transform_external_collision(self, collision: Collision) -> Collision:
        link_name = collision.original_link_a
        joint = god_map.world.links[link_name].parent_joint_name

        def stopper(joint_name):
            return god_map.world.is_joint_controlled(joint_name) and joint_name not in self.fixed_joints

        try:
            movable_joint = god_map.world.search_for_parent_joint(joint, stopper)
        except KeyError as e:
            movable_joint = joint
        new_a = god_map.world.joints[movable_joint].child_link_name
        collision.link_a = new_a
        if collision.map_P_pa is not None:
            new_a_T_map = god_map.world.compute_fk_np(new_a, god_map.world.root_link_name)
            collision.new_a_P_pa = new_a_T_map @ collision.map_P_pa
        else:
            new_a_T_a = god_map.world.compute_fk_np(new_a, collision.original_link_a)
            collision.new_a_P_pa = new_a_T_a @ collision.a_P_pa

        return collision

    @profile
    def get_external_collisions(self, link_name: Body) -> SortedCollisionResults:
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        """
        if link_name in self.external_collisions:
            return self.external_collisions[link_name]
        return SortedCollisionResults()

    def get_external_collisions_long_key(self, link_a: Body, link_b: Body) -> Collision:
        return self.external_collision_long_key[link_a, link_b]

    @profile
    def get_number_of_external_collisions(self, joint_name: Body) -> int:
        return self.number_of_external_collisions[joint_name]

    def get_self_collisions(self, link_a: Body, link_b: Body) -> SortedCollisionResults:
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        """
        if (link_a, link_b) in self.self_collisions:
            return self.self_collisions[link_a, link_b]
        return SortedCollisionResults()

    def get_number_of_self_collisions(self, link_a, link_b):
        return self.number_of_self_collisions[link_a, link_b]

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collisions


class CollisionDetector(abc.ABC):
    """
    Abstract class for collision detectors.
    """

    @abc.abstractmethod
    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """

    @abc.abstractmethod
    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """

    @abc.abstractmethod
    def check_collisions(self,
                         collision_matrix: Set[CollisionCheck],
                         buffer: float = 0.05) -> Collisions:
        pass


class NullCollisionDetector(CollisionDetector):
    def sync_world_model(self) -> None:
        pass

    def sync_world_state(self) -> None:
        pass

    def check_collisions(self, collision_matrix: Set[CollisionCheck], buffer: float = 0.05) -> Collisions:
        return Collisions()
