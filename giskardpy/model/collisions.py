from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from line_profiler import profile

from giskardpy.god_map import god_map
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionDetector,
    CollisionCheck,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(unsafe_hash=True)
class GiskardCollision:
    contact_distance_input: float = 0.0
    body_a: Body = field(default=None)
    body_b: Body = field(default=None)
    original_body_a: Body = field(init=False)
    original_body_b: Body = field(init=False)
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
        self.original_body_a = self.body_a
        self.original_body_b = self.body_b

        self.data = np.array(
            [
                self.body_b.__hash__(),  # hash
                0,
                0,
                1,  # map_V_n
                self.contact_distance_input,
                0,
                0,
                0,  # new_a_P_pa
                0,
                0,
                1,  # new_b_V_n
                0,
                0,
                0,  # new_b_P_pb
            ],
            dtype=float,
        )
        if self.map_V_n_input is not None:
            self.map_V_n = self.map_V_n_input

    @property
    def external_data(self) -> np.ndarray:
        return self.data[: self._new_b_V_n_idx]

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
        return (
            f"{self.original_body_a}|-|{self.original_body_b}: {self.contact_distance}"
        )

    def __repr__(self):
        return str(self)

    def reverse(self):
        return GiskardCollision(
            body_a=self.original_body_b,
            body_b=self.original_body_a,
            map_P_pa=self.map_P_pb,
            map_P_pb=self.map_P_pa,
            map_V_n_input=-self.map_V_n,
            a_P_pa=self.b_P_pb,
            b_P_pb=self.a_P_pa,
            contact_distance_input=self.contact_distance,
        )


@dataclass
class SortedCollisionResults:
    data: List[GiskardCollision] = field(default_factory=list)
    default_result: GiskardCollision = field(
        default_factory=lambda: GiskardCollision(contact_distance_input=100)
    )

    def _sort(self, x: GiskardCollision):
        return x.contact_distance

    def add(self, element: GiskardCollision):
        self.data.append(element)
        self.data = list(sorted(self.data, key=self._sort))

    def __getitem__(self, item: int) -> GiskardCollision:
        try:
            return self.data[item]
        except (KeyError, IndexError) as e:
            return self.default_result


@dataclass
class Collisions:
    collision_list_size: int
    self_collisions: Dict[Tuple[Body, Body], SortedCollisionResults] = field(
        default_factory=lambda: defaultdict(SortedCollisionResults)
    )
    external_collisions: Dict[Body, SortedCollisionResults] = field(
        default_factory=lambda: defaultdict(SortedCollisionResults)
    )
    external_collision_long_key: Dict[Tuple[Body, Body], GiskardCollision] = field(
        default_factory=lambda: defaultdict(
            lambda: SortedCollisionResults.default_result
        )
    )
    all_collisions: List[GiskardCollision] = field(default_factory=list)
    number_of_self_collisions: Dict[Tuple[Body, Body], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    number_of_external_collisions: Dict[Body, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def get_robot_from_self_collision(
        self, collision: GiskardCollision
    ) -> Optional[AbstractRobot]:
        body_a, body_b = collision.body_a, collision.body_b
        for robot in god_map.collision_scene.robots:
            if body_a in robot.bodies and body_b in robot.bodies:
                return robot

    @classmethod
    def from_collision_list(
        cls, collision_list: List[GiskardCollision], collision_list_size: int
    ):
        collisions = cls(collision_list_size)
        for collision in collision_list:
            collisions.add(collision)
        return collisions

    @profile
    def add(self, collision: GiskardCollision):
        robot = self.get_robot_from_self_collision(collision)
        collision.is_external = robot is None
        if collision.is_external:
            collision = self.transform_external_collision(collision)
            key = collision.body_a
            self.external_collisions[key].add(collision)
            self.number_of_external_collisions[key] = min(
                self.collision_list_size, self.number_of_external_collisions[key] + 1
            )
            key_long = (collision.original_body_a, collision.original_body_b)
            if key_long not in self.external_collision_long_key:
                self.external_collision_long_key[key_long] = collision
            else:
                self.external_collision_long_key[key_long] = min(
                    collision,
                    self.external_collision_long_key[key_long],
                    key=lambda x: x.contact_distance,
                )
        else:
            collision = self.transform_self_collision(collision, robot)
            key = collision.body_a, collision.body_b
            self.self_collisions[key].add(collision)
            try:
                self.number_of_self_collisions[key] = min(
                    self.collision_list_size, self.number_of_self_collisions[key] + 1
                )
            except Exception as e:
                pass
        self.all_collisions.append(collision)

    @profile
    def transform_self_collision(
        self, collision: GiskardCollision, robot: AbstractRobot
    ) -> GiskardCollision:
        link_a = collision.original_body_a
        link_b = collision.original_body_b
        new_link_a, new_link_b = (
            god_map.world.compute_chain_reduced_to_controlled_joints(link_a, link_b)
        )
        if new_link_a.name > new_link_b.name:
            collision = collision.reverse()
            new_link_a, new_link_b = new_link_b, new_link_a
        collision.body_a = new_link_a
        collision.body_b = new_link_b

        new_b_T_r = god_map.world.compute_forward_kinematics_np(new_link_b, robot.root)
        root_T_map = god_map.world.compute_forward_kinematics_np(
            robot.root, god_map.world.root
        )
        new_b_T_map = new_b_T_r @ root_T_map
        collision.new_b_V_n = new_b_T_map @ collision.map_V_n

        if collision.map_P_pa is not None:
            new_a_T_r = god_map.world.compute_forward_kinematics_np(
                new_link_a, robot.root
            )
            collision.new_a_P_pa = new_a_T_r @ root_T_map @ collision.map_P_pa
            collision.new_b_P_pb = new_b_T_map @ collision.map_P_pb
        else:
            new_a_T_a = god_map.world.compute_forward_kinematics_np(
                new_link_a, collision.original_body_a
            )
            collision.new_a_P_pa = new_a_T_a @ collision.a_P_pa
            new_b_T_b = god_map.world.compute_forward_kinematics_np(
                new_link_b, collision.original_body_b
            )
            collision.new_b_P_pb = new_b_T_b @ collision.b_P_pb
        return collision

    @profile
    def transform_external_collision(
        self, collision: GiskardCollision
    ) -> GiskardCollision:
        body_a = collision.original_body_a
        movable_joint = body_a.parent_connection

        def is_joint_movable(connection: ActiveConnection):
            return (
                isinstance(connection, ActiveConnection)
                and connection.has_hardware_interface
                and not connection.frozen_for_collision_avoidance
            )

        while movable_joint != god_map.world.root:
            if is_joint_movable(movable_joint):
                break
            movable_joint = movable_joint.parent.parent_connection
        else:
            raise Exception(
                f"{body_a.name} has no movable parent connection "
                f"and should't have collision checking enabled."
            )
        new_a = movable_joint.child
        collision.body_a = new_a
        if collision.map_P_pa is not None:
            new_a_T_map = god_map.world.compute_forward_kinematics_np(
                new_a, god_map.world.root
            )
            collision.new_a_P_pa = new_a_T_map @ collision.map_P_pa
        else:
            new_a_T_a = god_map.world.compute_forward_kinematics_np(
                new_a, collision.original_body_a
            )
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

    def get_external_collisions_long_key(
        self, link_a: Body, link_b: Body
    ) -> GiskardCollision:
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


class NullCollisionDetector(CollisionDetector):
    def sync_world_model(self) -> None:
        pass

    def sync_world_state(self) -> None:
        pass

    def check_collisions(
        self, collision_matrix: Optional[Set[CollisionCheck]] = None
    ) -> List[GiskardCollision]:
        return []

    def reset_cache(self):
        pass
