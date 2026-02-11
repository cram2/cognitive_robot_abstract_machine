from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
    ClosestPoints,
)
from semantic_digital_twin.collision_checking.collision_manager import (
    CollisionGroupConsumer,
    CollisionGroup,
)
from semantic_digital_twin.spatial_types import Vector3, Point3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from krrood.symbolic_math.float_variable_data import (
    FloatVariableData,
)


@dataclass
class ExternalCollisionVariableManager(CollisionGroupConsumer):
    """
    Owns symbols and buffer
    """

    float_variable_data: FloatVariableData = field(default_factory=FloatVariableData)

    registered_bodies: dict[KinematicStructureEntity, int] = field(
        default_factory=dict, init=False
    )
    """
    Maps bodies to the index of point_on_body_a in the collision buffer.
    """

    active_groups: set[CollisionGroup] = field(default_factory=set, init=False)

    block_size: int = field(default=9, init=False)
    """
    block layout:
        9 per collision
        point_on_body_a,  (3)
        contact_normal,  (3)
        contact_distance, (1)
        buffer_distance,  (1)
        violated_distance (1)
    """
    _point_on_a_offset: int = field(init=False, default=0)
    _contact_normal_offset: int = field(init=False, default=3)
    _contact_distance_offset: int = field(init=False, default=6)
    _buffer_distance_offset: int = field(init=False, default=7)
    _violated_distance_offset: int = field(init=False, default=8)

    _collision_data_start_index: int = field(init=False, default=None)
    _single_reset_block: np.ndarray = field(init=False)
    _reset_data: np.ndarray = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self):
        self._single_reset_block = np.zeros(self.block_size)
        self._single_reset_block[self._contact_distance_offset] = 100

    def __hash__(self):
        return hash(id(self))

    def on_reset(self):
        pass

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision: CollisionCheckingResult):
        """
        Takes collisions, checks if they are external, and inserts them
        into the buffer at the right place.
        """
        self.reset_collision_data()
        closest_contacts: dict[Body, list[ClosestPoints]] = defaultdict(list)
        for collision in collision.contacts:
            # 1. check if collision is external
            if (
                collision.body_a not in self.registered_bodies
                and collision.body_b not in self.registered_bodies
            ):
                continue
            if collision.body_a not in self.registered_bodies:
                collision = collision.reverse()
            closest_contacts[collision.body_a].append(collision)

        for body_a, collisions in closest_contacts.items():
            collisions = sorted(collisions, key=lambda c: c.distance)
            for i in range(
                min(
                    len(collisions),
                    self.collision_manager.get_max_avoided_bodies(body_a),
                )
            ):
                collision = collisions[i]
                group_a = self.get_collision_group(collision.body_a)
                group_a_T_root = group_a.root.global_pose.inverse().to_np()
                group_a_P_pa = group_a_T_root @ collision.root_P_point_on_body_a
                self.insert_data_block(
                    body=group_a.root,
                    idx=i,
                    group_a_P_point_on_a=group_a_P_pa,
                    root_V_contact_normal=collision.root_V_contact_normal_from_b_to_a,
                    contact_distance=collision.distance,
                    buffer_distance=self.collision_manager.get_buffer_zone_distance(
                        collision.body_a, collision.body_b
                    ),
                    violated_distance=self.collision_manager.get_violated_distance(
                        collision.body_a, collision.body_b
                    ),
                )

    def insert_data_block(
        self,
        body: KinematicStructureEntity,
        idx: int,
        group_a_P_point_on_a: np.ndarray,
        root_V_contact_normal: np.ndarray,
        contact_distance: float,
        buffer_distance: float,
        violated_distance: float,
    ):
        start_idx = self.registered_bodies[body] + idx * self.block_size
        self.float_variable_data.data[
            start_idx : start_idx + self._contact_normal_offset
        ] = group_a_P_point_on_a[:3]
        self.float_variable_data.data[
            start_idx
            + self._contact_normal_offset : start_idx
            + self._contact_distance_offset
        ] = root_V_contact_normal[:3]
        self.float_variable_data.data[start_idx + self._contact_distance_offset] = (
            contact_distance
        )
        self.float_variable_data.data[start_idx + self._buffer_distance_offset] = (
            buffer_distance
        )
        self.float_variable_data.data[start_idx + self._violated_distance_offset] = (
            violated_distance
        )

    def reset_collision_data(self):
        start_index = self._collision_data_start_index
        end_index = start_index + self._reset_data.size
        self.float_variable_data.data[start_index:end_index] = self._reset_data

    def register_body(self, body: Body):
        """
        Register a body
        """
        self.registered_bodies[body] = len(self.float_variable_data.data)
        if self._collision_data_start_index is None:
            self._collision_data_start_index = self.registered_bodies[body]
        for index in range(self.collision_manager.get_max_avoided_bodies(body)):
            self.get_group_a_P_point_on_a_symbol(body, index)
            self.get_root_V_contact_normal_symbol(body, index)
            self.get_contact_distance_symbol(body, index)
            self.get_buffer_distance_symbol(body, index)
            self.get_violated_distance_symbol(body, index)
            self._reset_data = np.append(self._reset_data, self._single_reset_block)

        for group in self.collision_groups:
            if body in group:
                self.active_groups.add(group)

    @lru_cache
    def get_group_a_P_point_on_a_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Point3:
        point = Point3.create_with_variables(
            name=f"group_a_P_point_on_a({body.name}, {idx})",
        )
        self.float_variable_data.add_variables_of_expression(point)
        return point

    @lru_cache
    def get_root_V_contact_normal_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Vector3:
        vector = Vector3.create_with_variables(
            f"root_V_contact_normal({body.name}, {idx})",
        )
        self.float_variable_data.add_variables_of_expression(vector)
        return vector

    @lru_cache
    def get_contact_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        variable = FloatVariable(f"contact_distance({body.name}, {idx})")
        self.float_variable_data.add_variable(variable)
        return variable

    @lru_cache
    def get_buffer_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        variable = FloatVariable(f"buffer_distance({body.name}, {idx})")
        self.float_variable_data.add_variable(variable)
        return variable

    @lru_cache
    def get_violated_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        variable = FloatVariable(f"violated_distance({body.name}, {idx})")
        self.float_variable_data.add_variable(variable)
        return variable


@dataclass
class SelfCollisionVariableManager(CollisionGroupConsumer):
    """
    Owns symbols and buffer
    """

    float_variable_data: FloatVariableData = field(default_factory=FloatVariableData)

    registered_body_combinations: dict[tuple[CollisionGroup, CollisionGroup], int] = (
        field(default_factory=dict, init=False)
    )
    """
    Maps body combinations to the index of point_on_body_a in the collision buffer.
    """

    active_groups: set[tuple[CollisionGroup, CollisionGroup]] = field(
        default_factory=set, init=False
    )

    block_size: int = field(default=12, init=False)
    """
    block layout:
        12 per collision
        point_on_body_a,  (3)
        point_on_body_b,  (3)
        contact_normal,  (3)
        contact_distance, (1)
        buffer_distance,  (1)
        violated_distance (1)
    """
    _point_on_a_offset: int = field(init=False, default=0)
    _point_on_b_offset: int = field(init=False, default=3)
    _contact_normal_offset: int = field(init=False, default=6)
    _contact_distance_offset: int = field(init=False, default=9)
    _buffer_distance_offset: int = field(init=False, default=10)
    _violated_distance_offset: int = field(init=False, default=11)

    _collision_data_start_index: int = field(init=False, default=None)
    _single_reset_block: np.ndarray = field(init=False)
    _reset_data: np.ndarray = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self):
        self._single_reset_block = np.zeros(self.block_size)
        self._single_reset_block[self._contact_distance_offset] = 100

    def __hash__(self):
        return hash(id(self))

    def on_reset(self):
        pass

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision_result: CollisionCheckingResult):
        """
        Takes collisions, checks if they are external, and inserts them
        into the buffer at the right place.
        """
        self.reset_collision_data()
        closest_contacts: dict[
            tuple[CollisionGroup, CollisionGroup], list[ClosestPoints]
        ] = defaultdict(list)
        for collision in collision_result.contacts:
            key = (
                self.get_collision_group(collision.body_b),
                self.get_collision_group(collision.body_a),
            )
            # 1. check if collision is external
            if key in self.registered_body_combinations:
                collision = collision.reverse()
            else:
                key = tuple(reversed(key))
            if key not in self.registered_body_combinations:
                continue
            closest_contacts[key].append(collision)

        for (group_a, group_b), collisions in closest_contacts.items():
            closest_contact = sorted(collisions, key=lambda c: c.distance)[0]
            group_a_T_root = group_a.root.global_pose.inverse().to_np()
            group_a_P_pa = group_a_T_root @ closest_contact.root_P_point_on_body_a

            group_b_T_root = group_b.root.global_pose.inverse().to_np()
            group_b_P_pb = group_b_T_root @ closest_contact.root_P_point_on_body_b

            group_b_V_contact_normal = (
                group_b_T_root @ closest_contact.root_V_contact_normal_from_b_to_a
            )

            self.insert_data_block(
                group_a=group_a,
                group_b=group_b,
                group_a_P_point_on_a=group_a_P_pa,
                group_b_P_point_on_b=group_b_P_pb,
                group_b_V_contact_normal=group_b_V_contact_normal,
                contact_distance=closest_contact.distance,
                buffer_distance=self.collision_manager.get_buffer_zone_distance(
                    closest_contact.body_a, closest_contact.body_b
                ),
                violated_distance=self.collision_manager.get_violated_distance(
                    closest_contact.body_a, closest_contact.body_b
                ),
            )

    def insert_data_block(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
        group_a_P_point_on_a: np.ndarray,
        group_b_P_point_on_b: np.ndarray,
        group_b_V_contact_normal: np.ndarray,
        contact_distance: float,
        buffer_distance: float,
        violated_distance: float,
    ):
        block_start_idx = self.registered_body_combinations[group_a, group_b]

        start_idx = block_start_idx + self._point_on_a_offset
        end_idx = block_start_idx + self._point_on_b_offset
        self.float_variable_data.data[start_idx:end_idx] = group_a_P_point_on_a[:3]

        start_idx = end_idx
        end_idx = block_start_idx + self._contact_normal_offset
        self.float_variable_data.data[start_idx:end_idx] = group_b_P_point_on_b[:3]

        start_idx = end_idx
        end_idx = block_start_idx + self._contact_distance_offset
        self.float_variable_data.data[start_idx:end_idx] = group_b_V_contact_normal[:3]

        self.float_variable_data.data[
            block_start_idx + self._contact_distance_offset
        ] = contact_distance
        self.float_variable_data.data[
            block_start_idx + self._buffer_distance_offset
        ] = buffer_distance
        self.float_variable_data.data[
            block_start_idx + self._violated_distance_offset
        ] = violated_distance

    def reset_collision_data(self):
        start_index = self._collision_data_start_index
        end_index = start_index + self._reset_data.size
        self.float_variable_data.data[start_index:end_index] = self._reset_data

    def body_pair_to_group_pair(
        self, body_a: Body, body_b: Body
    ) -> tuple[CollisionGroup, CollisionGroup]:
        group_a, group_b = self.get_collision_group(body_a), self.get_collision_group(
            body_b
        )
        if group_a.root.id < group_b.root.id:
            return group_a, group_b
        else:
            return group_b, group_a

    def register_body_combination(self, body_a: Body, body_b: Body):
        """
        Register a body
        """
        key = self.body_pair_to_group_pair(body_a, body_b)
        self.active_groups.add(key)

        self.registered_body_combinations[key] = len(self.float_variable_data.data)
        if self._collision_data_start_index is None:
            self._collision_data_start_index = self.registered_body_combinations[key]
        self.get_group_a_P_point_on_a_symbol(*key)
        self.get_group_b_P_point_on_b_symbol(*key)
        self.get_group_b_V_contact_normal_symbol(*key)
        self.get_contact_distance_symbol(*key)
        self.get_buffer_distance_symbol(*key)
        self.get_violated_distance_symbol(*key)
        self._reset_data = np.append(self._reset_data, self._single_reset_block)

    @lru_cache
    def get_group_a_P_point_on_a_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> Point3:
        point = Point3.create_with_variables(
            name=f"group_a_P_point_on_a({group_a}, {group_b})",
        )
        self.float_variable_data.add_variables_of_expression(point)
        return point

    @lru_cache
    def get_group_b_P_point_on_b_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> Point3:
        point = Point3.create_with_variables(
            name=f"group_b_P_point_on_b({group_a}, {group_b})",
        )
        self.float_variable_data.add_variables_of_expression(point)
        return point

    @lru_cache
    def get_group_b_V_contact_normal_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> Vector3:
        vector = Vector3.create_with_variables(
            f"group_b_V_contact_normal({group_a}, {group_b})",
        )
        self.float_variable_data.add_variables_of_expression(vector)
        return vector

    @lru_cache
    def get_contact_distance_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> FloatVariable:
        variable = FloatVariable(f"contact_distance({group_a}, {group_b})")
        self.float_variable_data.add_variable(variable)
        return variable

    @lru_cache
    def get_buffer_distance_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> FloatVariable:
        variable = FloatVariable(f"buffer_distance({group_a}, {group_b})")
        self.float_variable_data.add_variable(variable)
        return variable

    @lru_cache
    def get_violated_distance_symbol(
        self,
        group_a: CollisionGroup,
        group_b: CollisionGroup,
    ) -> FloatVariable:
        variable = FloatVariable(f"violated_distance({group_a}, {group_b})")
        self.float_variable_data.add_variable(variable)
        return variable
