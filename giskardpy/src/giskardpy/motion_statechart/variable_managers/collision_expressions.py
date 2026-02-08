from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
    Collision,
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
from .float_variable_manager import FloatVariableData


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
        closest_contacts: dict[Body, list[Collision]] = defaultdict(list)
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
            collisions = sorted(collisions, key=lambda c: c.contact_distance)
            for i in range(
                min(
                    len(collisions),
                    self.collision_manager.get_max_avoided_bodies(body_a),
                )
            ):
                collision = collisions[i]
                group1 = self.get_collision_group(collision.body_a)
                group1_T_root = group1.root.global_pose.inverse().to_np()
                group1_P_pa = group1_T_root @ collision.root_P_pa
                self.insert_data_block(
                    body=group1.root,
                    idx=i,
                    group1_P_point_on_a=group1_P_pa,
                    root_V_contact_normal=collision.root_V_n,
                    contact_distance=collision.contact_distance,
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
        group1_P_point_on_a: np.ndarray,
        root_V_contact_normal: np.ndarray,
        contact_distance: float,
        buffer_distance: float,
        violated_distance: float,
    ):
        start_idx = self.registered_bodies[body] + idx * self.block_size
        self.float_variable_data.data[
            start_idx : start_idx + self._contact_normal_offset
        ] = group1_P_point_on_a[:3]
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
            self.get_group1_P_point_on_a_symbol(body, index)
            self.get_group1_V_contact_normal_symbol(body, index)
            self.get_contact_distance_symbol(body, index)
            self.get_buffer_distance_symbol(body, index)
            self.get_violated_distance_symbol(body, index)
            self._reset_data = np.append(self._reset_data, self._single_reset_block)

        for group in self.collision_groups:
            if body in group:
                self.active_groups.add(group)

    @lru_cache
    def get_group1_P_point_on_a_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Point3:
        point = Point3.create_with_variables(
            name=f"group1_P_point_on_a({body.name}, {idx})",
        )
        self.float_variable_data.add_variables_of_expression(point)
        return point

    @lru_cache
    def get_group1_V_contact_normal_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Vector3:
        vector = Vector3.create_with_variables(
            f"group1_V_contact_normal({body.name}, {idx})",
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
