from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariable,
)
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
    Collision,
)
from semantic_digital_twin.collision_checking.collision_manager import (
    CollisionGroupConsumer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, Point3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class ExternalCollisionResults:
    """
    Maps bodies to any
    """

    def add_collision(self, collision: Collision): ...


@dataclass
class ExternalCollisionExpressionManager(CollisionGroupConsumer):
    """
    Owns symbols and buffer
    """

    robot: AbstractRobot
    """
    The robot to compute collisions for.
    """

    registered_bodies: dict[Body, int] = field(default_factory=dict)
    """
    Maps bodies to the index of point_on_body_a in the collision buffer.
    """

    collision_data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    """
    All collision data in a single numpy array.
    Repeats blocks of size block_size.
    """

    block_size: int = field(default=9, init=False)
    """
    block layout:
        9 per collision
        point_on_body_a,  (3)
        contract_normal,  (3)
        contact_distance, (1)
        buffer_distance,  (1)
        violated_distance (1)
    """

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
                    buffer_distance=max(
                        self.collision_manager.get_buffer_zone_distance(
                            collision.body_a
                        ),
                        self.collision_manager.get_buffer_zone_distance(
                            collision.body_b
                        ),
                    ),
                    violated_distance=max(
                        self.collision_manager.get_violated_distance(collision.body_a),
                        self.collision_manager.get_violated_distance(collision.body_b),
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
        self.collision_data[start_idx : start_idx + 3] = group1_P_point_on_a[:3]
        self.collision_data[start_idx + 3 : start_idx + 6] = root_V_contact_normal[:3]
        self.collision_data[start_idx + 6] = contact_distance
        self.collision_data[start_idx + 7] = buffer_distance
        self.collision_data[start_idx + 8] = violated_distance

    def register_body(self, body: Body, number_of_potential_collisions: int):
        """
        Register a body
        """
        self.registered_bodies[body] = len(self.collision_data)
        self.collision_data = np.resize(
            self.collision_data,
            self.collision_data.shape[0]
            + self.block_size * number_of_potential_collisions,
        )

    def get_external_collision_variables(self) -> list[FloatVariable]:
        """

        :return: A list of all external collision variables for registered bodies.
        """
        symbols = []
        for body, max_idx in self.external_monitored_links.items():
            for idx in range(max_idx + 1):
                symbols.append(self.external_link_b_hash_symbol(body, idx))

                v = self.external_map_V_n_symbol(body, idx)
                symbols.extend(
                    [
                        v.x.free_variables()[0],
                        v.y.free_variables()[0],
                        v.z.free_variables()[0],
                    ]
                )

                symbols.append(self.external_contact_distance_symbol(body, idx))

                p = self.external_new_a_P_pa_symbol(body, idx)
                symbols.extend(
                    [
                        p.x.free_variables()[0],
                        p.y.free_variables()[0],
                        p.z.free_variables()[0],
                    ]
                )

            symbols.append(self.external_number_of_collisions_symbol(body))
        if len(symbols) != self.external_collision_data.shape[0]:
            self.external_collision_data = np.zeros(len(symbols), dtype=float)
        return symbols

    def get_external_collision_data(self) -> np.ndarray:
        """

        :return: A numpy array containing the external collision data,
                 corresponding to the symbols returned by get_external_collision_variables.
        """

    @lru_cache
    def external_new_a_P_pa_symbol(self, body: Body, idx: int) -> Point3:
        return Point3.create_with_variables(
            f"closest_point({body.name})[{idx}].new_a_P_pa"
        )

    @lru_cache
    def external_map_V_n_symbol(self, body: Body, idx: int) -> Vector3:
        return Vector3.create_with_variables(
            f"closest_point({body.name})[{idx}].map_V_n"
        )

    @lru_cache
    def get_variable_buffer_zone_distance(self, body: Body, idx: int) -> FloatVariable:
        return FloatVariable(name=f"buffer_zone_distance({body.name}, {idx})")

    @lru_cache
    def get_variable_violated_distance(self, body: Body, idx: int) -> FloatVariable:
        return FloatVariable(name=f"violated_distance({body.name}, {idx})")

    @lru_cache
    def external_contact_distance_symbol(
        self, body: Body, idx: int | None = None, body_b: Body | None = None
    ) -> AuxiliaryVariable:
        if body_b is None:
            assert idx is not None
            provider = (
                lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[
                    i
                ].contact_distance
            )
            return AuxiliaryVariable(
                name=str(
                    PrefixedName(f"closest_point({body.name})[{idx}].contact_distance")
                ),
                provider=provider,
            )
        assert body_b is not None
        provider = lambda l1=body, l2=body_b: (
            self.closest_points.get_external_collisions_long_key(
                l1, l2
            ).contact_distance
        )
        return AuxiliaryVariable(
            name=str(
                PrefixedName(
                    f"closest_point({body.name}, {body_b.name}).contact_distance"
                )
            ),
            provider=provider,
        )

    @lru_cache
    def external_link_b_hash_symbol(
        self, body: Body, idx: int | None = None, body_b: Body | None = None
    ) -> AuxiliaryVariable:
        if body_b is None:
            assert idx is not None
            provider = (
                lambda n=body, i=idx: self.closest_points.get_external_collisions(n)[
                    i
                ].link_b_hash
            )
            return AuxiliaryVariable(
                name=str(
                    PrefixedName(f"closest_point({body.name})[{idx}].link_b_hash")
                ),
                provider=provider,
            )
        assert body_b is not None
        provider = lambda l1=body, l2=body_b: (
            self.closest_points.get_external_collisions_long_key(l1, l2).link_b_hash
        )
        return AuxiliaryVariable(
            name=str(
                PrefixedName(f"closest_point({body.name}, {body_b.name}).link_b_hash")
            ),
            provider=provider,
        )

    @lru_cache
    def external_number_of_collisions_symbol(
        self, body: KinematicStructureEntity
    ) -> AuxiliaryVariable:
        provider = lambda n=body: self.closest_points.get_number_of_external_collisions(
            n
        )
        return AuxiliaryVariable(
            name=str(PrefixedName(f"len(closest_point({body.name}))")),
            provider=provider,
        )
