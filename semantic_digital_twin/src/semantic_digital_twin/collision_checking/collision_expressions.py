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

    registered_bodies: dict[Body, int] = field(default_factory=dict)

    collision_data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    def on_reset(self):
        pass

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision: CollisionCheckingResult):
        """
        Takes collisions, checks if they are external and inserts them
        into the buffer at the right place.
        """
        # 1. check if collision is external
        # 2. check if body is registered
        # 2.1 maybe flip collision, if body_b is registered.
        # 3. transform collision into group frames
        # 4. insert collision into buffer

    def register_body(self, body: Body, idx: int):
        """
        Register a body
        """
        self.external_monitored_links[body] = max(
            idx, self.external_monitored_links.get(body, 0)
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
    def external_map_V_n_symbol(self, body: Body, idx: int) -> Vector3:
        return Vector3.create_with_variables(
            f"closest_point({body.name})[{idx}].map_V_n"
        )

    @lru_cache
    def external_new_a_P_pa_symbol(self, body: Body, idx: int) -> Point3:
        return Point3.create_with_variables(
            f"closest_point({body.name})[{idx}].new_a_P_pa"
        )

    @lru_cache
    def get_variable_buffer_zone_distance(self, body: Body) -> FloatVariable:
        return FloatVariable(name=f"buffer_zone_distance({body.name})")

    @lru_cache
    def get_variable_violated_distance(self, body: Body) -> FloatVariable:
        return FloatVariable(name=f"violated_distance({body.name})")

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

    def transform_to_collision_groups(
        self, collision_results: CollisionCheckingResult
    ) -> CollisionGroupResults:
        result = CollisionGroupResults()
        for collision in collision_results.contacts:
            group1 = self.get_collision_group(collision.body_a)
            group2 = self.get_collision_group(collision.body_b)
            if group1 == group2:
                raise YouFoundABugError(
                    message="Collision between two bodies in the same group."
                )
            group1_T_root = group1.root.global_pose.inverse().to_np()
            group2_T_root = group2.root.global_pose.inverse().to_np()
            group1_P_pa = group1_T_root @ collision.root_P_pa
            group2_P_pb = group2_T_root @ collision.root_P_pb
            data = np.concatenate(
                (
                    group1_P_pa,
                    group2_P_pb,
                    collision.root_V_n,
                    collision.contact_distance,
                    max(
                        self.get_buffer_zone_distance(collision.body_a),
                        self.get_buffer_zone_distance(collision.body_b),
                    ),
                    max(
                        self.get_violated_violated_distance(collision.body_a),
                        self.get_violated_violated_distance(collision.body_b),
                    ),
                )
            )
