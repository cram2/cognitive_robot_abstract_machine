from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from line_profiler import profile

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.model.collision_matrix_manager import CollisionMatrixManager
from giskardpy.model.collisions import NullCollisionDetector, Collisions
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariable,
    create_vector3,
    create_point,
)
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionDetector,
    CollisionCheck,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

np.random.seed(1337)


class CollisionCheckerLib(Enum):
    none = -1
    bpb = 1


@dataclass
class CollisionWorldSynchronizer:
    world: World
    robots: Set[AbstractRobot]

    collision_detector: CollisionDetector = None
    matrix_manager: CollisionMatrixManager = field(init=False)

    collision_matrix: Set[CollisionCheck] = field(default_factory=set)

    external_monitored_links: Dict[Body, int] = field(default_factory=dict)
    self_monitored_links: Dict[Tuple[Body, Body], int] = field(default_factory=dict)
    world_model_version: int = -1

    external_collision_data: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    self_collision_data: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    collision_list_sizes: int = 1000

    def __post_init__(self):
        self.matrix_manager = CollisionMatrixManager(
            world=self.world, robots=self.robots
        )

    def __hash__(self):
        return hash(id(self))

    def sync(self):
        if self.has_world_model_changed():
            self.collision_detector.sync_world_model()
        self.collision_detector.sync_world_state()

    def has_world_model_changed(self) -> bool:
        if self.world_model_version != self.world.get_world_model_manager().version:
            self.world_model_version = self.world.get_world_model_manager().version
            return True
        return False

    def set_collision_matrix(self, collision_matrix):
        self.collision_matrix = collision_matrix

    def check_collisions(self) -> Collisions:
        collisions = self.collision_detector.check_collisions(self.collision_matrix)
        self.closest_points = Collisions.from_collision_list(
            collisions, self.collision_list_sizes, robots=list(self.robots)
        )
        return self.closest_points

    def is_collision_checking_enabled(self) -> bool:
        return not isinstance(self.collision_detector, NullCollisionDetector)

    # %% external collision symbols
    def monitor_link_for_external(self, body: Body, idx: int):
        self.external_monitored_links[body] = max(
            idx, self.external_monitored_links.get(body, 0)
        )

    def reset_cache(self):
        self.collision_detector.reset_cache()

    def get_external_collision_symbol(self) -> List[sm.FloatVariable]:
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

    @lru_cache
    def external_map_V_n_symbol(self, body: Body, idx: int) -> Vector3:
        return create_vector3(
            name=PrefixedName(f"closest_point({body.name})[{idx}].map_V_n.x"),
            provider=lambda n=body, i=idx: self.closest_points.get_external_collisions(
                n
            )[i].map_V_n,
        )

    @lru_cache
    def external_new_a_P_pa_symbol(self, body: Body, idx: int) -> Point3:
        return create_point(
            name=PrefixedName(f"closest_point({body.name})[{idx}].new_a_P_pa"),
            provider=lambda n=body, i=idx: self.closest_points.get_external_collisions(
                n
            )[i].new_a_P_pa,
        )

    @lru_cache
    def external_contact_distance_symbol(
        self, body: Body, idx: Optional[int] = None, body_b: Optional[Body] = None
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
        self, body: Body, idx: Optional[int] = None, body_b: Optional[Body] = None
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

    # %% self collision symbols
    def monitor_link_for_self(self, body_a: Body, body_b: Body, idx: int):
        self.self_monitored_links[body_a, body_b] = max(
            idx, self.self_monitored_links.get((body_a, body_b), 0)
        )

    def get_self_collision_symbol(self) -> List[sm.FloatVariable]:
        symbols = []
        for (link_a, link_b), max_idx in self.self_monitored_links.items():
            for idx in range(max_idx + 1):
                symbols.append(self.self_contact_distance_symbol(link_a, link_b, idx))

                p = self.self_new_a_P_pa_symbol(link_a, link_b, idx)
                symbols.extend(
                    [
                        p.x.free_variables()[0],
                        p.y.free_variables()[0],
                        p.z.free_variables()[0],
                    ]
                )

                v = self.self_new_b_V_n_symbol(link_a, link_b, idx)
                symbols.extend(
                    [
                        v.x.free_variables()[0],
                        v.y.free_variables()[0],
                        v.z.free_variables()[0],
                    ]
                )

                p = self.self_new_b_P_pb_symbol(link_a, link_b, idx)
                symbols.extend(
                    [
                        p.x.free_variables()[0],
                        p.y.free_variables()[0],
                        p.z.free_variables()[0],
                    ]
                )

            symbols.append(self.self_number_of_collisions_symbol(link_a, link_b))
        if len(symbols) != self.self_collision_data.shape[0]:
            self.self_collision_data = np.zeros(len(symbols), dtype=float)
        return symbols

    @lru_cache
    def self_new_b_V_n_symbol(self, link_a: Body, link_b: Body, idx: int) -> Vector3:
        provider = (
            lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(
                a, b
            )[i].new_b_V_n
        )
        return create_vector3(
            name=PrefixedName(
                f"closest_point({link_a.name}, {link_b.name})[{idx}].new_b_V_n"
            ),
            provider=provider,
        )

    @lru_cache
    def self_new_a_P_pa_symbol(self, link_a: Body, link_b: Body, idx: int) -> Point3:
        provider = (
            lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(
                a, b
            )[i].new_a_P_pa
        )
        return create_point(
            name=PrefixedName(
                f"closest_point({link_a.name}, {link_b.name}).new_a_P_pa"
            ),
            provider=provider,
        )

    @lru_cache
    def self_new_b_P_pb_symbol(self, link_a: Body, link_b: Body, idx: int) -> Point3:
        provider = (
            lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(
                a, b
            )[i].new_b_P_pb
        )
        p = create_point(
            name=PrefixedName(
                f"closest_point({link_a.name}, {link_b.name}).new_b_P_pb"
            ),
            provider=provider,
        )
        return p

    @lru_cache
    def self_contact_distance_symbol(
        self, link_a: Body, link_b: Body, idx: int
    ) -> AuxiliaryVariable:
        provider = (
            lambda a=link_a, b=link_b, i=idx: self.closest_points.get_self_collisions(
                a, b
            )[i].contact_distance
        )
        return AuxiliaryVariable(
            name=str(
                PrefixedName(
                    f"closest_point({link_a.name}, {link_b.name}).contact_distance"
                )
            ),
            provider=provider,
        )

    @lru_cache
    def self_number_of_collisions_symbol(
        self, link_a: Body, link_b: Body
    ) -> AuxiliaryVariable:
        provider = lambda a=link_a, b=link_b: self.closest_points.get_number_of_self_collisions(
            a, b
        )
        return AuxiliaryVariable(
            name=str(PrefixedName(f"len(closest_point({link_a.name}, {link_b.name}))")),
            provider=provider,
        )

    @profile
    def get_external_collision_data(self) -> np.ndarray:
        offset = 0
        for link_name, max_idx in self.external_monitored_links.items():
            collisions = self.closest_points.get_external_collisions(link_name)

            for idx in range(max_idx + 1):
                np.copyto(
                    self.external_collision_data[offset : offset + 8],
                    collisions[idx].external_data,
                )
                offset += 8

            self.external_collision_data[offset] = (
                self.closest_points.get_number_of_external_collisions(link_name)
            )
            offset += 1

        return self.external_collision_data

    @profile
    def get_self_collision_data(self) -> np.ndarray:

        offset = 0
        for (link_a, link_b), max_idx in self.self_monitored_links.items():
            collisions = self.closest_points.get_self_collisions(link_a, link_b)

            for idx in range(max_idx + 1):
                np.copyto(
                    self.self_collision_data[offset : offset + 10],
                    collisions[idx].self_data,
                )
                offset += 10

            self.self_collision_data[offset] = (
                self.closest_points.get_number_of_self_collisions(link_a, link_b)
            )
            offset += 1

        return self.self_collision_data
