from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple, List, Optional

import giskardpy_bullet_bindings as bpb

from .bpb_wrapper import create_shape_from_link
from .collision_detector import (
    CollisionDetector,
    CollisionCheckingResult,
    Collision,
)
from .collision_matrix import CollisionMatrix
from ..world_description.world_entity import Body


@dataclass
class BulletCollisionDetector(CollisionDetector):
    kineverse_world: bpb.KineverseWorld = field(
        default_factory=bpb.KineverseWorld, init=False
    )
    body_to_bullet_object: Dict[Body, bpb.CollisionObject] = field(
        default_factory=dict, init=False
    )
    ordered_bullet_objects: List[bpb.CollisionObject] = field(default_factory=list)

    query: Optional[Dict[Tuple[bpb.CollisionObject, bpb.CollisionObject], float]] = (
        field(default=None, init=False)
    )

    buffer: float = field(default=0.05, init=False)

    def sync_world_model(self) -> None:
        if self.world.is_empty():
            return
        self.reset_cache()
        self.clear()
        self.body_to_bullet_object = {}
        for body in self.world.bodies_with_collision:
            self.add_body(body)
        self.ordered_bullet_objects = list(self.body_to_bullet_object.values())

    def clear(self):
        for o in self.kineverse_world.collision_objects:
            self.kineverse_world.remove_collision_object(o)

    def sync_world_state(self) -> None:
        bpb.batch_set_transforms(
            self.ordered_bullet_objects,
            self.get_all_collision_fks(),
        )

    def add_body(self, body: Body):
        o = create_shape_from_link(body=body)
        self.kineverse_world.add_collision_object(o)
        self.body_to_bullet_object[body] = o

    def reset_cache(self):
        self.query = None

    def __hash__(self):
        return hash(id(self))

    @lru_cache(maxsize=100)
    def collision_matrix_to_bullet_query(
        self, collision_matrix: CollisionMatrix
    ) -> Optional[Dict[Tuple[bpb.CollisionObject, bpb.CollisionObject], float]]:
        return {
            (
                self.body_to_bullet_object[check.body_a],
                self.body_to_bullet_object[check.body_b],
            ): check.distance
            + self.buffer
            for check in collision_matrix.collision_checks
        }

    def check_collisions(
        self, collision_matrix: CollisionMatrix
    ) -> CollisionCheckingResult:

        query = self.collision_matrix_to_bullet_query(collision_matrix)
        result: List[bpb.Collision] = (
            self.kineverse_world.get_closest_filtered_map_batch(query)
        )
        return CollisionCheckingResult(
            [
                Collision.from_parts(
                    body_a=self.world.get_kinematic_structure_entity_by_id(
                        collision.obj_a.name
                    ),
                    body_b=self.world.get_kinematic_structure_entity_by_id(
                        collision.obj_b.name
                    ),
                    contact_distance=collision.contact_distance,
                    root_P_pa=collision.map_P_pa,
                    root_P_pb=collision.map_P_pb,
                    root_V_n=collision.world_V_n,
                )
                for collision in result
            ]
        )
