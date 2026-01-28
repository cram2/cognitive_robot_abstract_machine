import tempfile
from dataclasses import dataclass, field

from typing import Dict, Tuple, DefaultDict, List, Set, Optional

import giskardpy_bullet_bindings as bpb


from .bpb_wrapper import create_shape_from_link, create_collision
from .collision_detector import (
    CollisionDetector,
    CollisionCheck,
    CollisionMatrix,
    CollisionCheckingResult,
)
from .collisions import GiskardCollision
from ..datastructures.prefixed_name import PrefixedName
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

    query: Optional[
        DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]
    ] = field(default=None, init=False)

    buffer: float = field(default=0.05, init=False)

    def sync_world_model(self) -> None:
        self.reset_cache()
        self.clear()
        self.body_to_bullet_object = {}
        for body in self.world.bodies_with_enabled_collision:
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

    def collision_matrix_to_bullet_query(
        self, collision_matrix: CollisionMatrix
    ) -> DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]:
        if self.query is None:
            self.query = {
                (
                    self.body_to_bullet_object[check.body_a],
                    self.body_to_bullet_object[check.body_b],
                ): check.distance
                + self.buffer
                for check in collision_matrix.collision_checks
            }
        return self.query

    def check_collisions(
        self, collision_matrix: CollisionMatrix
    ) -> CollisionCheckingResult:

        query = self.collision_matrix_to_bullet_query(collision_matrix)
        result: List[bpb.Collision] = (
            self.kineverse_world.get_closest_filtered_map_batch(query)
        )
        return CollisionCheckingResult(
            [create_collision(collision, self.world) for collision in result]
        )
