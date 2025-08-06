from typing import Dict, Tuple, DefaultDict, List, Set, Optional

import betterpybullet as bpb
from line_profiler import profile

from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.bpb_wrapper import create_shape_from_link, BPCollisionWrapper
from giskardpy.model.collision_detector import CollisionDetector
from giskardpy.model.collision_matrix_manager import CollisionCheck
from semantic_world.prefixed_name import PrefixedName
from semantic_world.world_entity import Body


class BulletCollisionDetector(CollisionDetector):
    collision_list_sizes: float = 1000

    def __init__(self, ):
        self.kw = bpb.KineverseWorld()
        self.object_name_to_id: Dict[PrefixedName, bpb.CollisionObject] = {}
        self.query: Optional[DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]] = None
        super().__init__()

    @profile
    def add_object(self, link: Body):
        if not link.has_collision():
            return
        o = create_shape_from_link(link)
        self.kw.add_collision_object(o)
        self.object_name_to_id[link.name] = o

    def reset_cache(self):
        self.query = None

    @profile
    def cut_off_distances_to_query(self, collision_matrix: Set[CollisionCheck],
                                   buffer: float = 0.05) -> DefaultDict[
        PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]:
        if self.query is None:
            self.query = {(self.object_name_to_id[check.body_a.name],
                           self.object_name_to_id[check.body_b.name]): check.distance + buffer for check in
                          collision_matrix}
        return self.query

    def check_collisions(self,
                         collision_matrix: Set[CollisionCheck],
                         buffer: float = 0.05) -> Collisions:

        query = self.cut_off_distances_to_query(collision_matrix, buffer=buffer)
        result: List[bpb.Collision] = self.kw.get_closest_filtered_map_batch(query)
        self.closest_points = self.bpb_result_to_collisions(result, self.collision_list_sizes)
        return self.closest_points

    # @profile
    # def find_colliding_combinations(self, link_combinations: Iterable[Tuple[PrefixedName, PrefixedName]],
    #                                 distance: float,
    #                                 update_query: bool) -> Set[Tuple[PrefixedName, PrefixedName, float]]:
    #     if update_query:
    #         self.query = None
    #         self.collision_matrix = {link_combination: distance for link_combination in link_combinations}
    #     else:
    #         self.collision_matrix = {}
    #     self.sync()
    #     collisions = self.check_collisions(buffer=0.0)
    #     colliding_combinations = {(c.original_link_a, c.original_link_b, c.contact_distance) for c in
    #                               collisions.all_collisions
    #                               if c.contact_distance <= distance}
    #     return colliding_combinations

    @profile
    def bpb_result_to_collisions(self, result: List[bpb.Collision],
                                 collision_list_size: int) -> Collisions:
        collisions = Collisions(collision_list_size)

        for collision in result:
            giskard_collision = BPCollisionWrapper(collision)
            collisions.add(giskard_collision)
        return collisions

    # def check_collision(self, link_a, link_b, distance):
    #     self.sync()
    #     query = defaultdict(set)
    #     query[self.object_name_to_id[link_a]].add((self.object_name_to_id[link_b], distance))
    #     return self.kw.get_closest_filtered_POD_batch(query)

    def sync_world_model(self) -> None:
        self.reset_cache()
        get_middleware().logdebug('hard sync')
        for o in self.kw.collision_objects:
            self.kw.remove_collision_object(o)
        self.object_name_to_id = {}
        self.objects_in_order = []

        for link in sorted(god_map.world.bodies_with_collisions):
            self.add_object(link)
            self.objects_in_order.append(self.object_name_to_id[link.name])

    def sync_world_state(self) -> None:
        bpb.batch_set_transforms(self.objects_in_order, god_map.world.compute_all_collision_fks())

    @profile
    def get_map_T_geometry(self, body: PrefixedName, collision_id: int = 0):
        collision_object = self.object_name_to_id[body]
        return collision_object.compound_transform(collision_id)
