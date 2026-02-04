from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict, Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.collision_checking.collision_detector import CollisionCheck
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class CollisionAvoidanceTypes(Enum):
    AVOID_COLLISION = 0
    ALLOW_COLLISION = 1


@dataclass
class CollisionRequest(SubclassJSONSerializer):
    type_: CollisionAvoidanceTypes = field(
        default=CollisionAvoidanceTypes.AVOID_COLLISION
    )
    distance: Optional[float] = None
    body_group1: List[Body] = field(default_factory=list)
    body_group2: List[Body] = field(default_factory=list)

    def __post_init__(self):
        if self.distance is not None and self.distance < 0:
            raise ValueError(f"Distance must be positive or None, got {self.distance}")

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "type_": to_json(self.type_),
            "distance": self.distance,
            "body_group1_ids": [to_json(body.id) for body in self.body_group1],
            "body_group2_ids": [to_json(body.id) for body in self.body_group2],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        body_group1 = [
            tracker.get_world_entity_with_id(from_json(id_))
            for id_ in data["body_group1_ids"]
        ]
        body_group2 = [
            tracker.get_world_entity_with_id(from_json(id_))
            for id_ in data["body_group2_ids"]
        ]
        return cls(
            type_=from_json(data["type_"], **kwargs),
            distance=data["distance"],
            body_group1=body_group1,
            body_group2=body_group2,
        )

    @classmethod
    def avoid_all_collision(cls, distance: Optional[float] = None) -> CollisionRequest:
        return CollisionRequest(
            type_=CollisionAvoidanceTypes.AVOID_COLLISION, distance=distance
        )

    def is_distance_set(self) -> bool:
        return self.distance is not None

    def all_bodies_for_group1(self) -> bool:
        """
        If the group is empty, then all bodies are included.
        """
        return len(self.body_group1) == 0

    def all_bodies_for_group2(self) -> bool:
        """
        If the group is empty, then all bodies are included.
        """
        return len(self.body_group2) == 0

    def is_avoid_collision(self) -> bool:
        return self.type_ == CollisionAvoidanceTypes.AVOID_COLLISION

    def is_allow_collision(self) -> bool:
        return self.type_ == CollisionAvoidanceTypes.ALLOW_COLLISION

    def is_avoid_all_collision(self) -> bool:
        return (
            self.is_avoid_collision()
            and self.all_bodies_for_group1()
            and self.all_bodies_for_group2()
        )

    def is_allow_all_collision(self) -> bool:
        return (
            self.is_allow_collision()
            and self.all_bodies_for_group1()
            and self.all_bodies_for_group2()
        )


class DisableCollisionReason(Enum):
    Unknown = -1
    Never = 1
    Adjacent = 2
    Default = 3
    AlmostAlways = 4


@dataclass
class CollisionMatrixManager:
    """
    Handles all matrix related operations for multiple robots.
    """

    world: World
    robots: Set[AbstractRobot]

    added_checks: Set[CollisionCheck] = field(default_factory=set)

    collision_requests: List[CollisionRequest] = field(default_factory=list)
    """
    Motion goal specific requests for collision avoidance checks.
    Can overwrite the thresholds and add additional disabled bodies/pairs.
    """

    def compute_collision_matrix(self) -> Set[CollisionCheck]:
        """
        Parses the collision requrests and (temporary) collision configs in the world
        to create a set of collision checks.
        """
        collision_matrix: Set[CollisionCheck] = set()
        for collision_request in self.collision_requests:
            if collision_request.all_bodies_for_group1():
                view_1_bodies = self.world.bodies_with_enabled_collision
            else:
                view_1_bodies = collision_request.body_group1
            if collision_request.all_bodies_for_group2():
                view2_bodies = self.world.bodies_with_enabled_collision
            else:
                view2_bodies = collision_request.body_group2
            disabled_pairs = self.world._collision_pair_manager.disabled_collision_pairs
            for body1 in view_1_bodies:
                for body2 in view2_bodies:
                    collision_check = CollisionCheck(
                        body_a=body1, body_b=body2, distance=0, _world=self.world
                    )
                    (robot_body, env_body) = collision_check.bodies()
                    if (robot_body, env_body) in disabled_pairs:
                        continue
                    if collision_request.distance is None:
                        distance = max(
                            robot_body.get_collision_config().buffer_zone_distance
                            or 0.0,
                            env_body.get_collision_config().buffer_zone_distance or 0.0,
                        )
                    else:
                        distance = collision_request.distance
                    if not collision_request.is_allow_collision():
                        collision_check.distance = distance
                        collision_check._validate()
                    if collision_request.is_allow_collision():
                        if collision_check in collision_matrix:
                            collision_matrix.remove(collision_check)
                    if collision_request.is_avoid_collision():
                        if collision_request.is_distance_set():
                            collision_matrix.add(collision_check)
                        else:
                            collision_matrix.add(collision_check)
        return collision_matrix

    def add_collision_check(self, body_a: Body, body_b: Body, distance: float):
        """
        Tell Giskard to check this collision, even if it got disabled through other means such as allow_all_collisions.
        """
        check = CollisionCheck.create_and_validate(
            body_a=body_a, body_b=body_b, distance=distance, world=self.world
        )
        if check in self.added_checks:
            raise ValueError(f"Collision check {check} already added")
        self.added_checks.add(check)

    def parse_collision_requests(self, collision_goals: List[CollisionRequest]) -> None:
        """
        Resolve an incoming list of collision goals into collision checks.
        1. remove redundancy
        2. remove entries before "avoid all" or "allow all"
        :param collision_goals:
        :return:
        """
        for i, collision_goal in enumerate(reversed(collision_goals)):
            if collision_goal.is_avoid_all_collision():
                # remove everything before the avoid all
                collision_goals = collision_goals[len(collision_goals) - i - 1 :]
                break
            if collision_goal.is_allow_all_collision():
                # remove everything before the allow all, including the allow all
                collision_goals = collision_goals[len(collision_goals) - i :]
                break
        else:
            # put an avoid all at the front
            collision_goal = CollisionRequest()
            collision_goal.type_ = CollisionAvoidanceTypes.AVOID_COLLISION
            collision_goal.distance = None
            collision_goals.insert(0, collision_goal)

        self.collision_requests = list(collision_goals)

    def get_non_robot_bodies(self) -> Set[Body]:
        return set(self.world.bodies_with_enabled_collision).difference(
            self.get_robot_bodies()
        )

    def get_robot_bodies(self) -> Set[Body]:
        robot_bodies = set()
        for robot in self.robots:
            robot_bodies.update(robot.bodies_with_enabled_collision)
        return robot_bodies
