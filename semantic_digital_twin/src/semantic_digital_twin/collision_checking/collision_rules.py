from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, Any
from itertools import combinations, combinations_with_replacement

import numpy as np
from lxml import etree
from typing_extensions import (
    List,
    TYPE_CHECKING,
    Self,
    Iterable,
    Callable,
    ClassVar,
)

from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints, CollisionCheckingResult
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionRule,
    CollisionMatrix,
    CollisionCheck,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from semantic_digital_twin.robots.abstract_robot import AbstractRobot

if TYPE_CHECKING:
    from ..world import World
    from ..world_description.world_entity import Body


@dataclass
class AvoidCollisionRule(CollisionRule, ABC):
    """
    Base class for collision rules that add collision checks to the collision matrix.
    """

    buffer_zone_distance: float = field(default=0.05)
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = field(default=0.0)
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """

    added_collision_checks: set[CollisionCheck] = field(default_factory=set, init=False)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        """
        Returns True if the rule configures collision distances for the given body.
        """
        return (
            CollisionCheck(body_a, body_b) in self.added_collision_checks
            or CollisionCheck(body_b, body_a) in self.added_collision_checks
        )

    def buffer_zone_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured buffer-zone distance for the body or None if not applicable.
        """
        return self.buffer_zone_distance if self.applies_to(body_a, body_b) else None

    def violated_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured violated distance for the body or None if not applicable.
        """
        return self.violated_distance if self.applies_to(body_a, body_b) else None

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.add_collision_checks(self.added_collision_checks)


@dataclass
class AllowCollisionRule(CollisionRule, ABC):
    """
    Base class for collision rules that remove collision checks from the collision matrix.
    """

    allowed_collision_pairs: set[CollisionCheck] = field(
        default_factory=set, init=False
    )
    """
    Set of collision checks that are allowed to occur.
    """
    allowed_collision_bodies: set[Body] = field(default_factory=set, init=False)
    """
    Set of bodies that are allowed to collide.
    """

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.remove_collision_checks(self.allowed_collision_pairs)
        collision_matrix.collision_checks = {
            collision_check
            for collision_check in collision_matrix.collision_checks
            if collision_check.body_a not in self.allowed_collision_bodies
            and collision_check.body_b not in self.allowed_collision_bodies
        }

    def update(self, world: World):
        if not self.is_up_to_date(world):
            self.allowed_collision_pairs = set()
            self.allowed_collision_bodies = set()
        super().update(world)


@dataclass
class AvoidCollisionBetweenGroups(AvoidCollisionRule):
    """
    Adds collision checks between all pairs of bodies in the given groups to the collision matrix.
    """

    body_group_a: List[Body] = field(default_factory=list)
    """
    List of bodies that should not collide with bodies in `body_group_b`.
    """
    body_group_b: List[Body] = field(default_factory=list)
    """
    List of bodies that should not collide with bodies in `body_group_a`.
    """

    def _update(self, world: World):
        self.added_collision_checks = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                self.added_collision_checks.add(collision_check)


@dataclass
class AvoidAllCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the world managed by the rule to the collision matrix.
    """

    def _update(self, world: World):
        self.added_collision_checks = set()
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            self.added_collision_checks.add(collision_check)


@dataclass
class AvoidExternalCollisions(AvoidCollisionRule, SubclassJSONSerializer):
    """
    Adds collision checks between all bodies managed by the rule and all bodies that do not belong to the robot.
    that are not managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """
    body_subset: set[Body] | None = field(default=None)
    """
    A subset of bodies managed by the rule. 
    All of them must belong to `robot`.
    If None, all robot bodies are used.
    """

    def _update(self, world: World):
        self.added_collision_checks = set()
        if self.body_subset is None:
            body_subset = set(self.robot.bodies_with_collision)
        else:
            body_subset = self.body_subset
        external_bodies = set(world.bodies_with_collision) - set(body_subset)
        for body_a, body_b in product(body_subset, external_bodies):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            self.added_collision_checks.add(collision_check)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "robot": to_json(self.robot.id),
            "body_subset": to_json(self.body_subset),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        robot = tracker.get_world_entity_with_id(id=from_json(data["robot"]))
        return cls(robot=robot, body_subset=from_json(data["body_subset"], **kwargs))


@dataclass
class AvoidSelfCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the robot managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """

    def _update(self, world: World):
        self.added_collision_checks = set(
            CollisionCheck.create_and_validate(
                body_a, body_b, distance=self.buffer_zone_distance
            )
            for body_a, body_b in combinations(self.robot.bodies_with_collision, 2)
        )


@dataclass
class AllowAllCollisions(AllowCollisionRule):
    """
    Removed all collision checks from the collision matrix.
    """

    def _update(self, world: World):
        self.allowed_collision_bodies = set(world.bodies_with_collision)


@dataclass
class AllowCollisionForBodies(AllowCollisionRule):
    """
    Removes all collision checks that include one of the given bodies from the collision matrix.
    """

    allowed_collision_bodies: set[Body] = field(default_factory=set)
    """
    The set of bodies for which all collisions should be allowed.
    """

    def _update(self, world: World): ...


@dataclass
class AllowCollisionBetweenGroups(AllowCollisionRule):
    """
    Allows collision checks between two groups of bodies.
    """

    body_group_a: List[Body] = field(default_factory=list)
    """
    The first group of bodies.
    """
    body_group_b: List[Body] = field(default_factory=list)
    """
    The second group of bodies.
    """

    def _update(self, world: World):
        self.allowed_collision_pairs = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b
                )
                self.allowed_collision_pairs.add(collision_check)


@dataclass
class AllowNonRobotCollisions(AllowCollisionRule):
    """
    Allows collision checks between all bodies that do not belong to any robot.
    """

    def _update(self, world: World):
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: set[Body] = {
            body
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collision
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: set[Body] = set(world.bodies_with_collision) - robot_bodies
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations(non_robot_bodies, 2):
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a=a, body_b=b, distance=0)
            )


@dataclass
class AllowSelfCollisions(AllowCollisionRule):
    """
    Allows collision checks between all body pairs of the robot managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot for which self-collisions should be allowed.
    """

    def _update(self, world: World):
        self.allowed_collision_pairs = set(
            CollisionCheck.create_and_validate(body_a, body_b)
            for body_a, body_b in combinations(self.robot.bodies_with_collision, 2)
        )


@dataclass
class AllowDefaultInCollision(AllowCollisionRule):
    """
    Allows collision between robot bodies that are in collision if the robot is in its default state.
    The default state is defined as all degrees of freedom being positioned in the center of their limits,
    or zero if they have no limits.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """
    bodies: list[Body] = field(kw_only=True)
    """
    Bodies to check. 
    .. warning: must be a subset of the robot bodies
    """
    collision_threshold: float = 0.0
    """
    The distance used for collision checking to decide if a robot is in collision by default.
    """

    def _update(self, world: World):
        with world.reset_state_context():
            self.set_robot_to_default_state(world)
            collision_matrix = CollisionMatrix()
            rule = AvoidCollisionBetweenGroups(
                buffer_zone_distance=self.collision_threshold,
                body_group_a=self.bodies,
                body_group_b=self.bodies,
            )
            rule.update(world)
            rule.apply_to_collision_matrix(collision_matrix)
            closest_points = (
                world.collision_manager.collision_detector.check_collisions(
                    collision_matrix
                )
            )
            self.allowed_collision_pairs = {
                CollisionCheck(closest_point.body_a, closest_point.body_b)
                for closest_point in closest_points.contacts
            }

    def set_robot_to_default_state(self, world: World):
        for degree_of_freedom in self.robot.degrees_of_freedom_with_hardware_interface:
            if degree_of_freedom.has_position_limits():
                lower_limit = degree_of_freedom.limits.lower.position
                upper_limit = degree_of_freedom.limits.upper.position
                world.state[degree_of_freedom.id].position = (
                    lower_limit + upper_limit / 2
                )
            else:
                world.state[degree_of_freedom.id].position = 0
        world.notify_state_change()


@dataclass
class AllowAlwaysInCollision(AllowCollisionRule):
    """
    Allows collision between robot bodies that are in collision always.
    This is computed by placing the robot in random states and checking if the robot is in collision in
    `almost_percentage` of the cases.
    .. note: This rule is expensive and should be used with caution.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """
    collision_checks: set[CollisionCheck] = field(default_factory=set)
    """
    The collision checks to use for collision checking.
    Allows you to prefilter the collision checks to perform.
    """
    distance_threshold_always: float = 0.005
    """
    The distance used for collision checking to decide if a robot is in collision always.
    """
    number_of_tries: int = 200
    """
    The number of tries to perform to determine if the robot is in collision always.
    """
    almost_percentage: float = 0.95
    """
    The percentage of tries that must result in collision for the rule to allow collision.
    """

    def _update(self, world: World):
        collision_matrix = CollisionMatrix()
        collision_matrix.collision_checks = self.collision_checks
        for collision_check in self.collision_checks:
            collision_check.distance = self.distance_threshold_always
        with world.reset_state_context():
            counts: dict[CollisionCheck, int] = defaultdict(int)
            for try_id in range(int(self.number_of_tries)):
                self.set_robot_to_rnd_state(world)
                closest_points = (
                    world.collision_manager.collision_detector.check_collisions(
                        collision_matrix
                    )
                )
                for closest_point in closest_points.contacts:
                    collision_check = CollisionCheck(
                        closest_point.body_a, closest_point.body_b
                    )
                    counts[collision_check] += 1
            for collision_check, count in counts.items():
                if count > self.number_of_tries * self.almost_percentage:
                    self.allowed_collision_pairs.add(collision_check)

    def set_robot_to_rnd_state(self, world: World):
        for degree_of_freedom in self.robot.degrees_of_freedom_with_hardware_interface:
            if degree_of_freedom.has_position_limits():
                lower_limit = degree_of_freedom.limits.lower.position
                upper_limit = degree_of_freedom.limits.upper.position
                rnd_position = (
                    np.random.random() * (upper_limit - lower_limit)
                ) + lower_limit
            else:
                rnd_position = np.random.random() * np.pi * 2
            world.state[degree_of_freedom.id].position = rnd_position
        world.notify_state_change()


@dataclass
class AllowNeverInCollision(AllowCollisionRule):
    """
    Allows collision between robot bodies that are never in collision.
    This is computed by placing the robot in random states and checking if the distance between any pair is always
    above a certain threshold.
    .. note: This rule is expensive and should be used with caution.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """
    collision_checks: set[CollisionCheck] = field(default_factory=set)
    """
    The collision checks to perform.
    Allows you to prefilter the collision checks to perform.
    """
    distance_threshold_max: float = 0.05
    """
    The maximum distance threshold for a pair to be considered as potentially never in collision.
    """
    distance_threshold_min: float = -0.02
    """
    If a pair is below this distance even once, they are not allowed.
    """
    distance_threshold_range: float = 0.05
    """
    The range of the distance threshold.
    """
    distance_threshold_zero: float = 0.0
    """
    The zero distance threshold.
    """
    number_of_tries: int = 10_000
    """
    The number of random states to check.
    """
    progress_callback: Callable[[int, str], None] | None = field(default=None)
    """
    A callback function to report progress.
    .. note: This callback is optional and can be used to monitor the progress of the collision check.
    """

    def __post_init__(self):
        if self.progress_callback is None:
            self.progress_callback = lambda value, text: None

    def _update(self, world: World):
        collision_matrix = CollisionMatrix()
        collision_matrix.collision_checks = self.collision_checks
        for collision_check in self.collision_checks:
            collision_check.distance = self.distance_threshold_max * 1.1
        with world.reset_state_context():
            one_percent = self.number_of_tries // 100
            distances_cache: dict[CollisionCheck, list[float]] = defaultdict(list)
            for try_id in range(int(self.number_of_tries)):
                self.set_robot_to_rnd_state(world)
                closest_points = (
                    world.collision_manager.collision_detector.check_collisions(
                        collision_matrix
                    )
                )
                self._update_collision_matrix(
                    closest_points=closest_points,
                    collision_matrix=collision_matrix,
                    distance_ranges=distances_cache,
                )
                if try_id % one_percent == 0:
                    self.progress_callback(try_id // one_percent, "checking collisions")
            self._allow_close_but_never_pairs(distances_cache)
            self._allow_never_pairs(collision_matrix, distances_cache)

    def _update_collision_matrix(
        self,
        closest_points: CollisionCheckingResult,
        collision_matrix: CollisionMatrix,
        distance_ranges: dict[CollisionCheck, list[float]],
    ):
        contact_keys = set()
        for contact in closest_points.contacts:
            collision_check = CollisionCheck(contact.body_a, contact.body_b)
            contact_keys.add(collision_check)
            distance_ranges[collision_check].append(contact.distance)
            if contact.distance < self.distance_threshold_min:
                collision_matrix.collision_checks.discard(collision_check)
                self.robot._world.collision_manager.collision_detector.reset_cache()
                del distance_ranges[collision_check]

    def _allow_close_but_never_pairs(
        self, distances_cache: dict[CollisionCheck, list[float]]
    ):
        for key, distances in list(distances_cache.items()):
            mean = np.mean(distances)
            std = np.std(distances)
            # 99.7% is close enough
            if (
                mean - 3 * std > self.distance_threshold_zero
                or mean + 3 * std < self.distance_threshold_range
            ):
                self.allowed_collision_pairs.add(key)
                del distances_cache[key]

    def _allow_never_pairs(
        self,
        collision_matrix: CollisionMatrix,
        distances_cache: dict[CollisionCheck, list[float]],
    ):
        never = collision_matrix.collision_checks - set(distances_cache.keys())
        self.allowed_collision_pairs.update(never)

    def set_robot_to_rnd_state(self, world: World):
        for degree_of_freedom in self.robot.degrees_of_freedom_with_hardware_interface:
            if degree_of_freedom.has_position_limits():
                lower_limit = degree_of_freedom.limits.lower.position
                upper_limit = degree_of_freedom.limits.upper.position
                rnd_position = (
                    np.random.random() * (upper_limit - lower_limit)
                ) + lower_limit
            else:
                rnd_position = np.random.random() * np.pi * 2
            world.state[degree_of_freedom.id].position = rnd_position
        world.notify_state_change()


@dataclass
class AllowCollisionForAdjacentPairs(AllowCollisionRule):
    """
    Allow collision between body pairs of the whole world model that are connected by a chain that has no controllable connection.
    Adjacent bodies in the kinematic chain that are not separated by a controlled joint are allowed to collide.
    """

    def _update(self, world: World):
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            if self._no_controlled_connection_between_bodies(world, body_a, body_b):
                self.allowed_collision_pairs.add(
                    CollisionCheck.create_and_validate(body_a, body_b)
                )

    def _no_controlled_connection_between_bodies(
        self, world: World, body_a: Body, body_b: Body
    ) -> bool:
        return (
            not world.is_controlled_connection_in_chain(body_a, body_b)
            or body_a == body_b.parent_kinematic_structure_entity
            or body_b == body_a.parent_kinematic_structure_entity
        )


@dataclass
class SelfCollisionMatrixRule(AllowCollisionRule, SubclassJSONSerializer):
    """
    Used to load and manage collision matrices, often represented as SRDF files.
    Allows specifying which body pairs or bodies are exempt from collision checking.
    Has functionality to safe this matrix to an SRDF file.
    `compute_self_collision_matrix` can be used to compute a self-collision matrix for a given robot.
    .. note: use from_collision_srdf to load a collision matrix from an SRDF file, unless you want to compute and safe it.
    """

    SRDF_DISABLE_ALL_COLLISIONS: ClassVar[str] = "disable_all_collisions"
    SRDF_DISABLE_SELF_COLLISION: ClassVar[str] = "disable_self_collision"
    SRDF_MOVEIT_DISABLE_COLLISIONS: ClassVar[str] = "disable_collisions"

    def update(self, world: World):
        """
        Updates the rule based on the current world state.
        """
        ...

    def _update(self, world: World):
        """
        Internal update method.
        """
        ...

    @classmethod
    def from_collision_srdf(cls, file_path: str, world: World) -> Self:
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        self = cls()

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == self.SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = world.get_body_by_name(c.attrib["link"])
            self.allowed_collision_bodies.add(body)

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag
            in {self.SRDF_MOVEIT_DISABLE_COLLISIONS, self.SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (body_a := world.get_body_by_name(child.attrib["link1"])).has_collision()
            and (
                body_b := world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            if body_a == body_b:
                continue
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a, body_b)
            )
        return self

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "allowed_body_ids": to_json(
                {body.id for body in self.allowed_collision_bodies}
            ),
            "allowed_collision_pairs": to_json(self.allowed_collision_pairs),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        allowed_body_ids = from_json(data["allowed_body_ids"], **kwargs)
        allowed_bodies = {
            tracker.get_world_entity_with_id(id=_id) for _id in allowed_body_ids
        }
        self = cls()
        self.allowed_collision_bodies = allowed_bodies
        self.allowed_collision_pairs = set(
            from_json(data["allowed_collision_pairs"], **kwargs)
        )

        return self

    def compute_self_collision_matrix(
        self,
        robot: AbstractRobot,
        distance_threshold_zero: float = 0.0,
        distance_threshold_always: float = 0.005,
        distance_threshold_never_max: float = 0.05,
        distance_threshold_never_min: float = -0.02,
        distance_threshold_never_range: float = 0.05,
        distance_threshold_never_zero: float = 0.0,
        number_of_tries_always: int = 200,
        almost_percentage: float = 0.95,
        number_of_tries_never: int = 10000,
        progress_callback: Callable[[int, str], None] | None = None,
    ):
        """
        Computes a self-collision matrix for the given robot by applying various rules.

        This method systematically identifies body pairs that should be allowed to collide by
        checking for adjacent links, default collisions, always-in-collision pairs, and
        never-in-collision pairs.

        :param robot: The robot for which to compute the matrix.
        :param distance_threshold_zero: Threshold for default collision check.
        :param distance_threshold_always: Threshold for always-in-collision check.
        :param distance_threshold_never_max: Maximum threshold for never-in-collision check.
        :param distance_threshold_never_min: Minimum threshold for never-in-collision check.
        :param distance_threshold_never_range: Range threshold for never-in-collision check.
        :param distance_threshold_never_zero: Zero threshold for never-in-collision check.
        :param number_of_tries_always: Number of random states to check for always-in-collision.
        :param almost_percentage: Percentage for always-in-collision rule.
        :param number_of_tries_never: Number of random states to check for never-in-collision.
        :param progress_callback: Callback for progress updates.
        """
        self.allowed_collision_pairs = set()
        np.random.seed(1337)
        # %% 0. GENERATE ALL POSSIBLE LINK PAIRS
        collision_matrix = CollisionMatrix()
        rule = AvoidSelfCollisions(robot=robot)
        rule.update(robot._world)
        rule.apply_to_collision_matrix(collision_matrix)

        rule = AllowCollisionForBodies(
            allowed_collision_bodies=self.allowed_collision_bodies
        )
        rule.apply_to_collision_matrix(collision_matrix)

        # %%
        rule = AllowCollisionForAdjacentPairs()
        rule.update(robot._world)
        rule.apply_to_collision_matrix(collision_matrix)
        self.allowed_collision_pairs.update(rule.allowed_collision_pairs)

        # %%
        rule = AllowDefaultInCollision(
            robot=robot,
            bodies=robot.bodies_with_collision,
            collision_threshold=distance_threshold_zero,
        )
        rule.update(robot._world)
        rule.apply_to_collision_matrix(collision_matrix)
        self.allowed_collision_pairs.update(rule.allowed_collision_pairs)

        # %%
        rule = AllowAlwaysInCollision(
            robot=robot,
            distance_threshold_always=distance_threshold_always,
            number_of_tries=number_of_tries_always,
            almost_percentage=almost_percentage,
            collision_checks=collision_matrix.collision_checks,
        )
        rule.update(robot._world)
        rule.apply_to_collision_matrix(collision_matrix)
        self.allowed_collision_pairs.update(rule.allowed_collision_pairs)

        # %%
        rule = AllowNeverInCollision(
            robot=robot,
            distance_threshold_range=distance_threshold_never_range,
            distance_threshold_zero=distance_threshold_never_zero,
            number_of_tries=number_of_tries_never,
            distance_threshold_min=distance_threshold_never_min,
            distance_threshold_max=distance_threshold_never_max,
            collision_checks=collision_matrix.collision_checks,
            progress_callback=progress_callback,
        )
        rule.update(robot._world)
        rule.apply_to_collision_matrix(collision_matrix)
        self.allowed_collision_pairs.update(rule.allowed_collision_pairs)

        self.allowed_collision_pairs = {
            collision_check
            for collision_check in self.allowed_collision_pairs
            if collision_check.body_a not in self.allowed_collision_bodies
            and collision_check.body_b not in self.allowed_collision_bodies
        }

    def save_self_collision_matrix(
        self,
        robot_name: str,
        file_name: str,
    ):
        """
        Saves the current self-collision matrix to an SRDF file.

        :param robot_name: The name of the robot.
        :param file_name: The path to the file where the matrix should be saved.
        """
        # Create the root element
        root = etree.Element("robot")
        root.set("name", robot_name)

        # %% disabled links
        for body in sorted(self.allowed_collision_bodies, key=lambda b: b.name.name):
            child = etree.SubElement(root, self.SRDF_DISABLE_ALL_COLLISIONS)
            child.set("link", body.name.name)

        # %% self collision matrix
        for collision_check in sorted(
            self.allowed_collision_pairs,
            key=lambda c: f"{c.body_a.name.name}{c.body_b.name.name}",
        ):
            body_a, body_b = collision_check.body_a, collision_check.body_b
            child = etree.SubElement(root, self.SRDF_DISABLE_SELF_COLLISION)
            child.set("link1", body_a.name.name)
            child.set("link2", body_b.name.name)
            child.set("reason", "Unknown")

        # Create the XML tree
        tree = etree.ElementTree(root)

        tree.write(
            file_name,
            pretty_print=True,
            xml_declaration=True,
            encoding=tree.docinfo.encoding,
        )
