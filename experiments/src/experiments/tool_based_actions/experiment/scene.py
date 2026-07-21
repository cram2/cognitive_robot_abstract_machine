"""
Seeded random scene sampling for the tool-based action experiment.

Targets are spawned on the support surfaces named in the configuration, at random
positions, sizes, and orientations. Placements keep their whole footprint on the
surface, keep clear of each other and of every other body in the world, and the same
seed always yields the same scene.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from semantic_digital_twin.world import World
from typing_extensions import ClassVar, List, Optional, Set, Tuple

from experiments.tool_based_actions.experiment.configuration import SpawnRegion


class MissingSpawnSurfaces(Exception):
    """
    Raised when none of the configured spawn surfaces exist in the world.
    """

    def __init__(self, surface_names: Tuple[str, ...]):
        super().__init__(
            f"None of the configured spawn surfaces {surface_names} exist in the "
            "world. Check the surface names against the environment."
        )


class SpawnRegionExhausted(Exception):
    """
    Raised when the spawn surfaces cannot hold the requested targets at the requested
    clearance.
    """

    def __init__(self, surfaces: List[SpawnSurface], clearance: float, count: int):
        surface_names = [surface.name for surface in surfaces]
        super().__init__(
            f"Could not place {count} targets with clearance {clearance} m on the "
            f"surfaces {surface_names}."
        )


@dataclass(frozen=True)
class SpawnSurface:
    """
    A named support surface targets can be spawned on.
    """

    name: str
    """
    Name of the surface body in the world.
    """

    region: SpawnRegion
    """
    The spawnable rectangle on top of the surface, in the world frame.
    """


@dataclass(frozen=True)
class ObjectFootprint:
    """
    The circular XY footprint of a spawnable object, over its allowed scales.
    """

    base_radius: float
    """
    Footprint radius in meters at scale 1.0.
    """

    scale_choices: Tuple[float, ...]
    """
    Uniform scale factors an object may be spawned with.
    """

    safety_factor: float = 1.0
    """
    Factor the radius is inflated by to absorb mesh irregularities.
    """

    @classmethod
    def point(cls) -> ObjectFootprint:
        """
        :return: The footprint of a dimensionless target, e.g. a wiping pose.
        """
        return cls(base_radius=0.0, scale_choices=(1.0,))

    def radius_for_scale(self, scale: float) -> float:
        """
        :param scale: The uniform scale the object is spawned with.
        :return: The inflated footprint radius in meters at that scale.
        """
        return self.base_radius * scale * self.safety_factor

    def largest_radius(self) -> float:
        """
        :return: The inflated footprint radius in meters at the largest scale.
        """
        return self.radius_for_scale(max(self.scale_choices))


@dataclass(frozen=True)
class ObstacleBox:
    """
    The world-frame bounding box of a body placements must keep clear of.
    """

    vertical_margin: ClassVar[float] = 0.03
    """
    Distance in meters a placement may sit below an obstacle's bottom before the
    obstacle stops blocking it.
    """

    top_epsilon: ClassVar[float] = 0.01
    """
    Band in meters below an obstacle's top within which a placement counts as resting on
    the obstacle instead of colliding with it.
    """

    name: str
    """
    Name of the obstacle body in the world.
    """

    minimum_x: float
    """
    Lower X bound of the box in the world frame.
    """

    maximum_x: float
    """
    Upper X bound of the box in the world frame.
    """

    minimum_y: float
    """
    Lower Y bound of the box in the world frame.
    """

    maximum_y: float
    """
    Upper Y bound of the box in the world frame.
    """

    minimum_z: float
    """
    Lower Z bound of the box in the world frame.
    """

    maximum_z: float
    """
    Upper Z bound of the box in the world frame.
    """

    def blocks(self, x: float, y: float, z: float, radius: float) -> bool:
        """
        Decide whether a placement collides with this obstacle.

        Placements below the obstacle or resting on its top do not collide; anything
        whose footprint disc overlaps the box within its vertical band does.

        :param x: X coordinate of the placement in the world frame.
        :param y: Y coordinate of the placement in the world frame.
        :param z: Z coordinate of the placement in the world frame.
        :param radius: Footprint radius of the placement in meters.
        :return: True if the placement collides with this obstacle.
        """
        if z < self.minimum_z - self.vertical_margin:
            return False
        if z >= self.maximum_z - self.top_epsilon:
            return False
        return (
            self.minimum_x - radius <= x <= self.maximum_x + radius
            and self.minimum_y - radius <= y <= self.maximum_y + radius
        )


def discover_obstacles(
    world: World, excluded_body_names: Set[str]
) -> List[ObstacleBox]:
    """
    Measure every collidable body in the world as a placement obstacle.

    :param world: The world to search in.
    :param excluded_body_names: Names of bodies that must not act as obstacles, e.g. the
        robot's.
    :return: One obstacle box per collidable, non-excluded body.
    """
    obstacles = []
    for body in world.bodies:
        name = body.name.name
        if name in excluded_body_names:
            continue
        if not body.collision.shapes:
            continue
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        obstacles.append(
            ObstacleBox(
                name=name,
                minimum_x=bounding_box.min_x,
                maximum_x=bounding_box.max_x,
                minimum_y=bounding_box.min_y,
                maximum_y=bounding_box.max_y,
                minimum_z=bounding_box.min_z,
                maximum_z=bounding_box.max_z,
            )
        )
    return obstacles


def discover_spawn_surfaces(
    world: World,
    surface_names: Tuple[str, ...],
    margin: float,
    height_offset: float,
) -> List[SpawnSurface]:
    """
    Measure the configured support surfaces in the world.

    :param world: The world to search in.
    :param surface_names: Names of the surface bodies to use.
    :param margin: Distance in meters kept from every surface edge.
    :param height_offset: Height in meters above the surface top at which targets are
        spawned.
    :return: One spawn surface per found name.
    :raises MissingSpawnSurfaces: If none of the names exist in the world.
    """
    surfaces = []
    body_names = {body.name.name for body in world.bodies}
    for surface_name in surface_names:
        if surface_name not in body_names:
            continue
        body = world.get_body_by_name(surface_name)
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        surfaces.append(
            SpawnSurface(
                name=surface_name,
                region=SpawnRegion(
                    minimum_x=bounding_box.min_x + margin,
                    maximum_x=bounding_box.max_x - margin,
                    minimum_y=bounding_box.min_y + margin,
                    maximum_y=bounding_box.max_y - margin,
                    height=bounding_box.max_z + height_offset,
                ),
            )
        )
    if not surfaces:
        raise MissingSpawnSurfaces(surface_names)
    return surfaces


@dataclass(frozen=True)
class TargetPlacement:
    """
    A named spawn location and orientation of one trial target.
    """

    name: str
    """
    Unique name of the target within its trial.
    """

    surface_name: str
    """
    Name of the surface the target is spawned on.
    """

    x: float
    """
    X coordinate of the target in the world frame.
    """

    y: float
    """
    Y coordinate of the target in the world frame.
    """

    z: float
    """
    Z coordinate of the target in the world frame.
    """

    yaw: float
    """
    Rotation in radians of the target around the world Z axis.
    """

    scale: float = 1.0
    """
    Uniform scale factor the target is spawned with.
    """

    footprint_radius: float = 0.0
    """
    Inflated footprint radius in meters of the target at its scale.
    """

    def distance_to(self, other: TargetPlacement) -> float:
        """
        :param other: The placement to measure against.
        :return: The XY distance in meters between the two placements.
        """
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class SceneSampler:
    """
    Samples reproducible, collision-free target placements on the spawn surfaces.

    Placements keep their whole footprint on the surface, keep clear of each other and
    of the given obstacles, and the same seed always yields the same placements.
    """

    surfaces: List[SpawnSurface]
    """
    The surfaces placements are sampled on.
    """

    clearance: float
    """
    Minimum center distance in meters between two placements.
    """

    seed: int
    """
    Seed of the random number generator.
    """

    footprint: ObjectFootprint = field(default_factory=ObjectFootprint.point)
    """
    Footprint of the spawned objects, driving their scales and required space.
    """

    obstacles: List[ObstacleBox] = field(default_factory=list)
    """
    Bodies placements must keep clear of.
    """

    footprint_clearance: float = 0.03
    """
    Minimum free gap in meters between the footprints of two placements.
    """

    maximum_spawn_height: float = math.inf
    """
    Highest surface top in meters, in the world frame, placements are sampled on.
    """

    maximum_attempts_per_target: int = 100
    """
    Number of rejection-sampling attempts per target before the whole scene is resampled
    from scratch.
    """

    maximum_scene_restarts: int = 20
    """
    Number of from-scratch resampling rounds before giving up, so early placements that
    block all remaining space do not fail an otherwise feasible scene.
    """

    def target_count(
        self, targets_per_square_meter: float, minimum: int, maximum: int
    ) -> int:
        """
        :param targets_per_square_meter: Desired target density on the surfaces.
        :param minimum: Smallest allowed number of targets.
        :param maximum: Largest allowed number of targets.
        :return: The density-based number of targets, clamped to
            ``[minimum, maximum]``.
        """
        area = sum(surface.region.area() for surface in self._usable_surfaces())
        return max(minimum, min(maximum, round(area * targets_per_square_meter)))

    def sample_placements(
        self, count: int, name_prefix: str, minimum_count: Optional[int] = None
    ) -> List[TargetPlacement]:
        """
        Sample up to ``count`` collision-free placements spread over the surfaces.

        :param count: Desired number of placements.
        :param name_prefix: Prefix of the generated target names.
        :param minimum_count: Smallest acceptable number of placements when the surfaces
            cannot hold all ``count``, or None to require all of them.
        :return: The sampled placements, at least ``minimum_count`` many.
        :raises SpawnRegionExhausted: If the surfaces provably cannot hold the required
            placements, or sampling stays unsuccessful within the restart budget.
        """
        required_count = count if minimum_count is None else minimum_count
        surfaces = self._usable_surfaces()
        if not surfaces or self._capacity(surfaces) < required_count:
            raise SpawnRegionExhausted(self.surfaces, self.clearance, required_count)
        generator = self._random_generator()
        best_placements: List[TargetPlacement] = []
        for _ in range(self.maximum_scene_restarts):
            placements = self._sample_scene(generator, surfaces, count, name_prefix)
            if len(placements) == count:
                return placements
            if len(placements) > len(best_placements):
                best_placements = placements
        if len(best_placements) >= required_count:
            return best_placements
        raise SpawnRegionExhausted(self.surfaces, self.clearance, required_count)

    def _usable_surfaces(self) -> List[SpawnSurface]:
        """
        :return: The surfaces whose top lies below the maximum spawn height.
        """
        return [
            surface
            for surface in self.surfaces
            if surface.region.height <= self.maximum_spawn_height
        ]

    def _capacity(self, surfaces: List[SpawnSurface]) -> int:
        """
        :param surfaces: The surfaces available for sampling.
        :return: A conservative number of targets that provably fit on all surfaces
            together, assuming every target has the largest footprint.
        """
        largest_radius = self.footprint.largest_radius()
        cell = max(
            self.clearance, 2.0 * largest_radius + self.footprint_clearance
        )
        capacity = 0
        for surface in surfaces:
            region = surface.region.inset(largest_radius)
            if region.is_empty():
                continue
            capacity += region.grid_capacity(cell)
        return capacity

    def _random_generator(self) -> random.Random:
        """
        :return: A fresh generator so every sampling call is independent of call
            order.
        """
        return random.Random(self.seed)

    def _sample_scene(
        self,
        generator: random.Random,
        surfaces: List[SpawnSurface],
        count: int,
        name_prefix: str,
    ) -> List[TargetPlacement]:
        """
        :param generator: The random number generator to draw from.
        :param surfaces: The surfaces available for sampling.
        :param count: Desired number of placements.
        :param name_prefix: Prefix of the generated target names.
        :return: The collision-free placements of this round, ending at the first
            target no free spot was found for.
        """
        placements: List[TargetPlacement] = []
        for target_index in range(count):
            placement = self._sample_free_placement(
                generator, surfaces, placements, f"{name_prefix}_{target_index}"
            )
            if placement is None:
                return placements
            placements.append(placement)
        return placements

    def _sample_free_placement(
        self,
        generator: random.Random,
        surfaces: List[SpawnSurface],
        existing: List[TargetPlacement],
        name: str,
    ) -> Optional[TargetPlacement]:
        """
        :param generator: The random number generator to draw from.
        :param surfaces: The surfaces available for sampling.
        :param existing: Placements the new one must keep clear of.
        :param name: Name of the new placement.
        :return: A placement keeping its footprint on the surface and clear of all
            obstacles and existing placements, or None if no free spot was found
            within the attempt budget.
        """
        for _ in range(self.maximum_attempts_per_target):
            surface = generator.choice(surfaces)
            scale = generator.choice(self.footprint.scale_choices)
            radius = self.footprint.radius_for_scale(scale)
            region = surface.region.inset(radius)
            if region.is_empty():
                continue
            candidate = TargetPlacement(
                name=name,
                surface_name=surface.name,
                x=generator.uniform(region.minimum_x, region.maximum_x),
                y=generator.uniform(region.minimum_y, region.maximum_y),
                z=region.height,
                yaw=generator.uniform(0.0, 2.0 * math.pi),
                scale=scale,
                footprint_radius=radius,
            )
            if any(
                obstacle.blocks(candidate.x, candidate.y, candidate.z, radius)
                for obstacle in self.obstacles
            ):
                continue
            if all(
                candidate.distance_to(placement)
                >= self._required_distance(candidate, placement)
                for placement in existing
            ):
                return candidate
        return None

    def _required_distance(
        self, first: TargetPlacement, second: TargetPlacement
    ) -> float:
        """
        :param first: One placement.
        :param second: Another placement.
        :return: The minimum center distance in meters the two placements must keep.
        """
        return max(
            self.clearance,
            first.footprint_radius + second.footprint_radius + self.footprint_clearance,
        )
