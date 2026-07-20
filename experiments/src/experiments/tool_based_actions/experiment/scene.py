"""
Seeded random scene sampling for the tool-based action experiment.

Targets are spawned on the support surfaces named in the configuration, at random
positions and with random orientations. The same seed always yields the same scene.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from semantic_digital_twin.world import World
from typing_extensions import List, Optional, Tuple

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

    The same seed always yields the same placements.
    """

    surfaces: List[SpawnSurface]
    """
    The surfaces placements are sampled on.
    """

    clearance: float
    """
    Minimum XY distance in meters between two placements.
    """

    seed: int
    """
    Seed of the random number generator.
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

    def sample_target_count(self, minimum: int, maximum: int) -> int:
        """
        :param minimum: Smallest allowed number of targets.
        :param maximum: Largest allowed number of targets.
        :return: A reproducible number of targets in ``[minimum, maximum]``.
        """
        return self._random_generator().randint(minimum, maximum)

    def sample_placements(self, count: int, name_prefix: str) -> List[TargetPlacement]:
        """
        :param count: Number of placements to sample.
        :param name_prefix: Prefix of the generated target names.
        :return: ``count`` collision-free placements spread over the surfaces.
        :raises SpawnRegionExhausted: If the surfaces provably cannot hold the
            placements, or sampling stays unsuccessful within the restart budget.
        """
        if self._capacity() < count:
            raise SpawnRegionExhausted(self.surfaces, self.clearance, count)
        generator = self._random_generator()
        for _ in range(self.maximum_scene_restarts):
            placements = self._sample_scene(generator, count, name_prefix)
            if placements is not None:
                return placements
        raise SpawnRegionExhausted(self.surfaces, self.clearance, count)

    def _capacity(self) -> int:
        """
        :return: A conservative number of targets that provably fit on all surfaces
            together.
        """
        return sum(
            surface.region.grid_capacity(self.clearance) for surface in self.surfaces
        )

    def _random_generator(self) -> random.Random:
        """
        :return: A fresh generator so every sampling call is independent of call
            order.
        """
        return random.Random(self.seed)

    def _sample_scene(
        self, generator: random.Random, count: int, name_prefix: str
    ) -> Optional[List[TargetPlacement]]:
        """
        :param generator: The random number generator to draw from.
        :param count: Number of placements to sample.
        :param name_prefix: Prefix of the generated target names.
        :return: A full set of collision-free placements, or None if this round ran
            into a dead end.
        """
        placements: List[TargetPlacement] = []
        for target_index in range(count):
            placement = self._sample_free_placement(
                generator, placements, f"{name_prefix}_{target_index}"
            )
            if placement is None:
                return None
            placements.append(placement)
        return placements

    def _sample_free_placement(
        self,
        generator: random.Random,
        existing: List[TargetPlacement],
        name: str,
    ) -> Optional[TargetPlacement]:
        """
        :param generator: The random number generator to draw from.
        :param existing: Placements the new one must keep clear of.
        :param name: Name of the new placement.
        :return: A placement respecting the clearance to all existing ones, or None
            if no free spot was found within the attempt budget.
        """
        for _ in range(self.maximum_attempts_per_target):
            surface = generator.choice(self.surfaces)
            candidate = TargetPlacement(
                name=name,
                surface_name=surface.name,
                x=generator.uniform(surface.region.minimum_x, surface.region.maximum_x),
                y=generator.uniform(surface.region.minimum_y, surface.region.maximum_y),
                z=surface.region.height,
                yaw=generator.uniform(0.0, 2.0 * math.pi),
            )
            if all(
                candidate.distance_to(placement) >= self.clearance
                for placement in existing
            ):
                return candidate
        return None
