"""
Region-based scene spawning for the tool-based action experiment.

The configured spawn surfaces are annotated as supporting surfaces
(:class:`~semantic_digital_twin.semantic_annotations.mixins.HasSupportingSurface`) and
target placements are drawn from their computed supporting-surface regions via
:meth:`~semantic_digital_twin.semantic_annotations.mixins.HasSupportingSurface.sample_points_from_surface`.
Sampled placements keep clear of the surface edges and of everything already placed on
the surface, and the same seed always yields the same scene.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, List, Optional

from experiments.tool_based_actions.experiment.exceptions import (
    InvalidSpawnRegion,
    MissingSpawnSurfaces,
    SpawnRegionExhausted,
)

if TYPE_CHECKING:
    from experiments.tool_based_actions.experiment.task_definitions import (
        ToolTaskDefinition,
    )


@dataclass(frozen=True)
class SpawnRegion:
    """
    An axis-aligned rectangle in the world's XY plane, used by the underspecified demos
    to bound free variables such as the robot's base pose.
    """

    minimum_x: float
    """
    Lower X bound of the region in the world frame.
    """

    maximum_x: float
    """
    Upper X bound of the region in the world frame.
    """

    minimum_y: float
    """
    Lower Y bound of the region in the world frame.
    """

    maximum_y: float
    """
    Upper Y bound of the region in the world frame.
    """

    height: float
    """
    Z coordinate of the region in the world frame.
    """

    def __post_init__(self) -> None:
        """
        :raises InvalidSpawnRegion: If the bounds contain no points.
        """
        if self.maximum_x <= self.minimum_x or self.maximum_y <= self.minimum_y:
            raise InvalidSpawnRegion(
                self.minimum_x, self.maximum_x, self.minimum_y, self.maximum_y
            )

    def contains(self, x: float, y: float) -> bool:
        """
        :param x: X coordinate in the world frame.
        :param y: Y coordinate in the world frame.
        :return: True if the point lies inside the region.
        """
        return (
            self.minimum_x <= x <= self.maximum_x
            and self.minimum_y <= y <= self.maximum_y
        )


@dataclass(frozen=True)
class PlanarPose:
    """
    A position and orientation in the world's XY plane, as plain floats.

    Kept separate from the symbolic spatial types so seeded sampling and result records
    stay value-comparable and serializable without a live world.
    """

    x: float
    """
    X coordinate in the world frame.
    """

    y: float
    """
    Y coordinate in the world frame.
    """

    yaw: float
    """
    Rotation in radians around the world Z axis.
    """

    def distance_to(self, other: PlanarPose) -> float:
        """
        :param other: The pose to measure against.
        :return: The XY distance in meters between the two poses.
        """
        return math.hypot(self.x - other.x, self.y - other.y)


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
    Name of the surface body the target is spawned on.
    """

    pose: PlanarPose
    """
    Position and orientation of the target in the world's XY plane.
    """

    z: float
    """
    Z coordinate of the target in the world frame.
    """

    scale: float = 1.0
    """
    Uniform scale factor the target is spawned with.
    """

    def distance_to(self, other: TargetPlacement) -> float:
        """
        :param other: The placement to measure against.
        :return: The XY distance in meters between the two placements.
        """
        return self.pose.distance_to(other.pose)


@dataclass(frozen=True)
class ExperimentTarget:
    """
    One spawned target of a trial, addressable by the tool action.
    """

    placement: TargetPlacement
    """
    The sampled placement the target was spawned at.
    """

    pose: Pose
    """
    Pose of the target in the world frame.
    """

    body: Optional[Body] = None
    """
    The spawned body, or None for targets that are pure poses (e.g. wiping patches).
    """


def annotate_spawn_surfaces(
    world: World, surface_names: List[str]
) -> List[HasSupportingSurface]:
    """
    Annotate the configured surface bodies as supporting surfaces and compute their
    spawnable regions.

    :param world: The world to search in.
    :param surface_names: Names of the surface bodies to use.
    :return: One supporting-surface annotation per found name whose region could be
        computed.
    :raises MissingSpawnSurfaces: If no surface yields a supporting-surface region.
    """
    surfaces = []
    body_names = {body.name.name for body in world.bodies}
    with world.modify_world():
        for surface_name in surface_names:
            if surface_name not in body_names:
                continue
            surface = Table(root=world.get_body_by_name(surface_name))
            world.add_semantic_annotations([surface])
            if surface.calculate_supporting_surface() is None:
                continue
            surfaces.append(surface)
    if not surfaces:
        raise MissingSpawnSurfaces(tuple(surface_names))
    return surfaces


@dataclass
class SceneSpawner:
    """
    Spawns reproducible trial targets on annotated supporting surfaces.

    Placements are drawn from the surfaces' supporting-surface regions, which keep
    clear of the surface edges and of everything already placed on the surface. The
    same seed always yields the same scene.

    .. note:: The region sampling draws from numpy's global random generator, so the
        spawner seeds it. Sampling is reproducible as long as no other numpy random
        draws interleave.
    """

    world: World
    """
    The world targets are spawned into.
    """

    surfaces: List[HasSupportingSurface]
    """
    The annotated surfaces targets are placed on.
    """

    seed: int
    """
    Seed that fixes the sampled scene.
    """

    scale_choices: List[float] = field(default_factory=lambda: [1.0])
    """
    Uniform scale factors a spawned target is randomly sized with.
    """

    edge_clearance: float = 0.15
    """
    Minimum distance in meters kept from every surface edge when placing.
    """

    _generator: random.Random = field(init=False)
    """
    Seeded generator for the surface, scale, and yaw draws.
    """

    def __post_init__(self) -> None:
        self._generator = random.Random(self.seed)
        np.random.seed(self.seed)

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
        area = sum(self._surface_area(surface) for surface in self.surfaces)
        return max(minimum, min(maximum, round(area * targets_per_square_meter)))

    def spawn_targets(
        self,
        definition: ToolTaskDefinition,
        count: int,
        minimum_count: int,
        name_prefix: str,
    ) -> List[ExperimentTarget]:
        """
        Spawn up to ``count`` targets of the given task, spread over the surfaces.

        :param definition: The task definition creating the targets.
        :param count: Desired number of targets.
        :param minimum_count: Smallest acceptable number of targets when the surfaces
            cannot hold all ``count``.
        :param name_prefix: Prefix of the generated target names.
        :return: The spawned targets, at least ``minimum_count`` many.
        :raises SpawnRegionExhausted: If the surfaces cannot hold ``minimum_count``
            targets.
        """
        targets = []
        for target_index in range(count):
            name = f"{name_prefix}_{target_index}"
            scale = self._generator.choice(self.scale_choices)
            annotation = definition.create_target(self.world, name, scale)
            placement = self._place(annotation, name, scale)
            if placement is None:
                if annotation is not None:
                    self._remove_target(annotation)
                break
            targets.append(
                ExperimentTarget(
                    placement=placement,
                    pose=self._placement_pose(placement),
                    body=annotation.root if annotation is not None else None,
                )
            )
        if len(targets) < minimum_count:
            raise SpawnRegionExhausted(self._surface_names(), minimum_count)
        return targets

    def _place(
        self, annotation: Optional[HasRootBody], name: str, scale: float
    ) -> Optional[TargetPlacement]:
        """
        Draw a placement from the surfaces' regions and move the target there.

        :param annotation: The target to place, or None for a pure pose target.
        :param name: Name of the placement.
        :param scale: Uniform scale factor the target was created with.
        :return: The placement, or None if no surface has free space left.
        """
        surfaces = self._generator.sample(self.surfaces, len(self.surfaces))
        for surface in surfaces:
            points = surface.sample_points_from_surface(
                body_to_sample_for=annotation, edge_clearance=self.edge_clearance
            )
            if not points:
                continue
            point = points[0]
            yaw = self._generator.uniform(0.0, 2.0 * math.pi)
            surface_T_target = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=point.x,
                y=point.y,
                z=point.z,
                yaw=yaw,
                reference_frame=point.reference_frame,
            )
            world_T_target = self.world.transform(surface_T_target, self.world.root)
            if annotation is not None:
                with self.world.modify_world():
                    annotation.root.parent_connection.origin = world_T_target
                    surface.add_object(annotation)
            return TargetPlacement(
                name=name,
                surface_name=surface.root.name.name,
                pose=self._world_planar_pose(world_T_target),
                z=float(world_T_target.z),
                scale=scale,
            )
        return None

    def _placement_pose(self, placement: TargetPlacement) -> Pose:
        """
        :param placement: The sampled placement.
        :return: The placement as a pose in the world frame.
        """
        return Pose.from_xyz_rpy(
            placement.pose.x,
            placement.pose.y,
            placement.z,
            yaw=placement.pose.yaw,
            reference_frame=self.world.root,
        )

    def _remove_target(self, annotation: HasRootBody) -> None:
        """
        Remove a target that could not be placed from the world.

        :param annotation: The target to remove.
        """
        with self.world.modify_world():
            self.world.remove_semantic_annotation(annotation)
            self.world.remove_kinematic_structure_entity(annotation.root)

    def _surface_names(self) -> List[str]:
        """
        :return: The names of the surface bodies targets are placed on.
        """
        return [surface.root.name.name for surface in self.surfaces]

    @staticmethod
    def _surface_area(surface: HasSupportingSurface) -> float:
        """
        :param surface: The annotated surface.
        :return: The XY area in square meters of the surface's supporting region.
        """
        bounding_box = SceneSpawner._surface_bounding_box(surface)
        return (bounding_box.max_x - bounding_box.min_x) * (
            bounding_box.max_y - bounding_box.min_y
        )

    @staticmethod
    def _surface_bounding_box(surface: HasSupportingSurface) -> BoundingBox:
        """
        :param surface: The annotated surface.
        :return: The bounding box of the surface's supporting region, in the region's
            own frame.
        """
        region_shapes = BoundingBoxCollection.from_shapes(
            surface.supporting_surface.area
        )
        region_shapes.transform_all_shapes_to_own_frame()
        return region_shapes.bounding_box()

    @staticmethod
    def _world_planar_pose(
        world_T_target: HomogeneousTransformationMatrix,
    ) -> PlanarPose:
        """
        :param world_T_target: The target's transform in the world frame.
        :return: The target's planar pose in the world frame.
        """
        rotation = world_T_target.to_np()
        return PlanarPose(
            x=float(world_T_target.x),
            y=float(world_T_target.y),
            yaw=float(math.atan2(rotation[1, 0], rotation[0, 0])),
        )
