from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, List, Optional, Sequence, Tuple

from krrood.class_diagrams.mocking import MockedClass
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    UnboundedSearchSpaceError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

logger = logging.getLogger(__name__)


# %% drake fallbacks
#
# ConvexSet, HPolyhedron, Iris, IrisOptions, Point, VPolytope, GcsTrajectoryOptimization
# and Subgraph are used below both as dataclass field types and as real runtime values.
# Leaving them unbound when drake is missing crashes any code that resolves this
# module's type hints (e.g. ORM/class-diagram generation), not just code that actually
# uses GraphOfConvexPolygons. Binding them to these mocks keeps every annotation
# resolvable; actually instantiating one still fails loudly, via
# krrood.class_diagrams.mocking.MockedClass.


@dataclass
class _MockedConvexSet(MockedClass):
    """
    Mocked class for ConvexSet in pydrake.
    """


@dataclass
class _MockedHPolyhedron(MockedClass):
    """
    Mocked class for HPolyhedron in pydrake.
    """


@dataclass
class _MockedIris(MockedClass):
    """
    Mocked class for the Iris function in pydrake.
    """


@dataclass
class _MockedIrisOptions(MockedClass):
    """
    Mocked class for IrisOptions in pydrake.
    """


@dataclass
class _MockedPoint(MockedClass):
    """
    Mocked class for Point in pydrake.
    """


@dataclass
class _MockedVPolytope(MockedClass):
    """
    Mocked class for VPolytope in pydrake.
    """


@dataclass
class _MockedGcsTrajectoryOptimization(MockedClass):
    """
    Mocked class for GcsTrajectoryOptimization in pydrake.
    """


@dataclass
class _MockedSubgraph(MockedClass):
    """
    Mocked class for GcsTrajectoryOptimization.Subgraph in pydrake.
    """


try:
    from pydrake.geometry.optimization import (
        ConvexSet,
        HPolyhedron,
        Iris,
        IrisOptions,
        Point,
        VPolytope,
    )
    from pydrake.planning import GcsTrajectoryOptimization

    Subgraph = GcsTrajectoryOptimization.Subgraph
    """
    Type alias for the ``Subgraph`` handle returned by
    ``GcsTrajectoryOptimization.AddRegions``.
    """

except ImportError:
    logger.warning(
        "drake is required for GraphOfConvexPolygons. Please install it using "
        "'pip install drake'."
    )
    ConvexSet = _MockedConvexSet
    HPolyhedron = _MockedHPolyhedron
    Iris = _MockedIris
    IrisOptions = _MockedIrisOptions
    Point = _MockedPoint
    VPolytope = _MockedVPolytope
    GcsTrajectoryOptimization = _MockedGcsTrajectoryOptimization
    Subgraph = _MockedSubgraph


def _default_iris_options() -> IrisOptions:
    """
    Build the default :class:`IrisOptions` used by :class:`GraphOfConvexPolygons`.

    :return: IRIS options with a bounded iteration count so a single region-growth call
        cannot dominate build time on large scenes.
    """
    options = IrisOptions()
    options.iteration_limit = 10
    options.termination_threshold = 2e-2
    options.relative_termination_threshold = 1e-2
    return options


@dataclass
class IrisSeedingSettings:
    """
    Settings controlling how many IRIS regions :class:`GraphOfConvexPolygons` grows, and
    from where.

    Drake's IRIS grows one convex region at a time from a single seed point, so
    covering a scene requires calling it repeatedly from different seeds. Seeding here
    is a simple, deterministic coverage heuristic (a regular grid), not a tuned or
    optimal strategy: a production seeding strategy (e.g. clique-cover-based, or seeded
    from known robot poses via ``extra_seed_points`` on
    :meth:`GraphOfConvexPolygons.from_world`) would likely produce a different, more
    efficient region layout.
    """

    grid_resolution: int = 5
    """
    Number of candidate seed points per axis in the coverage grid.
    """

    max_regions: int = 40
    """
    Upper bound on the number of IRIS regions grown.
    """

    iris_options: IrisOptions = field(default_factory=_default_iris_options)
    """
    Options forwarded to every :func:`pydrake.geometry.optimization.Iris` call.
    """

    def coverage_grid_seeds(self, lower: Point3, upper: Point3) -> List[Point3]:
        """
        Build a regular grid of candidate seed points strictly inside the axis-aligned
        box spanned by ``lower`` and ``upper``.

        :param lower: Lower corner of the region to seed.
        :param upper: Upper corner of the region to seed, in ``lower``'s reference
            frame.
        :return: ``grid_resolution`` candidate seed points per axis.
        """
        reference_frame = lower.reference_frame
        lower_array = lower.to_np()[:3]
        upper_array = upper.to_np()[:3]
        axes = [
            np.linspace(lower_array[axis], upper_array[axis], self.grid_resolution + 2)[
                1:-1
            ]
            for axis in range(3)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return [
            Point3(*point, reference_frame=reference_frame)
            for point in np.stack(grid, axis=-1).reshape(-1, 3)
        ]


def _validate_and_convert_domain(search_space: BoundingBoxCollection) -> HPolyhedron:
    """
    Convert a search space into the single, finite ``HPolyhedron`` domain IRIS grows
    regions within.

    :param search_space: The search space to convert.
    :raises UnboundedSearchSpaceError: If ``search_space`` is not exactly one finite
        bounding box.
    """
    boxes = search_space.bounding_boxes
    if len(boxes) != 1:
        raise UnboundedSearchSpaceError()
    bounds = boxes[0].to_array_bounds()
    if not (np.all(np.isfinite(bounds.lower)) and np.all(np.isfinite(bounds.upper))):
        raise UnboundedSearchSpaceError()
    return HPolyhedron.MakeBox(bounds.lower, bounds.upper)


def _bloat_convex_hull_points(
    vertices: np.ndarray, bloat_x: float, bloat_y: float
) -> np.ndarray:
    """
    Expand a convex point set by ``(bloat_x, bloat_y, 0)`` via a Minkowski sum with an
    axis-aligned box.

    The sum of two convex sets is the convex hull of the pairwise sums of their
    vertices; since a ``VPolytope``'s vertices do not have to be minimal, returning the
    union of four shifted copies (one per corner of the thin box) is sufficient without
    computing the hull explicitly.

    :param vertices: The point set to expand.
    :param bloat_x: Half-extent to add in x.
    :param bloat_y: Half-extent to add in y.
    :return: The expanded point set.
    """
    if bloat_x == 0.0 and bloat_y == 0.0:
        return vertices
    offsets = [
        (dx, dy, 0.0) for dx in (-bloat_x, bloat_x) for dy in (-bloat_y, bloat_y)
    ]
    return np.concatenate([vertices + offset for offset in offsets], axis=0)


def _shape_to_convex_set(
    shape: Shape,
    target_frame: KinematicStructureEntity,
    bloat_x: float,
    bloat_y: float,
) -> ConvexSet:
    """
    Convert a collision shape into an exact convex set when its mesh is convex, falling
    back to its axis-aligned bounding box otherwise.

    :param shape: The collision shape to convert.
    :param target_frame: The frame to express the result in.
    :param bloat_x: Amount to expand the result in x.
    :param bloat_y: Amount to expand the result in y.
    :return: A ``VPolytope`` built from the shape's own mesh if it is convex, or an
        ``HPolyhedron`` bounding box otherwise.
    """
    world_mesh = shape.mesh_in_frame(target_frame)

    if world_mesh.is_convex:
        vertices = _bloat_convex_hull_points(world_mesh.vertices, bloat_x, bloat_y)
        return VPolytope(vertices.T)

    bounding_box = shape.local_frame_bounding_box.transform_to_origin(
        HomogeneousTransformationMatrix(reference_frame=target_frame)
    ).bloat(bloat_x, bloat_y, 0.01)
    bounds = bounding_box.to_array_bounds()
    return HPolyhedron.MakeBox(bounds.lower, bounds.upper)


@dataclass
class GraphOfConvexPolygons(GraphOfConvexSets):
    """
    A graph of convex sets whose regions are grown by Drake's IRIS algorithm and solved
    with Drake's ``GcsTrajectoryOptimization`` (:cite:t:`marcucci2022shortest`).

    Unlike :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.boxes.GraphOfBoundingBoxes`,
    which exhaustively partitions free space into many small axis-aligned boxes, IRIS
    covers free space with a handful of large, non-axis-aligned convex regions -- a
    sufficient cover for solving path queries, not a complete map of free space (some
    free points may not be covered by any region, in which case ``path_from_to``
    returns None for them rather than raising, since they are not necessarily
    occupied).

    Region growth (in :meth:`from_world`) is the expensive part of building this
    class; once built, :meth:`path_from_to` can be called repeatedly and cheaply for
    different start/goal pairs, since only the per-query endpoint subgraphs are added
    and removed rather than rebuilding the whole graph.
    """

    obstacles: List[ConvexSet] = field(default_factory=list)
    """
    The obstacle convex sets every region was grown to avoid, expressed relative to
    ``world.root``.

    Each collision shape with a convex mesh is used exactly (as a ``VPolytope``); shapes
    with a non-convex mesh fall back to an axis-aligned bounding box (``HPolyhedron``).
    """

    regions: List[HPolyhedron] = field(default_factory=list)
    """
    The IRIS regions covering the search space.
    """

    _trajectory_optimization: GcsTrajectoryOptimization = field(
        default_factory=lambda: GcsTrajectoryOptimization(3)
    )
    """
    The persistent trajectory-optimization instance holding the region subgraph and,
    while a query is in flight, its source/target point subgraphs.
    """

    _region_subgraph: Optional[Subgraph] = None
    """
    The subgraph spanning ``regions``, built once in :meth:`from_world` and reused by
    every :meth:`path_from_to` query.
    """

    _query_subgraphs: Optional[Tuple[Subgraph, Subgraph]] = None
    """
    The current query's ``(source, target)`` point subgraphs, or None between queries.
    """

    @property
    def region_count(self) -> int:
        """
        :return: The number of IRIS regions in this graph.
        """
        return len(self.regions)

    @classmethod
    def from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        bloat_obstacles: float = 0.0,
        seeding_settings: Optional[IrisSeedingSettings] = None,
        extra_seed_points: Sequence[Point3] = (),
    ) -> GraphOfConvexPolygons:
        """
        Grow IRIS regions covering the free space of *world* within *search_space*.

        :param world: The belief state to plan in.
        :param search_space: The navigable search volume; must consist of exactly one
            finite bounding box (see :class:`UnboundedSearchSpaceError`).
        :param bloat_obstacles: Amount to expand each obstacle in x/y, closing gaps
            between the individual panel shapes that make up a single piece of furniture
            (mirrors ``GraphOfBoundingBoxes``'s bloat parameter).
        :param seeding_settings: IRIS seeding strategy and options; defaults to
            :class:`IrisSeedingSettings`'s defaults.
        :param extra_seed_points: Points to seed a region from first, before the regular
            coverage grid -- e.g. known robot poses that should be covered by a region
            even if the grid would otherwise miss them.
        :return: A graph ready to answer repeated :meth:`path_from_to` queries.
        """
        result = cls(world, search_space)
        settings = seeding_settings or IrisSeedingSettings()
        domain = _validate_and_convert_domain(result.search_space)

        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )
        obstacle_entities = cls._obstacle_entities(
            result.search_space, semantic_annotation
        )
        result.obstacles = [
            _shape_to_convex_set(shape, world.root, bloat_obstacles, bloat_obstacles)
            for entity in obstacle_entities
            for shape in entity.collision.shapes
        ]

        bounds = result.search_space.bounding_boxes[0].to_point3_bounds()
        seeds = list(extra_seed_points) + settings.coverage_grid_seeds(
            bounds.lower, bounds.upper
        )

        regions: List[HPolyhedron] = []
        for seed_point in seeds:
            if len(regions) >= settings.max_regions:
                break
            seed = world.transform(seed_point, world.root).to_np()[:3]
            if any(obstacle.PointInSet(seed) for obstacle in result.obstacles):
                continue
            if any(region.PointInSet(seed) for region in regions):
                continue
            regions.append(Iris(result.obstacles, seed, domain, settings.iris_options))

        result.regions = regions
        result._region_subgraph = result._trajectory_optimization.AddRegions(
            regions, order=1
        )
        result._trajectory_optimization.AddPathLengthCost()
        return result

    def path_from_to(self, start: Point3, goal: Point3) -> Optional[List[Point3]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to, or None if no region
            covers a path between them.
        :raises PointOccupiedError: If ``start`` or ``goal`` lies inside an obstacle.
        """
        start_array = self.world.transform(start, self.world.root).to_np()[:3]
        goal_array = self.world.transform(goal, self.world.root).to_np()[:3]

        if any(obstacle.PointInSet(start_array) for obstacle in self.obstacles):
            raise PointOccupiedError(start)
        if any(obstacle.PointInSet(goal_array) for obstacle in self.obstacles):
            raise PointOccupiedError(goal)

        self._clear_previous_query()
        source_subgraph = self._trajectory_optimization.AddRegions(
            [Point(start_array)], order=0
        )
        target_subgraph = self._trajectory_optimization.AddRegions(
            [Point(goal_array)], order=0
        )
        self._trajectory_optimization.AddEdges(source_subgraph, self._region_subgraph)
        self._trajectory_optimization.AddEdges(self._region_subgraph, target_subgraph)
        self._query_subgraphs = (source_subgraph, target_subgraph)

        trajectory, result = self._trajectory_optimization.SolvePath(
            source_subgraph, target_subgraph
        )
        if not result.is_success():
            return None

        breakpoints = trajectory.get_segment_times()
        interior_waypoints = [
            Point3(*trajectory.value(t).flatten(), reference_frame=self.world.root)
            for t in breakpoints[1:-1]
        ]
        return [start] + interior_waypoints + [goal]

    def _clear_previous_query(self) -> None:
        if self._query_subgraphs is None:
            return
        for subgraph in self._query_subgraphs:
            self._trajectory_optimization.RemoveSubgraph(subgraph)
        self._query_subgraphs = None
