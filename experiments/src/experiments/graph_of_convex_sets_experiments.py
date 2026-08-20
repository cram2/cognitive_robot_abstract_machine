"""
End-to-end performance benchmark for the Graph of Convex Sets (GCS) free- space
pipeline, covering three families of environments:

  * **URDF environments** — rooms loaded from `semantic_digital_twin/resources/urdf`
    via :class:`URDFParser`.

  * **PartNet-Mobility scenes** — articulated objects loaded from
    ``~/partnet-mobility-dataset`` via :class:`PartNetMobilityDatasetLoader`.

  * **Sage10k scenes** — real, multi-room scenes (walls, doors, furniture from PLY
    meshes) downloaded from the ``nvidia/SAGE-10k`` dataset on Hugging Face via
    :class:`Sage10kDatasetLoader`. Optional and off by default, since loading a scene's
    meshes typically takes minutes: ``--sage10k-count`` benchmarks scenes from the
    curated :class:`Sage10kActionableScenes`, and ``--sage10k-random-count`` benchmarks
    additional scenes sampled at random from the full, uncurated dataset.

For all three, the navigable search space is the minimal box covering every obstacle in
the scene, widened by 1 m in x and y (see
:meth:`GraphOfConvexSetsFreespaceBenchmark._compute_minimal_search_space`).

For every environment, both GCS implementations are benchmarked and compared against
a mesh-accurate free-space ground truth:

  * :class:`GraphOfBoundingBoxes` exhaustively partitions free space into disjoint
    axis-aligned boxes; its own volume is an exact, always-valid lower bound on the
    true free volume (obstacle bounding boxes only ever over-approximate the real
    geometry).

  * :class:`GraphOfConvexPolygons` grows a handful of possibly-overlapping IRIS
    regions; since their union has no closed-form volume, it is scored via
    :class:`~experiments.free_space_volume_estimation.MonteCarloFreeSpaceSampler`
    against the same sample points used for the ground truth.

Results are collected into :class:`GraphOfConvexSetsFreespaceExperimentResult`
rows and printed as a Typst ``#table`` block.  All timing columns report
wall-clock milliseconds; repeated phases show mean ± standard deviation over
three runs.

Run with::

    python -m experiments.graph_of_convex_sets_experiments
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
from urllib.parse import urlparse

import tqdm
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    PercentageBound,
    TypstRenderer,
    Unit,
    VolumeBound,
)
from experiments.free_space_volume_estimation import MonteCarloFreeSpaceSampler
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kActionableScenes,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox, Shape
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.polygons import (
    GraphOfConvexPolygons,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class GraphOfConvexSetsFreespaceExperimentResult(ExperimentResult):
    """
    Performance measurements for the Graph of Convex Sets free-space pipeline.

    All duration fields are in milliseconds.
    """

    world_loading_duration_milliseconds: float
    """
    Wall-clock time to parse the URDF and set up collision geometry.

    Reported for context only, not as part of the GCS comparison: every other timing
    field in this row is measured against an already-loaded world, since a real system
    only starts building a GCS once it already has a belief state and a 3D goal to
    reach.
    """

    obstacle_count: int
    """
    Number of obstacle bounding boxes found in the world.
    """

    free_space_computation_duration: MeanAndStandardDeviation
    """
    Time to compute the free-space event via ``subtract_disjoint`` (mean ± standard
    deviation).
    """

    free_space_simple_set_count: int
    """
    Number of simple sets (axis-aligned boxes) in the resulting free-space event.
    """

    materialise_duration: MeanAndStandardDeviation
    """
    Time to convert the free-space event into a :class:`BoundingBoxCollection` (mean ±
    standard deviation).
    """

    free_space_bounding_box_count: int
    """
    Number of bounding boxes in the materialised free-space collection.
    """

    connectivity_duration: MeanAndStandardDeviation
    """
    Time to build the R-tree intersection graph (mean ± standard deviation).
    """

    graph_node_count: int
    """
    Number of nodes (free-space bounding boxes) in the connectivity graph.
    """

    graph_edge_count: int
    """
    Number of edges (adjacencies) in the connectivity graph.
    """

    box_construction_duration: MeanAndStandardDeviation
    """
    Time to build a complete, query-ready :class:`GraphOfBoundingBoxes` from an already-
    loaded world via ``GraphOfBoundingBoxes.free_space_from_world`` (mean ± standard
    deviation).

    World loading is excluded: by the time a navigation goal arrives, the world is
    already loaded, so this is the number directly comparable to
    :attr:`graph_of_convex_polygons_construction_duration` for deciding
    which GCS implementation is faster to construct.
    """

    mesh_obstacle_volume_total: float
    """
    Sum of every obstacle's own mesh volume, ignoring overlaps between obstacles.
    """

    naive_free_volume_lower_bound: float
    """
    Search-space volume minus :attr:`mesh_obstacle_volume_total`.

    Always a valid lower bound on the true free volume regardless of overlaps between
    obstacles, since the sum of individual obstacle volumes is never smaller than the
    volume of their union.
    """

    box_free_space_volume: float
    """
    Volume of :class:`GraphOfBoundingBoxes`'s own free-space partition (the sum of its
    disjoint box volumes).

    Also a valid, exact lower bound on the true free volume, since obstacle bounding
    boxes only ever over-approximate the real obstacle geometry.
    """

    true_free_volume_bound: VolumeBound
    """
    Bound on the true, mesh-accurate free volume: the tighter of
    :attr:`naive_free_volume_lower_bound` and :attr:`box_free_space_volume` for the
    lower end, and a Monte Carlo confidence-interval estimate against the real obstacle
    meshes for the upper end.
    """

    graph_of_convex_polygons_construction_duration: MeanAndStandardDeviation
    """
    Time to grow the IRIS regions via ``GraphOfConvexPolygons.from_world`` on the
    already-loaded world (mean ± standard deviation).

    Directly comparable to :attr:`box_construction_duration`, since both measure
    construction from the same loaded world with world loading excluded.
    """

    graph_of_convex_polygons_region_count: int
    """
    Number of IRIS regions grown.
    """

    graph_of_convex_polygons_region_volume_sum: float
    """
    Sum of every IRIS region's own volume, ignoring overlaps between regions.
    """

    graph_of_convex_polygons_coverage_bound: VolumeBound
    """
    Bound on the free volume covered by the union of IRIS regions: a Monte Carlo
    confidence-interval estimate, capped above by
    :attr:`graph_of_convex_polygons_region_volume_sum` (an exact upper bound, since
    overlap only ever inflates the sum past the union's true volume).
    """

    graph_of_convex_polygons_coverage_percentage: PercentageBound
    """
    :attr:`graph_of_convex_polygons_coverage_bound` as a percentage of
    :attr:`true_free_volume_bound`.
    """

    environment_name: str
    """
    Human-readable label identifying the environment (e.g. ``"apartment"`` or
    ``"partnet_179"``).
    """


@dataclass
class GraphOfConvexSetsFreespaceBenchmark:
    """
    Runs all GCS free-space benchmark phases for a single environment and produces a
    single :class:`GraphOfConvexSetsFreespaceExperimentResult` row.

    Use :meth:`for_urdf`, :meth:`for_partnet_model`, or :meth:`for_sage10k_scene` to
    construct one for a given environment family, then call :meth:`run`.
    """

    world_loader: Callable[[], World]
    """
    Zero-argument callable that loads and returns a :class:`World`.

    Called once by :meth:`run`, to build the world every other phase operates on. World
    loading is timed separately (:attr:`~GraphOfConvexSetsFreespaceExperimentResult.world_loading_duration_milliseconds`)
    but excluded from every construction-time comparison, since the world is already
    loaded by the time a navigation goal arrives in the scenario this benchmark models:
    GCS construction only starts once there is a 3D goal to reach.
    """

    search_space_factory: Callable[[World], BoundingBoxCollection]
    """
    Callable that receives a loaded :class:`World` and returns the
    :class:`BoundingBoxCollection` to use as the navigable search volume.
    """

    environment_name: str
    """
    Human-readable label stored in the result row (e.g. ``"apartment"`` or
    ``"partnet_179"``).
    """

    @classmethod
    def for_urdf(cls, urdf_path: Path) -> GraphOfConvexSetsFreespaceBenchmark:
        """
        Build a benchmark for a single URDF environment file.

        The search space is the minimal box covering every obstacle in the scene,
        widened by 1 m in x and y.

        :param urdf_path: Path to the ``.urdf`` file to benchmark.
        """

        def _search_space_factory(world):
            return cls._compute_minimal_search_space(
                cls._collect_obstacles(world), world
            )

        return cls(
            world_loader=lambda: URDFParser.from_file(str(urdf_path)).parse(),
            search_space_factory=_search_space_factory,
            environment_name=Path(urdf_path).stem,
        )

    @classmethod
    def for_partnet_model(
        cls, loader: PartNetMobilityDatasetLoader, model_id: int
    ) -> GraphOfConvexSetsFreespaceBenchmark:
        """
        Build a benchmark for a single PartNet-Mobility model.

        The search space is the minimal box covering every obstacle in the scene,
        widened by 1 m in x and y.

        :param loader: A configured :class:`PartNetMobilityDatasetLoader` (already
            points at the local dataset directory).
        :param model_id: PartNet-Mobility model identifier (e.g. ``179``).
        """
        return cls(
            world_loader=lambda: loader.load(model_id),
            search_space_factory=lambda world: cls._compute_minimal_search_space(
                cls._collect_obstacles(world), world
            ),
            environment_name=f"partnet_{model_id}",
        )

    @classmethod
    def for_sage10k_scene(
        cls, loader: Sage10kDatasetLoader, scene_url: str, environment_name: str
    ) -> GraphOfConvexSetsFreespaceBenchmark:
        """
        Build a benchmark for a single Sage10k scene.

        The search space is the minimal box covering every obstacle in the scene,
        widened by 1 m in x and y.

        :param loader: A configured :class:`Sage10kDatasetLoader`.
        :param scene_url: URL of the Sage10k scene to benchmark; downloaded (and cached)
            from Hugging Face on first use. Loading a scene's meshes typically takes
            minutes, not seconds, so this is far more expensive per environment than the
            URDF or PartNet-Mobility families.
        :param environment_name: Human-readable label stored in the result row.
        """
        return cls(
            world_loader=lambda: loader.create_scene(
                scene_url=scene_url
            ).create_world(),
            search_space_factory=lambda world: cls._compute_minimal_search_space(
                cls._collect_obstacles(world), world
            ),
            environment_name=environment_name,
        )

    @staticmethod
    def environment_name_for_sage10k_url(scene_url: str) -> str:
        """
        :param scene_url: A Sage10k scene URL (e.g. one from
            :meth:`Sage10kDatasetLoader.available_scenes`).
        :return: A human-readable label derived from the URL's filename, e.g.
            ``sage10k_random_20251213_171403_layout_26384448``.
        """
        return f"sage10k_random_{Path(urlparse(scene_url).path).stem}"

    def run(self) -> GraphOfConvexSetsFreespaceExperimentResult:
        """
        Run all GCS free-space benchmark phases and return a single result row.
        """
        world, world_loading_elapsed = self._measure(self.world_loader)
        world_loading_duration_milliseconds = world_loading_elapsed[0] * 1000.0

        obstacle_bounding_boxes, _ = self._measure(
            lambda: self._collect_obstacles(world)
        )
        search_space = self.search_space_factory(world)
        search_event = search_space.event

        def _compute_free_space():
            free_space_accumulator = search_event
            for bounding_box in obstacle_bounding_boxes:
                obstacle = bounding_box.simple_event.as_composite_set() & search_event
                if not obstacle.is_empty():
                    free_space_accumulator = free_space_accumulator.subtract_disjoint(
                        obstacle
                    )
            return free_space_accumulator

        free_space, free_space_elapsed = self._measure(
            _compute_free_space, repetitions=3
        )
        free_space_simple_set_count = len(list(free_space.simple_sets))

        def _materialise_free_space():
            return BoundingBoxCollection.from_event(
                reference_frame=world.root, event=free_space
            )

        free_space_collection, materialise_elapsed = self._measure(
            _materialise_free_space, repetitions=3
        )

        def _compute_connectivity():
            graph_of_convex_sets = GraphOfBoundingBoxes(
                world=world, search_space=search_space
            )
            for bounding_box in free_space_collection:
                graph_of_convex_sets.add_node(bounding_box)
            graph_of_convex_sets.calculate_connectivity(tolerance=0.001)
            return graph_of_convex_sets

        connectivity_graph, connectivity_elapsed = self._measure(
            _compute_connectivity, repetitions=3
        )

        def _construct_box_graph_from_loaded_world():
            return GraphOfBoundingBoxes.free_space_from_world(world, search_space)

        _, box_construction_elapsed = self._measure(
            _construct_box_graph_from_loaded_world, repetitions=3
        )

        search_space_box = search_space.bounding_boxes[0]
        obstacle_shapes = self._collect_obstacle_shapes(world)
        mesh_obstacle_volume_total = sum(shape.volume for shape in obstacle_shapes)
        naive_free_volume_lower_bound = (
            self._box_volume(search_space_box) - mesh_obstacle_volume_total
        )
        box_free_space_volume = sum(
            self._box_volume(box) for box in free_space_collection
        )

        sampler = MonteCarloFreeSpaceSampler.for_obstacle_shapes(
            search_space_bounding_box=search_space_box,
            obstacle_shapes=obstacle_shapes,
            target_frame=world.root,
        )
        ground_truth_bound = sampler.free_volume_bound()
        true_free_volume_lower_bound = max(
            0.0, naive_free_volume_lower_bound, box_free_space_volume
        )
        true_free_volume_bound = VolumeBound(
            lower=true_free_volume_lower_bound,
            # Widened rather than left to raise: the Monte Carlo upper end is only a
            # confidence-interval estimate, so an unlucky draw could in principle fall
            # below an exact lower bound above.
            upper=max(true_free_volume_lower_bound, ground_truth_bound.upper),
        )

        def _grow_convex_polygons():
            return GraphOfConvexPolygons.from_world(world, search_space)

        graph_of_convex_polygons, polygon_construction_elapsed = self._measure(
            _grow_convex_polygons, repetitions=3
        )
        region_volume_sum = sum(
            self._region_volume(region) for region in graph_of_convex_polygons.regions
        )
        raw_coverage_bound = sampler.region_coverage_bound(
            graph_of_convex_polygons.regions
        )
        coverage_bound = VolumeBound(
            lower=raw_coverage_bound.lower,
            # Capped rather than left to raise: overlap between regions only ever
            # inflates region_volume_sum past the true covered volume, so it is a valid
            # upper bound even where it is tighter than the sampled estimate.
            upper=max(
                raw_coverage_bound.lower,
                min(raw_coverage_bound.upper, region_volume_sum),
            ),
        )
        coverage_percentage = PercentageBound.ratio_of(
            coverage_bound, true_free_volume_bound
        )

        return GraphOfConvexSetsFreespaceExperimentResult(
            world_loading_duration_milliseconds=round(
                world_loading_duration_milliseconds, 2
            ),
            obstacle_count=len(obstacle_bounding_boxes),
            free_space_computation_duration=MeanAndStandardDeviation.from_measurements(
                free_space_elapsed, unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            free_space_simple_set_count=free_space_simple_set_count,
            materialise_duration=MeanAndStandardDeviation.from_measurements(
                materialise_elapsed, unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            free_space_bounding_box_count=len(free_space_collection),
            connectivity_duration=MeanAndStandardDeviation.from_measurements(
                connectivity_elapsed, unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            graph_node_count=len(connectivity_graph.graph.nodes()),
            graph_edge_count=len(connectivity_graph.graph.edges()),
            box_construction_duration=MeanAndStandardDeviation.from_measurements(
                box_construction_elapsed, unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            mesh_obstacle_volume_total=round(mesh_obstacle_volume_total, 4),
            naive_free_volume_lower_bound=round(naive_free_volume_lower_bound, 4),
            box_free_space_volume=round(box_free_space_volume, 4),
            true_free_volume_bound=true_free_volume_bound,
            graph_of_convex_polygons_construction_duration=MeanAndStandardDeviation.from_measurements(
                polygon_construction_elapsed, unit=Unit.SECONDS
            ).to(
                Unit.MILLISECONDS
            ),
            graph_of_convex_polygons_region_count=graph_of_convex_polygons.region_count,
            graph_of_convex_polygons_region_volume_sum=round(region_volume_sum, 4),
            graph_of_convex_polygons_coverage_bound=coverage_bound,
            graph_of_convex_polygons_coverage_percentage=coverage_percentage,
            environment_name=self.environment_name,
        )

    @staticmethod
    def _measure(function_to_time, repetitions: int = 1):
        """
        Call *function_to_time* *repetitions* times and return the last result together
        with all elapsed times.

        The last result is returned rather than the first so callers can directly unpack
        the value they want to inspect.

        :param function_to_time: Zero-argument callable to benchmark.
        :type function_to_time: Callable[[], Any]
        :param repetitions: Number of times to invoke *function_to_time*.
        :type repetitions: int
        :returns: A 2-tuple ``(last_result, elapsed_seconds_list)`` where *last_result*
            is the return value of the final invocation and *elapsed_seconds_list* is a
            list of wall-clock durations in seconds, one per repetition.
        :rtype: tuple[Any, list[float]]
        """
        elapsed_times: List[float] = []
        result = None
        for _ in range(repetitions):
            start = time.perf_counter()
            result = function_to_time()
            elapsed_times.append(time.perf_counter() - start)
        return result, elapsed_times

    @staticmethod
    def _collect_obstacles(world: World) -> List[BoundingBox]:
        """
        Return all obstacle bounding boxes from world expressed at the world root frame.

        :param world: The world to query.
        :returns: List of bounding boxes.
        """
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        return list(annotation.as_bounding_box_collection_at_origin(origin))

    @staticmethod
    def _collect_obstacle_shapes(world: World) -> List[Shape]:
        """
        Return every collision shape of every obstacle body in world.

        Mirrors the obstacle selection of :meth:`_collect_obstacles` (same bodies, one
        entry per collision shape instead of one bounding box per body), so the mesh-
        accurate ground truth in :meth:`run` stays comparable to that method's bounding-
        box-based numbers.

        :param world: The world to query.
        :returns: List of collision shapes.
        """
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        return [
            shape
            for entity in annotation.kinematic_structure_entities
            if isinstance(entity, Body) and entity.has_collision()
            for shape in entity.collision.shapes
        ]

    @staticmethod
    def _box_volume(box: BoundingBox) -> float:
        """
        :param box: The bounding box to measure.
        :return: The volume enclosed by box.
        """
        return box.depth * box.width * box.height

    @staticmethod
    def _region_volume(region) -> float:
        """
        :param region: An IRIS region.
        :return: The region's volume.

        Falls back to Drake's own suggested Monte Carlo alternative for the regions
        ``HPolyhedron.CalcVolume`` cannot compute exactly (this is Drake's documented
        escape hatch, raised as a plain ``RuntimeError`` with no more specific type to
        catch).
        """
        try:
            return region.CalcVolume()
        except RuntimeError:
            from pydrake.common import RandomGenerator

            # Explicit accuracy/sample-count arguments, not just RandomGenerator(0):
            # this pydrake binding's default arguments do not resolve correctly for
            # HPolyhedron and raise a spurious TypeError.
            return region.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000).volume

    @staticmethod
    def _compute_minimal_search_space(
        obstacle_bounding_boxes: List[BoundingBox], world, xy_widen: float = 1.0
    ) -> BoundingBoxCollection:
        """
        Derive a search-space bounding box as the minimal box covering every obstacle,
        widened in x and y only.

        z is left tight to the scene's own extent (unwidened), since the navigable
        volume should not grow taller than the scene that defines it.

        :param obstacle_bounding_boxes: List of bounding boxes to include in the search
            space.
        :param world: The world to which the bounding boxes belong.
        :param xy_widen: Total amount to widen the minimal covering box by, split evenly
            between both sides, in x and y.
        """
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        if not obstacle_bounding_boxes:
            return BoundingBoxCollection(
                shapes=[
                    BoundingBox(
                        min_x=-2.0,
                        min_y=-2.0,
                        min_z=-2.0,
                        max_x=2.0,
                        max_y=2.0,
                        max_z=2.0,
                        origin=origin,
                    )
                ],
                reference_frame=world.root,
            )
        all_min_x = min(bb.min_x for bb in obstacle_bounding_boxes)
        all_min_y = min(bb.min_y for bb in obstacle_bounding_boxes)
        all_min_z = min(bb.min_z for bb in obstacle_bounding_boxes)
        all_max_x = max(bb.max_x for bb in obstacle_bounding_boxes)
        all_max_y = max(bb.max_y for bb in obstacle_bounding_boxes)
        all_max_z = max(bb.max_z for bb in obstacle_bounding_boxes)
        half_xy_widen = xy_widen / 2.0
        return BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=all_min_x - half_xy_widen,
                    min_y=all_min_y - half_xy_widen,
                    min_z=all_min_z,
                    max_x=all_max_x + half_xy_widen,
                    max_y=all_max_y + half_xy_widen,
                    max_z=all_max_z,
                    origin=origin,
                )
            ],
            reference_frame=world.root,
        )


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting how many Sage10k scenes to benchmark.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage10k-count",
        type=int,
        default=0,
        choices=range(len(Sage10kActionableScenes) + 1),
        help="Number of Sage10k scenes to benchmark, taken in the order listed by "
        "Sage10kActionableScenes. Defaults to 0 (skipped): each scene downloads (once, "
        "then caches) and rebuilds its world from PLY meshes, which can take minutes "
        "per scene.",
    )
    parser.add_argument(
        "--sage10k-random-count",
        type=int,
        default=0,
        help="Number of additional Sage10k scenes to benchmark, sampled at random "
        "(without replacement) from the full nvidia/SAGE-10k dataset rather than the "
        "curated Sage10kActionableScenes. Defaults to 0 (skipped). Requires "
        "huggingface_hub to list the dataset.",
    )
    parser.add_argument(
        "--sage10k-random-seed",
        type=int,
        default=42,
        help="Seed for --sage10k-random-count's sampling, so repeated runs pick the "
        "same random scenes.",
    )
    return parser.parse_args()


def main():
    """
    Benchmark all URDF environments, the first 10 PartNet-Mobility models, and
    optionally some Sage10k scenes (see ``--sage10k-count`` and ``--sage10k-random-
    count``).

    Results are collected into a single :class:`ExperimentsTable` and printed as a Typst
    ``#table`` block ready for inclusion in a scientific article. Progress is tracked
    via tqdm progress bars.
    """
    arguments = _parse_arguments()

    urdf_directory_path = (
        Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / "semantic_digital_twin"
        / "resources"
        / "urdf"
    )

    results = []
    for urdf_file in (
        pbar := tqdm.tqdm(list(sorted(urdf_directory_path.glob("*.urdf"))))
    ):
        pbar.set_description(f"Running benchmark for {urdf_file.stem}")
        results.append(GraphOfConvexSetsFreespaceBenchmark.for_urdf(urdf_file).run())

    loader = PartNetMobilityDatasetLoader()
    for model_id in (pbar := tqdm.tqdm(loader.available_model_ids[:10])):
        pbar.set_description(f"Running benchmark for partnet model {model_id}")
        results.append(
            GraphOfConvexSetsFreespaceBenchmark.for_partnet_model(
                loader, model_id
            ).run()
        )

    sage10k_scenes = list(Sage10kActionableScenes)[: arguments.sage10k_count]
    sage10k_loader = Sage10kDatasetLoader()
    for scene_url in (pbar := tqdm.tqdm(sage10k_scenes)):
        pbar.set_description(f"Running benchmark for sage10k {scene_url.name}")
        results.append(
            GraphOfConvexSetsFreespaceBenchmark.for_sage10k_scene(
                sage10k_loader, scene_url, f"sage10k_{scene_url.name.lower()}"
            ).run()
        )

    if arguments.sage10k_random_count:
        all_scene_urls = Sage10kDatasetLoader.available_scenes()
        random_scene_urls = random.Random(arguments.sage10k_random_seed).sample(
            all_scene_urls, arguments.sage10k_random_count
        )
        for scene_url in (pbar := tqdm.tqdm(random_scene_urls)):
            environment_name = (
                GraphOfConvexSetsFreespaceBenchmark.environment_name_for_sage10k_url(
                    scene_url
                )
            )
            pbar.set_description(f"Running benchmark for {environment_name}")
            try:
                results.append(
                    GraphOfConvexSetsFreespaceBenchmark.for_sage10k_scene(
                        sage10k_loader, scene_url, environment_name
                    ).run()
                )
            except Exception as error:
                # Scenes here are sampled from the full, uncurated dataset (unlike
                # Sage10kActionableScenes, which are verified to load), so a malformed
                # scene is expected occasionally and should not sink the whole run.
                logger.warning(f"Skipping {environment_name}: {error}")

    table = ExperimentsTable(results)
    print(TypstRenderer(table).render_table())


if __name__ == "__main__":
    main()
