"""
End-to-end performance benchmark for the Graph of Convex Sets (GCS) free- space
pipeline, covering two families of environments:

  * **URDF environments** — rooms loaded from `semantic_digital_twin/resources/urdf`
    via :class:`URDFParser`.  The navigable search volume is a fixed axis-aligned
    box (±20 m in XY, 0–3 m in Z) that conservatively contains all scenes.

  * **PartNet-Mobility scenes** — articulated objects loaded from
    ``~/partnet-mobility-dataset`` via :class:`PartNetMobilityDatasetLoader`.
    The search space is derived automatically from the union of each object's
    obstacle bounding boxes plus 0.5 m of padding.

Results are collected into :class:`GraphOfConvexSetsFreespaceExperimentResult`
rows and printed as a Typst ``#table`` block.  All timing columns report
wall-clock milliseconds; repeated phases show mean ± standard deviation over
three runs.

Run with::

    python -m experiments.graph_of_convex_sets_experiments
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import tqdm
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)


@dataclass
class GraphOfConvexSetsFreespaceExperimentResult(ExperimentResult):
    """
    Performance measurements for the Graph of Convex Sets free-space pipeline.

    All duration fields are in milliseconds.
    """

    world_loading_duration_milliseconds: float
    """
    Wall-clock time to parse the URDF and set up collision geometry.
    """

    obstacle_count: int
    """
    Number of obstacle bounding boxes found in the world.
    """

    free_space_computation_duration_milliseconds: MeanAndStandardDeviation
    """
    Time to compute the free-space event via ``subtract_disjoint`` (mean ±
    standard deviation).
    """

    free_space_simple_set_count: int
    """
    Number of simple sets (axis-aligned boxes) in the resulting free-space event.
    """

    materialise_duration_milliseconds: MeanAndStandardDeviation
    """
    Time to convert the free-space event into a :class:`BoundingBoxCollection`
    (mean ± standard deviation).
    """

    free_space_bounding_box_count: int
    """
    Number of bounding boxes in the materialised free-space collection.
    """

    connectivity_duration_milliseconds: MeanAndStandardDeviation
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

    end_to_end_duration_milliseconds: float
    """
    Wall-clock time for a complete ``free_space_from_world`` call including world
    loading.
    """

    environment_name: str
    """
    Human-readable label identifying the environment (e.g. ``"apartment"`` or
    ``"partnet_179"``).
    """


def _measure(function_to_time, repetitions: int = 1):
    """
    Call *function_to_time* *repetitions* times and return the last result together with
    all elapsed times.

    The last result is returned rather than the first so callers can directly unpack the
    value they want to inspect.

    :param function_to_time: Zero-argument callable to benchmark.
    :type function_to_time: Callable[[], Any]
    :param repetitions: Number of times to invoke *function_to_time*.
    :type repetitions: int
    :returns: A 2-tuple ``(last_result, elapsed_seconds_list)`` where *last_result* is
        the return value of the final invocation and *elapsed_seconds_list* is a list of
        wall-clock durations in seconds, one per repetition.
    :rtype: tuple[Any, list[float]]
    """
    elapsed_times: List[float] = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function_to_time()
        elapsed_times.append(time.perf_counter() - start)
    return result, elapsed_times


def _to_mean_and_standard_deviation_milliseconds(
    elapsed_seconds: List[float],
) -> MeanAndStandardDeviation:
    """
    Convert a list of raw elapsed-seconds values to a :class:`MeanAndStandardDeviation`
    in milliseconds.

    :param elapsed_seconds: Raw timing samples in seconds, as returned by the second
        element of :func:`_measure`.
    :type elapsed_seconds: list[float]
    :returns: Mean and standard deviation of the samples converted to milliseconds.
    """
    return MeanAndStandardDeviation.from_measurements(
        [seconds * 1000.0 for seconds in elapsed_seconds]
    )


def _collect_obstacles(world: World) -> List[BoundingBox]:
    """
    Return all obstacle bounding boxes from world expressed at the world root frame.

    :param world: The world to query.
    :returns: List of bounding boxes.
    """
    annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    return list(annotation.as_bounding_box_collection_at_origin(origin))


def _compute_search_space_from_obstacles(
    obstacle_bounding_boxes: List[BoundingBox], world, padding: float = 0.5
) -> BoundingBoxCollection:
    """
    Derive a search-space bounding box from the union of obstacle bounding boxes plus
    padding.

    :param obstacle_bounding_boxes: List of bounding boxes to include in the search
        space.
    :param world: The world to which the bounding boxes belong.
    :param padding: Amount of padding to add around the union of the
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
    return BoundingBoxCollection(
        shapes=[
            BoundingBox(
                min_x=all_min_x - padding,
                min_y=all_min_y - padding,
                min_z=all_min_z - padding,
                max_x=all_max_x + padding,
                max_y=all_max_y + padding,
                max_z=all_max_z + padding,
                origin=origin,
            )
        ],
        reference_frame=world.root,
    )


def _run_benchmark(
    world_loader: Callable[[], object],
    search_space_factory: Callable[[object], BoundingBoxCollection],
    environment_name: str,
) -> GraphOfConvexSetsFreespaceExperimentResult:
    """
    Run all GCS free-space benchmark phases and return a single result row.

    :param world_loader: Zero-argument callable that loads and returns a :class:`World`.
        Called once for the timed world-loading phase and once more inside the end-to-
        end phase.
    :param search_space_factory: Callable that receives a loaded :class:`World` and
        returns the :class:`BoundingBoxCollection` to use as the navigable search
        volume. For URDF environments this is a fixed room-scale box; for PartNet models
        it is derived from the obstacle extents.
    :param environment_name: Human-readable label stored in the result row (e.g.
        ``"apartment"`` or ``"partnet_179"``).
    """
    world, world_loading_elapsed = _measure(world_loader)
    world_loading_duration_milliseconds = world_loading_elapsed[0] * 1000.0

    obstacle_bounding_boxes, _ = _measure(lambda: _collect_obstacles(world))
    search_space = search_space_factory(world)
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

    free_space, free_space_elapsed = _measure(_compute_free_space, repetitions=3)
    free_space_simple_set_count = len(list(free_space.simple_sets))

    def _materialise_free_space():
        return BoundingBoxCollection.from_event(
            reference_frame=world.root, event=free_space
        )

    free_space_collection, materialise_elapsed = _measure(
        _materialise_free_space, repetitions=3
    )

    def _compute_connectivity():
        graph_of_convex_sets = GraphOfConvexSets(world=world, search_space=search_space)
        for bounding_box in free_space_collection:
            graph_of_convex_sets.add_node(bounding_box)
        graph_of_convex_sets.calculate_connectivity(tolerance=0.001)
        return graph_of_convex_sets

    connectivity_graph, connectivity_elapsed = _measure(
        _compute_connectivity, repetitions=3
    )

    def _run_end_to_end():
        loaded_world = world_loader()
        e2e_search_space = search_space_factory(loaded_world)
        return GraphOfConvexSets.free_space_from_world(loaded_world, e2e_search_space)

    _, end_to_end_elapsed = _measure(_run_end_to_end)
    end_to_end_duration_milliseconds = end_to_end_elapsed[0] * 1000.0

    return GraphOfConvexSetsFreespaceExperimentResult(
        world_loading_duration_milliseconds=round(
            world_loading_duration_milliseconds, 2
        ),
        obstacle_count=len(obstacle_bounding_boxes),
        free_space_computation_duration_milliseconds=_to_mean_and_standard_deviation_milliseconds(
            free_space_elapsed
        ),
        free_space_simple_set_count=free_space_simple_set_count,
        materialise_duration_milliseconds=_to_mean_and_standard_deviation_milliseconds(
            materialise_elapsed
        ),
        free_space_bounding_box_count=len(free_space_collection),
        connectivity_duration_milliseconds=_to_mean_and_standard_deviation_milliseconds(
            connectivity_elapsed
        ),
        graph_node_count=len(connectivity_graph.graph.nodes()),
        graph_edge_count=len(connectivity_graph.graph.edges()),
        end_to_end_duration_milliseconds=round(end_to_end_duration_milliseconds, 2),
        environment_name=environment_name,
    )


def _perform_benchmark_for_environment(
    urdf_path: Path,
) -> GraphOfConvexSetsFreespaceExperimentResult:
    """
    Run the GCS benchmark for a single URDF environment file.

    The search space is a fixed ±20 m × ±20 m × 3 m box centred at the world root, which
    conservatively contains all scenes in the URDF resource folder.

    :param urdf_path: Path to the ``.urdf`` file to benchmark.
    """

    def _search_space_factory(world):
        return BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=-20.0,
                    min_y=-20.0,
                    min_z=0.0,
                    max_x=20.0,
                    max_y=20.0,
                    max_z=3.0,
                    origin=HomogeneousTransformationMatrix(reference_frame=world.root),
                )
            ],
            reference_frame=world.root,
        )

    return _run_benchmark(
        world_loader=lambda: URDFParser.from_file(str(urdf_path)).parse(),
        search_space_factory=_search_space_factory,
        environment_name=Path(urdf_path).stem,
    )


def _perform_benchmark_for_partnet_model(
    loader: PartNetMobilityDatasetLoader, model_id: int
) -> GraphOfConvexSetsFreespaceExperimentResult:
    """
    Run the GCS benchmark for a single PartNet-Mobility model.

    The search space is derived automatically: obstacle bounding boxes are
    collected from the loaded world and a single enclosing box with 0.5 m
    padding on every side is used as the navigable volume.

    :param loader: A configured :class:`PartNetMobilityDatasetLoader` (already
        points at the local dataset directory).
    :param model_id: PartNet-Mobility model identifier (e.g. ``179``).
    """
    return _run_benchmark(
        world_loader=lambda: loader.load(model_id),
        search_space_factory=lambda world: _compute_search_space_from_obstacles(
            _collect_obstacles(world), world
        ),
        environment_name=f"partnet_{model_id}",
    )


def main():
    """
    Benchmark all URDF environments and the first 10 PartNet-Mobility models.

    Results are collected into a single :class:`ExperimentsTable` and printed as a Typst
    ``#table`` block ready for inclusion in a scientific article. Progress is tracked
    via tqdm progress bars.
    """
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
        results.append(_perform_benchmark_for_environment(urdf_file))

    loader = PartNetMobilityDatasetLoader()
    for model_id in (pbar := tqdm.tqdm(loader.available_model_ids[:10])):
        pbar.set_description(f"Running benchmark for partnet model {model_id}")
        results.append(_perform_benchmark_for_partnet_model(loader, model_id))

    table = ExperimentsTable(results)
    print(TypstRenderer(table).render_table())


if __name__ == "__main__":
    main()
