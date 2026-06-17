"""
End-to-end performance benchmark for the Graph of Convex Sets free-space pipeline
on the IAI Apartment world loaded via the semantic_digital_twin package.

Results are collected into a GCSFreespaceExperimentResult dataclass and printed
as a Typst table, following the same pattern used by other experiments in this
repository.

Phases measured:
  Phase 1 – World loading    : URDF parse and pybullet collision setup
  Phase 3 – Collect obstacles: gather all obstacle bounding boxes from the world
  Phase 4+5 – Free space     : bounded incremental subtraction via subtract_disjoint
  Phase 6 – Materialise      : convert the free-space Event to a BoundingBoxCollection
  Phase 7 – Connectivity     : build the R-tree intersection graph
  Phase 8 – End-to-end       : single call to GraphOfConvexSets.free_space_from_world

Run with:
    python3 coraplex/demos/coraplex_gcs_demo/experiments/graph_of_convex_sets_experiments.py

Requirements (all installed in-repo):
    semantic_digital_twin, random_events, rtree, trimesh, urdf_parser_py, experiments
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

_APARTMENT_URDF_PATH = (
    Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / ".."
    / "semantic_digital_twin"
    / "resources"
    / "urdf"
    / "apartment.urdf"
)


@dataclass
class GCSFreespaceExperimentResult(ExperimentResult):
    """
    Performance measurements for the Graph of Convex Sets free-space pipeline
    on the IAI Apartment world.

    All duration fields are in milliseconds.
    """

    world_loading_duration_milliseconds: float
    """Wall-clock time to parse the URDF and set up pybullet collision geometry."""

    obstacle_count: int
    """Number of obstacle bounding boxes found in the apartment world."""

    free_space_computation_duration_milliseconds: MeanAndStandardDeviation
    """Time to compute the free-space event via subtract_disjoint (mean and standard deviation)."""

    free_space_simple_set_count: int
    """Number of simple sets (axis-aligned boxes) in the resulting free-space event."""

    materialise_duration_milliseconds: MeanAndStandardDeviation
    """Time to convert the free-space event into a BoundingBoxCollection (mean and standard deviation)."""

    free_space_bounding_box_count: int
    """Number of bounding boxes in the materialised free-space collection."""

    connectivity_duration_milliseconds: MeanAndStandardDeviation
    """Time to build the R-tree intersection graph (mean and standard deviation)."""

    graph_node_count: int
    """Number of nodes (free-space bounding boxes) in the connectivity graph."""

    graph_edge_count: int
    """Number of edges (adjacencies) in the connectivity graph."""

    end_to_end_duration_milliseconds: float
    """Wall-clock time for a complete free_space_from_world call including world loading."""


def _measure(function_to_time, repetitions: int = 1):
    """Run function_to_time the given number of times and return (result, elapsed_seconds_list)."""
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
    return MeanAndStandardDeviation.from_measurements(
        [seconds * 1000.0 for seconds in elapsed_seconds]
    )


def main():
    print("=" * 65)
    print("Graph of Convex Sets Free-Space Benchmark  –  IAI Apartment World")
    print("=" * 65)

    def _load_apartment_world():
        parser = URDFParser.from_file(_APARTMENT_URDF_PATH)
        return parser.parse()

    world, world_loading_elapsed = _measure(_load_apartment_world)
    world_loading_duration_milliseconds = world_loading_elapsed[0] * 1000.0

    body_count = len(list(world.bodies))
    collision_body_count = sum(
        1 for body in world.bodies if isinstance(body, Body) and body.has_collision()
    )
    print(f"  World loaded: {body_count} bodies, {collision_body_count} with collision")

    # Define the navigable search volume of the apartment.
    # The apartment furniture root is at (8.85, 1.75, 0).
    # Walls span roughly x ∈ [-1, 12],  y ∈ [-3, 5],  z ∈ [0, 3].
    search_space = BoundingBoxCollection(
        shapes=[
            BoundingBox(
                min_x=-1.0,
                min_y=-3.0,
                min_z=0.0,
                max_x=12.0,
                max_y=5.0,
                max_z=3.0,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        reference_frame=world.root,
    )
    search_event = search_space.event

    def _collect_obstacle_bounding_boxes():
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        return list(annotation.as_bounding_box_collection_at_origin(origin))

    obstacle_bounding_boxes, _ = _measure(_collect_obstacle_bounding_boxes)
    print(f"  Obstacle bounding boxes: {len(obstacle_bounding_boxes)}")

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
    print(f"  Free-space simple sets: {free_space_simple_set_count}")

    def _materialise_free_space():
        return BoundingBoxCollection.from_event(
            reference_frame=world.root, event=free_space
        )

    free_space_collection, materialise_elapsed = _measure(
        _materialise_free_space, repetitions=3
    )
    print(f"  Free-space bounding boxes: {len(free_space_collection)}")

    def _compute_connectivity():
        graph_of_convex_sets = GraphOfConvexSets(world=world, search_space=search_space)
        for bounding_box in free_space_collection:
            graph_of_convex_sets.add_node(bounding_box)
        graph_of_convex_sets.calculate_connectivity(tolerance=0.001)
        return graph_of_convex_sets

    connectivity_graph, connectivity_elapsed = _measure(
        _compute_connectivity, repetitions=3
    )
    print(
        f"  Graph: {len(connectivity_graph.graph.nodes())} nodes,"
        f" {len(connectivity_graph.graph.edges())} edges"
    )

    def _run_end_to_end():
        loaded_world = _load_apartment_world()
        apartment_search_space = BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=-1.0,
                    min_y=-3.0,
                    min_z=0.0,
                    max_x=12.0,
                    max_y=5.0,
                    max_z=3.0,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=loaded_world.root
                    ),
                )
            ],
            reference_frame=loaded_world.root,
        )
        return GraphOfConvexSets.free_space_from_world(
            loaded_world, apartment_search_space
        )

    end_to_end_graph, end_to_end_elapsed = _measure(_run_end_to_end)
    end_to_end_duration_milliseconds = end_to_end_elapsed[0] * 1000.0

    result = GCSFreespaceExperimentResult(
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
    )

    table = ExperimentsTable(experiments=[result])
    renderer = TypstRenderer(experiments_table=table)
    print("\n" + "=" * 65)
    print(renderer.render_table())


if __name__ == "__main__":
    main()
