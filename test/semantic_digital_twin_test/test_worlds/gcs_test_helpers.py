"""
Helpers for building demonstrative path queries over a graph of convex sets.

Not part of the production API: nothing in the codebase plans a path between the two
hardest-to-reach convex sets on its own -- this exists only to build interesting,
reproducible scenes for the figure tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import rustworkx as rx
from typing_extensions import Generic

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.graph_of_convex_sets.base import PointT
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    BoxT,
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    UnconnectedGraphError,
)


@dataclass(frozen=True)
class PathQuery(Generic[PointT]):
    """
    A start and a goal to plan a path between.
    """

    start: PointT
    """
    Where the path begins.
    """

    goal: PointT
    """
    Where the path ends.
    """


def hardest_path_query(
    graph: GraphOfBoundingBoxes[BoxT, PointT],
) -> PathQuery[PointT]:
    """
    Pick the query that is hardest to answer: the two convex set centers whose shortest
    path through the graph is the longest one it holds.

    Distance is measured along the graph rather than straight-line, so the query lands
    on the pair the environment actually forces a detour between instead of on whichever
    two sets happen to sit in opposite corners of an open room. Centers of convex sets
    are free by construction, and a pair connected by a path is solvable by definition.
    Ties are broken by coordinate rather than by the order the graph happens to hold its
    nodes in, so that the same graph always yields the same query.

    :param graph: The graph to query.
    :return: The query.
    :raises UnconnectedGraphError: If no two convex sets are connected.
    """
    path_lengths = rx.all_pairs_dijkstra_path_lengths(
        graph.graph, edge_cost_fn=lambda adjacency: adjacency.distance
    )
    # VolumetricBoundingBox.center recomputes symbolic arithmetic on every access, and the
    # tie-break below reads one per pair, so every center is resolved to floats once
    # here instead.
    coordinates = {
        index: _coordinates_of(graph, index) for index in graph.graph.node_indices()
    }
    # Each connected pair once, so which end is named start never depends on which
    # direction the search happened to report first.
    connected_pairs = [
        (source, target)
        for source, targets in path_lengths.items()
        for target in targets
        if source < target
    ]
    if not connected_pairs:
        raise UnconnectedGraphError(graph.graph.num_nodes())

    # Ties are broken on the pair's coordinates rather than on its graph indices, which
    # depend on the order the world happened to yield its obstacles in.
    most_distant_pair = max(
        connected_pairs,
        key=lambda pair: (
            path_lengths[pair[0]][pair[1]],
            *sorted((coordinates[pair[0]], coordinates[pair[1]])),
        ),
    )
    start_index, goal_index = sorted(
        most_distant_pair, key=lambda index: coordinates[index]
    )
    return PathQuery(
        start=graph.graph[start_index].center, goal=graph.graph[goal_index].center
    )


def _coordinates_of(
    graph: GraphOfBoundingBoxes[BoxT, PointT], index: int
) -> tuple[float, float, float]:
    """
    :param graph: The graph holding the convex set.
    :param index: The index of the convex set in that graph.
    :return: The center of the convex set, as plain floats to compare by. The third
        coordinate reads as 0 for a planar graph, whose centers have no z.
    """
    center = graph.graph[index].center
    z = float(center.z) if isinstance(center, Point3) else 0.0
    return float(center.x), float(center.y), z
