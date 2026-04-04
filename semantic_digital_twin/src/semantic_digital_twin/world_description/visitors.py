from __future__ import annotations

from typing import Tuple

import rustworkx as rx
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.world_description.world_entity import Body, Connection

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class CollisionBodyCollector(rx.visit.DFSVisitor):
    """
    Collects all bodies with collision geometries in the kinematic structure of the world.
    """

    def __init__(self, world: World):
        self.world = world
        self.bodies = []

    def discover_vertex(self, node_index: int, time: int) -> None:
        """Called for each vertex during DFS traversal"""
        body = self.world.kinematic_structure[node_index]
        if isinstance(body, Body) and body.has_collision():
            self.bodies.append(body)

    def tree_edge(self, args: Tuple[int, int, Connection]) -> None:
        """Called for each tree edge during DFS traversal"""
        parent_index, child_index, e = args
        if e.is_controlled:
            raise rx.visit.PruneSearch()


class ConnectionCollector(rx.visit.DFSVisitor):
    """
    Collects all connections in the kinematic structure of the world.
    """

    def __init__(self, world: World):
        self.world = world
        self.connections = []

    def tree_edge(self, edge: Tuple[int, int, Connection]):
        """Called for each tree edge during DFS traversal"""
        self.connections.append(edge[2])  # edge[2] is the connection
