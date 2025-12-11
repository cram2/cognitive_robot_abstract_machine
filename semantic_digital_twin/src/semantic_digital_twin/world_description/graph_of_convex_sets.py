from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from .geometry import BoundingBox
from .shape_collection import BoundingBoxCollection
from ..datastructures.variables import SpatialVariables
from ..world import World
from .world_entity import SemanticAnnotation, SemanticEnvironmentAnnotation

logger = logging.getLogger(__name__)

import time
from functools import reduce
from operator import or_
from typing_extensions import List, Optional, Dict, Sequence

# typing.Self is available starting with Python 3.11
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
from random_events.interval import reals
from random_events.product_algebra import SimpleEvent, Event
from rtree import index
from sortedcontainers import SortedSet

from ..spatial_types import Point3, HomogeneousTransformationMatrix


class PoseOccupiedError(Exception):
    """
    Error that is raised when a pose is occupied or not in the search space of a Connectivity Graphs.
    """

    def __init__(self, point: Point3):
        """
        Construct a new pose occupied error.
        :param pose: The pose that is occupied.
        """
        super().__init__(f"The pose {point} is occupied.")
        self.point = point


class GraphOfConvexSets:
    """
    A graph that represents the connectivity between convex sets.

    Every node in the graph is a convex set, represented by a bounding box.
    Every edge in the graph represents the connectivity between two convex sets.
    """

    search_space: BoundingBoxCollection
    """
    The bounding box of the search space. Defaults to the entire three dimensional space.
    """

    graph: rx.PyGraph[BoundingBox]
    """
    The connectivity graph of the convex sets.
    """

    box_to_index_map: Dict[BoundingBox, int]
    """
    A mapping from bounding boxes to their indices in the graph.
    """

    world: World
    """
    The world that the graph is based on.
    """

    def __init__(
        self, world: World, search_space: Optional[BoundingBoxCollection] = None
    ):
        self.search_space = self._make_search_space(world, search_space)
        self.graph = rx.PyGraph(multigraph=False)
        self.box_to_index_map = {}
        self.world = world

    def create_subgraph(self, nodes: Sequence[int]) -> Self:
        """
        Create a subgraph of the current graph containing only the given nodes.

        :param nodes: The nodes to include in the subgraph.
        :return: The subgraph.
        """
        subgraph = GraphOfConvexSets(self.world, self.search_space)
        subgraph.graph = self.graph.subgraph(nodes)
        subgraph.box_to_index_map = {
            box: index for box, index in self.box_to_index_map.items() if index in nodes
        }
        return subgraph

    def add_node(self, box: BoundingBox):
        self.box_to_index_map[box] = self.graph.add_node(box)

    def calculate_connectivity(self, tolerance=0.001):
        """
        Calculate the connectivity of the graph by checking for intersections between the bounding boxes of the nodes.
        This uses an R-tree for efficient spatial indexing and intersection queries.

        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        """

        def _overlap(a_min, a_max, b_min, b_max) -> bool:
            return (
                a_min[0] <= b_max[0]
                and b_min[0] <= a_max[0]
                and a_min[1] <= b_max[1]
                and b_min[1] <= a_max[1]
                and a_min[2] <= b_max[2]
                and b_min[2] <= a_max[2]
            )

        def _intersection_box(a_min, a_max, b_min, b_max):
            return BoundingBox(
                max(a_min[0], b_min[0]),
                max(a_min[1], b_min[1]),
                max(a_min[2], b_min[2]),
                min(a_max[0], b_max[0]),
                min(a_max[1], b_max[1]),
                min(a_max[2], b_max[2]),
                HomogeneousTransformationMatrix(reference_frame=self.world.root),
            )

        # Build a 3-D R-tree
        prop = index.Property()
        prop.dimension = 3
        rtree_idx = index.Index(properties=prop)

        node_list = list(self.graph.nodes())
        orig_mins, orig_maxs, expanded = [], [], []

        # Record every node once, insert it into the index
        for n in node_list:
            mn = (n.min_x, n.min_y, n.min_z)
            mx = (n.max_x, n.max_y, n.max_z)
            ex = (
                mn[0] - tolerance,
                mn[1] - tolerance,
                mn[2] - tolerance,
                mx[0] + tolerance,
                mx[1] + tolerance,
                mx[2] + tolerance,
            )

            orig_mins.append(mn)
            orig_maxs.append(mx)
            expanded.append(ex)
            rtree_idx.insert(len(orig_mins) - 1, ex)

        # Query & link, skip self-loops and symmetric pairs
        for i, (mn_i, mx_i, ex_i) in enumerate(zip(orig_mins, orig_maxs, expanded)):
            for j in rtree_idx.intersection(ex_i):
                if j <= i:  # symmetry → skip
                    continue
                mn_j, mx_j = orig_mins[j], orig_maxs[j]
                if not _overlap(mn_i, mx_i, mn_j, mx_j):
                    continue  # no true overlap
                box = _intersection_box(mn_i, mx_i, mn_j, mx_j)

                self.graph.add_edge(i, j, box)

    def draw(self):
        import rustworkx.visualization

        rustworkx.visualization.mpl_draw(self.graph)
        plt.show()

    def plot_free_space(self) -> List[go.Mesh3d]:
        """
        Plot the free space of the environment in blue.
        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event(*[node.simple_event for node in self.graph.nodes()])
        return free_space.plot(color="blue")

    def plot_occupied_space(self) -> List[go.Mesh3d]:
        """
        Plot the occupied space of the environment in red.
        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event(*[node.simple_event for node in self.graph.nodes()])
        occupied_space = ~free_space & self.search_space.event
        return occupied_space.plot(color="red")

    def node_of_point(self, point: Point3) -> Optional[BoundingBox]:
        """
        Find the node that contains a point.

        :return: The node that contains the point or None if no node contains the point.
        """
        for node in self.graph.nodes():
            if node.contains(point):
                return node
        return None

    def path_from_to(self, start: Point3, goal: Point3) -> Optional[List[Point3]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to or None if no path exists.
        """

        # get poses from params
        start_node = self.node_of_point(start)
        goal_node = self.node_of_point(goal)

        # validate if the poses are part of the graph
        if start_node is None:
            raise PoseOccupiedError(start)
        if goal_node is None:
            raise PoseOccupiedError(goal)

        if start_node == goal_node:
            return [start, goal]

        # get the shortest path (perhaps replace with a*?)
        paths = rx.all_shortest_paths(
            self.graph,
            self.box_to_index_map[start_node],
            self.box_to_index_map[goal_node],
        )

        # if it is not possible to find a path
        if len(paths) == 0:
            return None

        path = paths[0]

        # build the path
        result = [start]

        for source, target in zip(path, path[1:]):

            intersection: BoundingBox = self.graph.get_edge_data(source, target)
            x_target = intersection.x_interval.center()
            y_target = intersection.y_interval.center()
            z_target = intersection.z_interval.center()
            result.append(Point3(x_target, y_target, z_target))

        result.append(goal)
        return result

    @classmethod
    def _make_search_space(
        cls, world: World, search_space: Optional[BoundingBoxCollection] = None
    ):
        """
        Create the default search space if it is not given.
        """
        if search_space is None:
            search_space = BoundingBoxCollection(
                shapes=[
                    BoundingBox(
                        min_x=-np.inf,
                        min_y=-np.inf,
                        min_z=-np.inf,
                        max_x=np.inf,
                        max_y=np.inf,
                        max_z=np.inf,
                        origin=HomogeneousTransformationMatrix(
                            reference_frame=world.root
                        ),
                    )
                ],
                reference_frame=world.root,
            )
        return search_space

    @classmethod
    def obstacles_from_semantic_annotations(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
        keep_z=True,
    ) -> Event:
        """
        Create a connectivity graph from a list of semantic annotations.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation to create the connectivity graph from.
        :param semantic_wall_annotation: An optional semantic annotation containing walls to be considered as obstacles.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :param keep_z: If True, the z-axis is kept in the resulting event. Default is True.

        :return: An event representing the obstacles in the search space.
        """

        def bloat_obstacle(bb):
            return bb.bloat(bloat_obstacles, bloat_obstacles, 0.01)

        def bloat_wall(bb):
            if bb.width > bb.depth:
                return bb.bloat(bloat_walls, 0, 0.01)
            else:
                return bb.bloat(0, bloat_walls, 0.01)

        world_root = search_space.reference_frame

        bloated_obstacles: BoundingBoxCollection = BoundingBoxCollection(
            [
                bloat_obstacle(bb)
                for bb in semantic_obstacle_annotation.as_bounding_box_collection_at_origin(
                    HomogeneousTransformationMatrix(reference_frame=world_root)
                )
            ],
            world_root,
        )

        if semantic_wall_annotation is not None:
            bloated_walls: BoundingBoxCollection = BoundingBoxCollection(
                [
                    bloat_wall(bb)
                    for bb in semantic_wall_annotation.as_bounding_box_collection_at_origin(
                        HomogeneousTransformationMatrix(reference_frame=world_root)
                    )
                ],
                world_root,
            )
            bloated_obstacles.merge(bloated_walls)

        return cls.obstacles_from_bounding_boxes(
            bloated_obstacles, search_space.event, keep_z
        )

    @classmethod
    def obstacles_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
        keep_z: bool = True,
    ) -> Optional[Event]:
        """
        Create a connectivity graph from a list of bounding boxes.

        :param bounding_boxes: The list of bounding boxes to create the connectivity graph from.
        :param search_space_event: The search space event to limit the connectivity graph to.
        :param keep_z: If True, the z-axis is kept in the resulting event. Default is True.

        :return: An event representing the obstacles in the search space, or None if no obstacles are found.
        """

        if not keep_z:
            search_space_event = search_space_event.marginal(SpatialVariables.xy)

        events = (
            bb.simple_event.as_composite_set() & search_space_event
            for bb in bounding_boxes
        )

        # skip bbs outside the search space
        events = (event for event in events if not event.is_empty())

        if not keep_z:
            events = (event.marginal(SpatialVariables.xy) for event in events)

        try:
            return reduce(or_, events)
        except TypeError:
            logger.warning(
                "No obstacles found in the given semantic annotations. Returning None."
            )
            return None

    @classmethod
    def free_space_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Self:
        """
        Create a connectivity graph from the free space in the belief state of the robot.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.

        :return: The connectivity graph. If no obstacles are found, an empty graph is returned.
        """

        # get obstacles
        obstacles = cls.obstacles_from_semantic_annotations(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles=bloat_obstacles,
            bloat_walls=bloat_walls,
        )

        if obstacles is None or obstacles.is_empty():
            return cls(
                search_space=search_space,
                world=search_space.reference_frame._world,
            )

        search_event = search_space.event

        start_time = time.time_ns()
        # calculate the free space and limit it to the searching space
        free_space = ~obstacles & search_event
        logger.info(
            f"Free space calculated in {(time.time_ns() - start_time) / 1e6} ms"
        )

        # create a connectivity graph from the free space and calculate the edges
        result = cls(
            search_space=search_space, world=semantic_obstacle_annotation._world
        )
        [
            result.add_node(bb)
            for bb in BoundingBoxCollection.from_event(
                reference_frame=search_space.reference_frame,
                event=free_space,
            )
        ]

        start_time = time.time_ns()
        result.calculate_connectivity(tolerance)
        logger.info(
            f"Connectivity calculated in {(time.time_ns() - start_time) / 1e6} ms"
        )

        return result

    @classmethod
    def free_space_from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
    ) -> Self:
        """
        Create a connectivity graph from the free space in the belief state of the robot.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.

        :return: The connectivity graph.
        """

        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return cls.free_space_from_semantic_annotation(
            search_space=search_space,
            semantic_obstacle_annotation=semantic_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
        )

    @classmethod
    def obstacles_from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        bloat_obstacles: float = 0.0,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles in the belief state of the robot.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param bloat_obstacles: The amount to bloat the obstacles.

        :return: An event representing the obstacles in the search space.
        """

        view = SemanticEnvironmentAnnotation(root=world.root, _world=world)

        return cls.obstacles_from_semantic_annotations(
            search_space=search_space,
            semantic_obstacle_annotation=view,
            bloat_obstacles=bloat_obstacles,
        )

    @classmethod
    def navigation_map_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for navigation.
        The resulting GCS describes the paths for navigation, meaning that changing the z-axis position is not
        possible.
        Furthermore, it is taken into account that the robot has to fit through the entire space and not just
        through the floor level obstacles.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.

        :return: The connectivity graph. If no obstacles are found, an empty graph is returned.
        """

        # create search space for calculations
        obstacles = cls.obstacles_from_semantic_annotations(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
            keep_z=False,
        )

        if obstacles is None or obstacles.is_empty():
            return cls(
                world=search_space.reference_frame._world, search_space=search_space
            )

        # remove the z axis
        og_search_event = search_space.event
        search_event = og_search_event.marginal(SpatialVariables.xy)

        free_space = ~obstacles & search_event

        SimpleEvent({SpatialVariables.z.value: reals()})
        # create floor level
        z_event = SimpleEvent({SpatialVariables.z.value: reals()}).as_composite_set()
        z_event.fill_missing_variables(SpatialVariables.xy)
        free_space.fill_missing_variables(SortedSet([SpatialVariables.z.value]))
        free_space &= z_event
        free_space &= og_search_event

        # create a connectivity graph from the free space and calculate the edges
        result = cls(
            world=search_space.reference_frame._world,
            search_space=search_space,
        )
        free_space_boxes = BoundingBoxCollection.from_event(
            search_space.reference_frame, free_space
        )
        [result.add_node(bb) for bb in free_space_boxes]
        result.calculate_connectivity(tolerance)

        return result

    @classmethod
    def navigation_map_from_world(
        cls,
        world: World,
        tolerance=0.001,
        search_space: Optional[BoundingBoxCollection] = None,
        bloat_obstacles: float = 0.0,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for navigation.
        The resulting GCS describes the paths for navigation, meaning that changing the z-axis position is not
        possible.
        Furthermore, it is taken into account that the robot has to fit through the entire space and not just
        through the floor level obstacles.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.

        :return: The connectivity graph.
        """

        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return cls.navigation_map_from_semantic_annotation(
            search_space,
            semantic_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
        )
