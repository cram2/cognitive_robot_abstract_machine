from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import reduce
from operator import or_

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import rustworkx as rx
from random_events.interval import reals, closed
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from rtree import index
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Dict, Sequence, Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox, Color
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphOfBoundingBoxes(GraphOfConvexSets):
    """
    A graph of convex sets whose nodes are axis-aligned bounding boxes.

    Free space is decomposed into an exact, exhaustive partition of boxes via the
    `random_events` product algebra (obstacles subtracted from the search space). Every
    node is a box; every edge represents the adjacency between two boxes.
    """

    graph: rx.PyGraph[BoundingBox] = field(
        default_factory=lambda: rx.PyGraph(multigraph=False)
    )
    """
    The connectivity graph of the convex sets.
    """

    box_to_index_map: Dict[BoundingBox, int] = field(default_factory=dict)
    """
    A mapping from bounding boxes to their indices in the graph.
    """

    def create_subgraph(self, nodes: Sequence[int]) -> Self:
        """
        Create a subgraph of the current graph containing only the given nodes.

        :param nodes: The nodes to include in the subgraph.
        :return: The subgraph.
        """
        subgraph = GraphOfBoundingBoxes(self.world, self.search_space)
        subgraph.graph = self.graph.subgraph(nodes)
        subgraph.box_to_index_map = {
            box: index for box, index in self.box_to_index_map.items() if index in nodes
        }
        return subgraph

    def add_node(self, box: BoundingBox):
        self.box_to_index_map[box] = self.graph.add_node(box)

    def calculate_connectivity(self, tolerance=0.001):
        """
        Calculate the connectivity of the graph by checking for intersections between
        the bounding boxes of the nodes. This uses an R-tree for efficient spatial
        indexing and intersection queries.

        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
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
            mn = (n.x_interval.lower, n.y_interval.lower, n.z_interval.lower)
            mx = (n.x_interval.upper, n.y_interval.upper, n.z_interval.upper)
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

                # Map from the local list positions back to the graph node indices
                u = self.box_to_index_map[node_list[i]]
                v = self.box_to_index_map[node_list[j]]

                self.graph.add_edge(u, v, box)

    def draw(self):
        import rustworkx.visualization

        rustworkx.visualization.mpl_draw(self.graph)
        plt.show()

    def plot_free_space(self) -> List[go.Mesh3d]:
        """
        Plot the free space of the environment in blue.

        :return: A list of traces that can be put into a plotly figure.
        """
        return self.free_space_event.plot(color="blue")

    def plot_and_show_free_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_free_space()).show()

    def plot_occupied_space(self) -> List[go.Mesh3d]:
        """
        Plot the occupied space of the environment in red.

        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )
        occupied_space = ~free_space & self.search_space.event
        return occupied_space.plot(color="red")

    def plot_and_show_occupied_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_occupied_space()).show()

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

        .. note::
            Uses a single-source Dijkstra search (unweighted, i.e. hop-count shortest
            path) rather than enumerating all shortest paths and picking the first one.
            Free-space decompositions with thousands of nodes routinely have an
            exponential number of equally-short paths, which makes enumerating all of
            them intractable; finding just one is not.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to or None if no path
            exists.
        """
        # get poses from params
        start_node = self.node_of_point(start)
        goal_node = self.node_of_point(goal)

        # validate if the poses are part of the graph
        if start_node is None:
            raise PointOccupiedError(start)
        if goal_node is None:
            raise PointOccupiedError(goal)

        if start_node == goal_node:
            return [start, goal]

        start_index = self.box_to_index_map[start_node]
        goal_index = self.box_to_index_map[goal_node]

        paths = rx.dijkstra_shortest_paths(self.graph, start_index, target=goal_index)

        # if it is not possible to find a path
        if goal_index not in paths:
            return None

        path = paths[goal_index]

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
        :param semantic_obstacle_annotation: The semantic annotation to create the
            connectivity graph from.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :param keep_z: If True, the z-axis is kept in the resulting event. Default is
            True.
        :return: An event representing the obstacles in the search space.
        """
        bloated_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )
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

        :param bounding_boxes: The list of bounding boxes to create the connectivity
            graph from.
        :param search_space_event: The search space event to limit the connectivity
            graph to.
        :param keep_z: If True, the z-axis is kept in the resulting event. Default is
            True.
        :return: An event representing the obstacles in the search space, or None if no
            obstacles are found.
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
    def free_space_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
        keep_z: bool = True,
    ) -> Event:
        """
        Compute the free space by subtracting each obstacle bounding box from the search
        space incrementally (subtract_disjoint), avoiding complement in the full ambient
        space and the costly union-then-complement pipeline.

        This is 40-50× faster than
        ``~obstacles_from_bounding_boxes(...) & search_space_event``
        because:
        - The subtraction stays bounded inside search_space_event at every step.
        - No make_disjoint() calls are needed (disjointness is maintained by construction).
        - The intermediate obstacle union is never materialised.

        :param bounding_boxes: The obstacle bounding boxes to subtract.
        :param search_space_event: The search space; the result is always a subset of this.
        :param keep_z: If True, the z-axis is kept. Default is True.
        :return: The free space as a disjoint Event.
        """
        if not keep_z:
            search_space_event = search_space_event.marginal(SpatialVariables.xy)

        free_space = search_space_event
        for bounding_box in bounding_boxes:
            obstacle = bounding_box.simple_event.as_composite_set()
            if not keep_z:
                obstacle = obstacle.marginal(SpatialVariables.xy)
            obstacle_in_search = obstacle & search_space_event
            if not obstacle_in_search.is_empty():
                free_space = free_space.subtract_disjoint(obstacle_in_search)
            if free_space.is_empty():
                break
        return free_space

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
        Create a connectivity graph from the free space in the belief state of the
        robot.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the
            obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        bloated_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )

        search_event = search_space.event

        start_time = time.time_ns()
        # compute free space via bounded incremental subtraction (avoids complement in ℝ³)
        free_space = cls.free_space_from_bounding_boxes(bloated_obstacles, search_event)
        logger.info(
            f"Free space calculated in {(time.time_ns() - start_time) / 1e6} ms"
        )

        # create a connectivity graph from the free space and calculate the edges
        result = cls(
            search_space=search_space, world=semantic_obstacle_annotation._world
        )
        [
            result.add_node(bounding_box)
            for bounding_box in BoundingBoxCollection.from_event(
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
        Create a connectivity graph from the free space in the belief state of the
        robot.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
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
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the
            obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        nav_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )

        if not nav_obstacles:
            return cls(
                world=search_space.reference_frame._world, search_space=search_space
            )

        # Remove the z-axis so free-space is computed on the 2-D floor plane.
        full_search_event = search_space.event
        search_event = full_search_event.marginal(SpatialVariables.xy)

        free_space = cls.free_space_from_bounding_boxes(
            nav_obstacles, full_search_event, keep_z=False
        )

        SimpleEvent.from_data({SpatialVariables.z.value: reals()})
        # create floor level
        z_event = SimpleEvent.from_data(
            {SpatialVariables.z.value: reals()}
        ).as_composite_set()
        z_event.fill_missing_variables(SpatialVariables.xy)
        free_space.fill_missing_variables(SortedSet([SpatialVariables.z.value]))
        free_space &= z_event
        free_space &= full_search_event

        # create a connectivity graph from the free space and calculate the edges
        result = cls(
            world=search_space.reference_frame._world,
            search_space=search_space,
        )
        free_space_boxes = BoundingBoxCollection.from_event(
            search_space.reference_frame, free_space
        )
        [result.add_node(bounding_box) for bounding_box in free_space_boxes]
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
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
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

    @property
    def free_space_event(self) -> Event:
        return Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )

    def create_as_region(
        self,
        name: Optional[PrefixedName] = None,
        color: Color = Color(0.5, 1.0, 0.5, 0.5),
    ) -> Region:
        """
        Spawn the GCS as a region (world_entity) connected with a fixed connection with
        the root of the GCS search space. The geometry should be all boxes extracted
        from its free space.

        :param name: The name of the region.
        :param color: The color of the region.
        :return: The region.
        """
        if name is None:
            name = PrefixedName("gcs_region")

        bbox_collection = BoundingBoxCollection(
            shapes=list(self.graph.nodes()),
            reference_frame=self.search_space.reference_frame,
        )

        shapes = bbox_collection.as_shapes()
        shapes.dye_shapes(color)
        region = Region.from_shape_collection(name, shapes)

        with self.world.modify_world():
            self.world.add_region(region)

            self.world.add_connection(
                FixedConnection(
                    parent=self.search_space.reference_frame,
                    child=region,
                )
            )
        return region


def navigation_map_at_target(
    target: Body,
    search_range_x: float = 2.0,
    search_range_y: float = 2.0,
    max_height: float = 2.0,
    bloat_obstacles: float = 0.02,
) -> GraphOfBoundingBoxes:
    """
    Create a navigation map around the target.

    The navigation map is a Graph of Convex Sets that represents the navigable space
    around the target. The search space is constructed as a box around the target with
    the specified search ranges in the x and y directions.

    :param target: The target around which the navigation map is created.
    :param search_range_x: The search range in the x-direction.
    :param search_range_y: The search range in the y-direction.
    :param max_height: The maximum height of the navigation map from the floor.
    :param bloat_obstacles: The amount to bloat obstacles in the navigation map.
    :return: The navigation map as a Graph of Convex Sets.
    """
    search_space = BoundingBoxCollection.from_simple_event(
        reference_frame=target,
        simple_event=SimpleEvent.from_data(
            {
                SpatialVariables.x.value: closed(
                    -search_range_x / 2, search_range_x / 2
                ),
                SpatialVariables.y.value: closed(
                    -search_range_y / 2, search_range_y / 2
                ),
                SpatialVariables.z.value: closed(
                    -target.global_pose.z, max_height - target.global_pose.z
                ),
            }
        ),
    )

    graph_of_bounding_boxes = GraphOfBoundingBoxes.navigation_map_from_world(
        world=target._world, search_space=search_space, bloat_obstacles=bloat_obstacles
    )
    return graph_of_bounding_boxes
