from __future__ import annotations

import logging
import time
from functools import reduce
from operator import or_

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
from rtree import index
from sortedcontainers import SortedSet

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
    Agent,
)

logger = logging.getLogger("semantic_digital_twin")
from typing_extensions import List, Optional, Dict, Sequence
from typing_extensions import Self

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.operators.core_logical_operators import (
    OR,
    AND,
    chained_logic,
)
from random_events.interval import reals, Interval, SimpleInterval, closed, Bound
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    BoundingBox,
    Color,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

logger = logging.getLogger(__name__)


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

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to or None if no path exists.
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
    def _build_bloated_obstacle_collection(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> BoundingBoxCollection:
        """
        Collect and bloat obstacle bounding boxes from semantic annotations.

        Filters out agent entities so the robot does not treat itself as an obstacle.
        Applies independent bloat amounts to obstacles and walls.

        :param search_space: The search space; its reference frame is used as the origin.
        :param semantic_obstacle_annotation: The annotation containing obstacle entities.
        :param semantic_wall_annotation: An optional annotation containing wall entities.
        :param bloat_obstacles: Amount to expand each obstacle bounding box symmetrically in x and y.
        :param bloat_walls: Amount to expand wall bounding boxes in their thinner dimension.
        :return: A BoundingBoxCollection of the bloated obstacle and wall bounding boxes.
        """
        world_root = search_space.reference_frame
        world = world_root._world

        agents = world.get_semantic_annotations_by_type(Agent)
        agent_entities = set()
        for agent in agents:
            agent_entities.update(agent.kinematic_structure_entities)

        entities_to_consider = [
            entity
            for entity in semantic_obstacle_annotation.kinematic_structure_entities
            if isinstance(entity, Body)
            and entity.has_collision()
            and entity not in agent_entities
        ]

        collections = [
            entity.collision.as_bounding_box_collection_at_origin(
                HomogeneousTransformationMatrix(reference_frame=world_root)
            )
            for entity in entities_to_consider
        ]

        obstacle_bounding_boxes = BoundingBoxCollection([], world_root)
        for bounding_box_collection in collections:
            obstacle_bounding_boxes = obstacle_bounding_boxes.merge(bounding_box_collection)

        bloated_obstacles = BoundingBoxCollection(
            [bounding_box.bloat(bloat_obstacles, bloat_obstacles, 0.01) for bounding_box in obstacle_bounding_boxes],
            world_root,
        )

        if semantic_wall_annotation is not None:
            bloated_walls: BoundingBoxCollection = BoundingBoxCollection(
                [
                    bounding_box.bloat(bloat_walls, 0, 0.01)
                    if bounding_box.width > bounding_box.depth
                    else bounding_box.bloat(0, bloat_walls, 0.01)
                    for bounding_box in semantic_wall_annotation.as_bounding_box_collection_at_origin(
                        HomogeneousTransformationMatrix(reference_frame=world_root)
                    )
                ],
                world_root,
            )
            bloated_obstacles.merge(bloated_walls)

        return bloated_obstacles

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
        Create a connectivity graph from the free space in the belief state of the robot.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.

        :return: The connectivity graph. If no obstacles are found, an empty graph is returned.
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

        free_space = cls.free_space_from_bounding_boxes(nav_obstacles, full_search_event, keep_z=False)

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
        Spawn the GCS as a region (world_entity) connected with a fixed connection with the root of the GCS search space.
        The geometry should be all boxes extracted from its free space.

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


def translate_event_to(
    event: Event,
    position: Point3,
) -> Event:
    """
    Translates an event by a given position.
    A translation is a change in the position of an entity in space without altering its shape or orientation.

    :param event: The event to translate.
    :param position: The position to translate the event by.
    :return: The translated event.
    """
    variable_to_offset = {
        SpatialVariables.x.value: position.x,
        SpatialVariables.y.value: position.y,
        SpatialVariables.z.value: position.z,
    }
    results = []
    for simple_event in event.simple_sets:
        data = dict()
        for v, offset in variable_to_offset.items():
            data[v] = Interval.from_simple_sets(
                *[
                    SimpleInterval.from_data(
                        lower=simple_interval.lower + offset,
                        upper=simple_interval.upper + offset,
                        left=simple_interval.left,
                        right=simple_interval.right,
                    )
                    for simple_interval in simple_event[v]
                ]
            )
        results.append(SimpleEvent.from_data(data))
    return Event.from_simple_sets(*results)


def navigation_map_at_target(
    target: Body,
    search_range_x: float = 2.0,
    search_range_y: float = 2.0,
    max_height: float = 2.0,
    bloat_obstacles: float = 0.02,
) -> GraphOfConvexSets:
    """
    Create a navigation map around the target.
    The navigation map is a Graph of Convex Sets that represents the navigable space around the target.
    The search space is constructed as a box around the target with the specified search ranges in the x and y directions.

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

    gcs = GraphOfConvexSets.navigation_map_from_world(
        world=target._world, search_space=search_space, bloat_obstacles=bloat_obstacles
    )
    return gcs


def translate_free_space_to_where_condition(
    free_space: Event,
    expression: SymbolicExpression,
    x_variable_name: str = "x",
    y_variable_name: str = "y",
) -> OR:
    """
    Translate the free space event generated by a GCS to a where condition describing the constraints of X and Y
    variables.
    This results in an OR statement containing a union over all simple events in the free space.
    The components of the OR statement are conjunctions of constraints on the X and Y variables extracted from the simple
    events.

    :param free_space: The free space to parse
    :param expression: The expression where to get the variables from
    :param x_variable_name: The name of the X variable in the expression
    :param y_variable_name: The name of the Y variable in the expression
    :return: The where condition describing the constraints of X and Y variables
    """

    def resolve_variable(expr: SymbolicExpression, name: str) -> SymbolicExpression:
        if hasattr(expr, "selected_variable"):
            var = expr.selected_variable
            if name.startswith(var._name_ + "."):
                name = name[len(var._name_) + 1 :]
                expr = var

        for part in name.split("."):
            expr = getattr(expr, part)
        return expr

    x_var = resolve_variable(expression, x_variable_name)
    y_var = resolve_variable(expression, y_variable_name)

    free_space = free_space.marginal(SpatialVariables.xy)

    simple_event_conditions = []

    for simple_event in free_space.simple_sets:
        x_interval = simple_event[SpatialVariables.x.value]
        y_interval = simple_event[SpatialVariables.y.value]

        for si_x in x_interval.simple_sets:
            for si_y in y_interval.simple_sets:
                x_low = (
                    x_var >= si_x.lower
                    if si_x.left == Bound.CLOSED
                    else x_var > si_x.lower
                )
                x_high = (
                    x_var <= si_x.upper
                    if si_x.right == Bound.CLOSED
                    else x_var < si_x.upper
                )
                y_low = (
                    y_var >= si_y.lower
                    if si_y.left == Bound.CLOSED
                    else y_var > si_y.lower
                )
                y_high = (
                    y_var <= si_y.upper
                    if si_y.right == Bound.CLOSED
                    else y_var < si_y.upper
                )
                simple_event_conditions.append(
                    chained_logic(AND, x_low, x_high, y_low, y_high)
                )

    return chained_logic(OR, *simple_event_conditions)


def create_reference_frame_with_only_yaw_from_body(body: Body) -> Body:
    """
    Create a reference frame (new body without visual and collision) in the world.
    This reference frame is a body that ignores the roll and pitch but keeps the yaw and position.

    :param body: The body to create the reference frame from.
    :return: The newly created reference frame.
    """

    world = body._world
    reference_frame = Body(
        name=PrefixedName(prefix=str(body.name), name="base_with_yaw")
    )

    world_T_body = world.transform(body.global_pose, world.root)
    reference_frame_T_world = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=world_T_body.x,
        y=world_T_body.y,
        z=world_T_body.z,
        roll=0.0,
        pitch=0.0,
        yaw=world_T_body.yaw,
        reference_frame=world.root,
    )

    with world.modify_world():
        world.add_body(reference_frame)
        reference_frame_C_world = FixedConnection(
            world.root,
            child=reference_frame,
            parent_T_connection_expression=reference_frame_T_world,
        )
        world.add_connection(reference_frame_C_world)

    return reference_frame
