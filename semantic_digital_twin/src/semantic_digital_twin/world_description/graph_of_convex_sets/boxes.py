from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
from random_events.interval import Bound, closed, Interval, SimpleInterval
from random_events.plotting import EventPlotter
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from rtree import index
from typing_extensions import (
    Generic,
    List,
    Optional,
    Dict,
    Sequence,
    Self,
    Type,
    TypeVar,
)

from krrood.entity_query_language.core.mapped_variable import (
    CanBehaveLikeAVariable,
    MappedVariable,
)
from krrood.entity_query_language.factories import ConditionType, and_, or_
from krrood.entity_query_language.query.query import Query
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
    Wall,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    AxisAlignedBox,
    VolumetricBoundingBox,
    PlanarBoundingBox,
    Bounds,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    GraphOfConvexSets,
    PointT,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    AmbiguousSelectedVariableError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

BoxT = TypeVar("BoxT", bound=AxisAlignedBox)
"""
The bounding-box type a :class:`GraphOfBoundingBoxes` subclass decomposes free space
into -- :class:`VolumetricBoundingBox` for a volumetric decomposition,
:class:`PlanarBoundingBox` for a planar one.
"""


@dataclass
class BoundingBoxAdjacency(Generic[BoxT]):
    """
    Edge payload connecting two adjacent bounding boxes in a
    :class:`GraphOfBoundingBoxes`.
    """

    intersection: BoxT
    """
    The region where the two adjacent boxes overlap or touch.
    """

    distance: float
    """
    Euclidean distance between the centers of the two adjacent boxes.

    Used as the edge cost for shortest-path search, so that the search minimizes
    travelled distance instead of the number of boxes crossed.
    """


@dataclass
class GraphOfBoundingBoxes(
    Generic[BoxT, PointT],
    GraphOfConvexSets[PointT, BoundingBoxCollection[BoxT]],
):
    """
    Abstract base for graphs of convex sets whose nodes are axis-aligned bounding boxes.

    Free space is decomposed into an exact, exhaustive partition of boxes via the
    `random_events` product algebra (obstacles subtracted from the search space). Every
    node is a box; every edge represents the adjacency between two boxes. Concrete
    subclasses differ in how many dimensions that decomposition happens in --
    :class:`VolumetricGraphOfBoundingBoxes` partitions all three,
    :class:`PlanarGraphOfBoundingBoxes` partitions the floor plane only.
    """

    graph: rx.PyGraph[BoxT, BoundingBoxAdjacency[BoxT]] = field(
        default_factory=lambda: rx.PyGraph(multigraph=False)
    )
    """
    The connectivity graph of the convex sets.
    """

    box_to_index_map: Dict[BoxT, int] = field(default_factory=dict)
    """
    A mapping from bounding boxes to their indices in the graph.
    """

    @abstractmethod
    def _default_search_space(self) -> BoundingBoxCollection[BoxT]:
        raise NotImplementedError

    @classmethod
    def box_type(cls) -> Type[BoxT]:
        """
        :return: The concrete box type this subclass decomposes free space into --
            :class:`VolumetricBoundingBox` for :class:`VolumetricGraphOfBoundingBoxes`,
            :class:`PlanarBoundingBox` for :class:`PlanarGraphOfBoundingBoxes`.
        """
        return cls.get_generic_type_parameters()[0]

    @property
    def dimensionality(self) -> int:
        """
        :return: The number of spatial axes this graph's boxes are expressed over.
        """
        return self.box_type().dimensionality()

    def create_subgraph(self, nodes: Sequence[int]) -> Self:
        """
        Create a subgraph of the current graph containing only the given nodes.

        :param nodes: The nodes to include in the subgraph.
        :return: The subgraph.
        """
        subgraph = type(self)(self.world, self.search_space)
        subgraph.graph = self.graph.subgraph(nodes)
        subgraph.box_to_index_map = {
            box: index for box, index in self.box_to_index_map.items() if index in nodes
        }
        return subgraph

    def add_node(self, box: BoxT):
        """
        :param box: The box to add as a node.
        """
        self.box_to_index_map[box] = self.graph.add_node(box)

    def calculate_connectivity(self, tolerance: float = 0.001):
        """
        Calculate the connectivity of the graph by checking for intersections between
        the bounding boxes of the nodes. This uses an R-tree for efficient spatial
        indexing and intersection queries. Each edge is weighted by the Euclidean
        distance between the centers of the two boxes it connects, for use by
        :meth:`path_from_to`.

        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        """

        def _overlap(a_min, a_max, b_min, b_max) -> bool:
            return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))

        node_list = list(self.graph.nodes())
        if not node_list:
            return

        # VolumetricBoundingBox.x_interval/y_interval/z_interval (and their planar counterparts)
        # recompute symbolic arithmetic on every access, so every node's bounds are read
        # as plain floats exactly once here rather than once per pair below.
        bounds_list = [node.to_array_bounds() for node in node_list]
        dimensionality = self.dimensionality
        centers = [(bounds.lower + bounds.upper) / 2 for bounds in bounds_list]
        expanded = [
            tuple(bounds.lower - tolerance) + tuple(bounds.upper + tolerance)
            for bounds in bounds_list
        ]

        prop = index.Property()
        prop.dimension = dimensionality
        rtree_idx = index.Index(properties=prop)
        for i, box_expansion in enumerate(expanded):
            rtree_idx.insert(i, box_expansion)

        # Query & link, skip self-loops and symmetric pairs
        for i, bounds_i in enumerate(bounds_list):
            for j in rtree_idx.intersection(expanded[i]):
                if j <= i:  # symmetry → skip
                    continue
                bounds_j = bounds_list[j]
                if not _overlap(
                    bounds_i.lower, bounds_i.upper, bounds_j.lower, bounds_j.upper
                ):
                    continue  # no true overlap
                lower = np.maximum(bounds_i.lower, bounds_j.lower)
                upper = np.minimum(bounds_i.upper, bounds_j.upper)
                box = type(node_list[i]).from_array_bounds(
                    lower,
                    upper,
                    HomogeneousTransformationMatrix(reference_frame=self.world.root),
                )
                distance = float(np.linalg.norm(centers[i] - centers[j]))

                # Map from the local list positions back to the graph node indices
                u = self.box_to_index_map[node_list[i]]
                v = self.box_to_index_map[node_list[j]]

                self.graph.add_edge(u, v, BoundingBoxAdjacency(box, distance))

    def draw(self):
        import rustworkx.visualization

        rustworkx.visualization.mpl_draw(self.graph)
        plt.show()

    def plot_free_space(self) -> List[go.BaseTraceType]:
        """
        Plot the free space of the environment in blue.

        :return: A list of traces that can be put into a plotly figure.
        """
        return EventPlotter(self.free_space_event).plot(color="blue")

    def plot_and_show_free_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_free_space()).show()

    def plot_occupied_space(self) -> List[go.BaseTraceType]:
        """
        Plot the occupied space of the environment in red.

        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )
        occupied_space = ~free_space & self.search_space.event
        return EventPlotter(occupied_space).plot(color="red")

    def plot_and_show_occupied_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_occupied_space()).show()

    def node_of_point(self, point: PointT) -> Optional[BoxT]:
        """
        Find the node that contains a point.

        :return: The node that contains the point or None if no node contains the point.
        """
        for node in self.graph.nodes():
            if node.contains(point):
                return node
        return None

    def path_from_to(self, start: PointT, goal: PointT) -> Optional[List[PointT]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        .. note::
            Uses a single-source Dijkstra search, weighted by the Euclidean distance
            between adjacent boxes' centers, rather than enumerating all shortest paths
            and picking the first one. Free-space decompositions with thousands of
            nodes routinely have an exponential number of equally-short (by hop count)
            paths, which makes enumerating all of them intractable; finding the one
            that minimizes travelled distance is not.

        .. note::
            The resulting waypoints are shortcut afterwards: any waypoint that a
            straight line can bypass without leaving free space is dropped. See
            :meth:`_shortcut_waypoints`.

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

        paths = rx.dijkstra_shortest_paths(
            self.graph,
            start_index,
            target=goal_index,
            weight_fn=lambda adjacency: adjacency.distance,
        )

        # if it is not possible to find a path
        if goal_index not in paths:
            return None

        path = paths[goal_index]

        # build the path
        reference_frame = self.search_space.reference_frame
        waypoints = [self.world.transform(start, reference_frame)]

        for source, target in zip(path, path[1:]):
            intersection = self.graph.get_edge_data(source, target).intersection
            waypoints.append(intersection.center)

        waypoints.append(self.world.transform(goal, reference_frame))
        waypoints = self._shortcut_waypoints(waypoints)

        result = [start]
        result.extend(waypoints[1:-1])
        result.append(goal)
        return result

    def _shortcut_waypoints(self, waypoints: List[PointT]) -> List[PointT]:
        """
        Drop waypoints that a straight line can bypass without leaving free space.

        Greedily extends the current anchor waypoint forward as far as a straight
        line to it stays collision-free, then commits the farthest waypoint still
        visible from it and continues from there (classic "string pulling"). Each
        waypoint is tested against the current anchor at most once, so this is
        linear in the number of waypoints rather than quadratic.

        :param waypoints: The waypoints of a path, in the search space's reference
            frame.
        :return: The shortcut waypoints.
        """
        if len(waypoints) <= 2:
            return list(waypoints)

        node_bounds = [node.to_array_bounds() for node in self.graph.nodes()]

        result = [waypoints[0]]
        anchor_index = 0
        for index in range(2, len(waypoints)):
            if not self._segment_is_collision_free(
                waypoints[anchor_index], waypoints[index], node_bounds
            ):
                result.append(waypoints[index - 1])
                anchor_index = index - 1
        result.append(waypoints[-1])
        return result

    def _segment_is_collision_free(
        self, start: PointT, end: PointT, node_bounds: List[Bounds[np.ndarray]]
    ) -> bool:
        """
        Check whether a straight-line segment stays entirely within free space.

        :param start: The segment's start point, in the search space's reference frame.
        :param end: The segment's end point, in the search space's reference frame.
        :param node_bounds: The graph's nodes' bounds, in the same order
            :meth:`_shortcut_waypoints` collected them.
        :return: True if the segment never leaves the union of the graph's bounding-box
            nodes.
        """
        dimensionality = self.dimensionality
        coordinates = np.asarray(start.to_np()[:dimensionality], dtype=float)
        end_coordinates = np.asarray(end.to_np()[:dimensionality], dtype=float)
        deltas = end_coordinates - coordinates
        covered_intervals = [
            interval
            for bounds in node_bounds
            if (interval := bounds.clip_segment(coordinates, deltas)) is not None
        ]
        if not covered_intervals:
            return False
        covered = Interval.from_simple_sets(*covered_intervals).make_disjoint()
        return (closed(0.0, 1.0) - covered).is_empty()

    @property
    def free_space_event(self) -> Event:
        return Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )

    def constrain_to_free_space(self, variable: MappedVariable) -> ConditionType:
        """
        Add a where condition to ``variable``'s query, restricting it to lie within
        this graph's free space.

        The free space is a union of boxes, so the condition is an ``OR`` over one
        ``AND`` per box, each conjoining a lower and an upper bound per coordinate.
        Which coordinates that is -- x, y for a planar graph; x, y, z for a
        volumetric one -- follows from :attr:`free_space_event` alone, so this one
        implementation serves every :class:`GraphOfBoundingBoxes` subclass.

        ``variable`` is rerooted onto its query's own ``selected_variable`` before
        its fields are read: a query's own type does not resolve (what it selects
        lives on ``selected_variable`` instead), and a condition built directly
        against the selection is exactly what evaluates correctly per query result
        without needing the query's own rerooting machinery to intervene.

        :param variable: The eql variable to constrain, e.g. a Pose2D- or
            Point3-typed attribute of a query, or the query itself.
        :return: The condition that was added.
        :raises NotNumberLikeFieldError: If ``variable`` has no number-like field for
            one of this graph's spatial coordinates.
        """
        chain_root = (
            variable._chain_root_ if isinstance(variable, MappedVariable) else variable
        )
        selected_variable = GraphOfBoundingBoxes._selected_variable_of(chain_root)
        if selected_variable is not None:
            variable = (
                variable._reroot_on_(selected_variable, chain_root)
                if isinstance(variable, MappedVariable)
                else selected_variable
            )

        free_space_event = self.free_space_event
        fields = {
            spatial_variable.name: getattr(
                variable, spatial_variable.name
            ).number_like_field()
            for spatial_variable in free_space_event.variables
        }

        simple_event_conditions = [
            and_(
                *(
                    self._axis_condition(fields[name], simple_event[name].simple_sets)
                    for name in fields
                )
            )
            for simple_event in free_space_event.simple_sets
        ]

        condition = or_(*simple_event_conditions)
        chain_root.where(condition)
        return condition

    @staticmethod
    def _axis_condition(
        field: MappedVariable, intervals: Sequence[SimpleInterval]
    ) -> ConditionType:
        """
        Build the condition constraining one coordinate to a union of intervals.

        Mirrors the event's own OR-of-AND structure for that coordinate directly,
        rather than joining it with every other coordinate's intervals first: a
        query condition does not need each axis combination spelled out as its own
        convex clause the way a :class:`GraphOfBoundingBoxes` node does -- that
        flattening exists only to keep the graph's own nodes convex, and belongs on
        :meth:`~semantic_digital_twin.world_description.geometry.AxisAlignedBox.from_simple_event`,
        not here.

        :param field: The query field this coordinate constrains.
        :param intervals: The simple intervals the field must lie within one of.
        :return: An OR of one AND(lower bound, upper bound) condition per interval.
        """
        return or_(
            *(
                and_(
                    (
                        field >= interval.lower
                        if interval.left == Bound.CLOSED
                        else field > interval.lower
                    ),
                    (
                        field <= interval.upper
                        if interval.right == Bound.CLOSED
                        else field < interval.upper
                    ),
                )
                for interval in intervals
            )
        )

    @staticmethod
    def _selected_variable_of(root) -> Optional[CanBehaveLikeAVariable]:
        """
        :param root: The chain root of an eql variable.
        :return: The single variable ``root`` selects, or None if ``root`` is not a
            query at all (e.g. a bare :class:`~krrood.entity_query_language.core.variable.Variable`).
        :raises AmbiguousSelectedVariableError: If ``root`` is a query selecting more
            than one variable -- an :class:`Entity` picking the first would silently
            constrain the wrong one, and a :class:`SetOf` has no single variable to
            constrain at all.
        """
        if not isinstance(root, Query):
            return None
        selected_variable_count = len(root._selected_variables_)
        if selected_variable_count != 1:
            raise AmbiguousSelectedVariableError(root, selected_variable_count)
        return root._selected_variables_[0]


@dataclass
class VolumetricGraphOfBoundingBoxes(
    GraphOfBoundingBoxes[VolumetricBoundingBox, Point3]
):
    """
    A graph of convex sets whose nodes are axis-aligned bounding boxes, partitioning
    free space in all three dimensions.
    """

    def _default_search_space(self) -> BoundingBoxCollection[VolumetricBoundingBox]:
        """
        :return: A search space spanning the entire three-dimensional space around
            ``self.world.root``.
        """
        return BoundingBoxCollection(
            shapes=[
                VolumetricBoundingBox(
                    min_x=-np.inf,
                    min_y=-np.inf,
                    min_z=-np.inf,
                    max_x=np.inf,
                    max_y=np.inf,
                    max_z=np.inf,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=self.world.root
                    ),
                )
            ],
            reference_frame=self.world.root,
        )

    @classmethod
    def free_space_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection[VolumetricBoundingBox],
        search_space_event: Event,
    ) -> Event:
        """
        Compute the free space by subtracting each obstacle bounding box from the search
        space incrementally (subtract_disjoint), avoiding complement in the full ambient
        space and the costly union-then-complement pipeline.

        This is 40-50× faster than computing the obstacles' union directly and
        complementing it, because:
        - The subtraction stays bounded inside search_space_event at every step.
        - No make_disjoint() calls are needed (disjointness is maintained by construction).
        - The intermediate obstacle union is never materialised.

        :param bounding_boxes: The obstacle bounding boxes to subtract.
        :param search_space_event: The search space; the result is always a subset of this.
        :return: The free space as a disjoint Event.
        """
        free_space = search_space_event
        for bounding_box in bounding_boxes:
            obstacle = bounding_box.simple_event.as_composite_set()
            obstacle_in_search = obstacle & search_space_event
            if not obstacle_in_search.is_empty():
                free_space = free_space.subtract_disjoint(obstacle_in_search)
            if free_space.is_empty():
                break
        return free_space

    @classmethod
    def free_space_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection[VolumetricBoundingBox],
        semantic_obstacle_annotation: SemanticEnvironmentAnnotation,
        semantic_wall_annotation: Optional[Wall] = None,
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
        :param semantic_wall_annotation: An optional wall annotation to be considered
            as an obstacle.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        bloated_obstacles = (
            semantic_obstacle_annotation.build_bloated_obstacle_collection(
                search_space,
                semantic_wall_annotation,
                bloat_obstacles,
                bloat_walls,
            )
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
                cls.box_type(),
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
        search_space: BoundingBoxCollection[VolumetricBoundingBox],
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


@dataclass
class PlanarGraphOfBoundingBoxes(GraphOfBoundingBoxes[PlanarBoundingBox, Point2]):
    """
    A graph of convex sets whose nodes are axis-aligned bounding boxes, partitioning
    free space on a single navigable plane.

    Built for base navigation: the z-axis never enters the decomposition, so a query
    only has to answer whether an x,y footprint is free -- and an obstacle blocks a
    footprint at every height, since the robot has to fit through the entire column of
    space above it, not just its floor-level silhouette.
    """

    def _default_search_space(self) -> BoundingBoxCollection[PlanarBoundingBox]:
        """
        :return: A search space spanning the entire two-dimensional plane around
            ``self.world.root``.
        """
        return BoundingBoxCollection(
            shapes=[
                PlanarBoundingBox(
                    min_x=-np.inf,
                    min_y=-np.inf,
                    max_x=np.inf,
                    max_y=np.inf,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=self.world.root
                    ),
                )
            ],
            reference_frame=self.world.root,
        )

    @classmethod
    def free_space_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection[VolumetricBoundingBox],
        search_space_event: Event,
    ) -> Event:
        """
        Compute the floor-plan free space by subtracting each obstacle's floor
        footprint from the search space's floor footprint incrementally
        (subtract_disjoint), the same way :meth:`VolumetricGraphOfBoundingBoxes.free_space_from_bounding_boxes`
        does in three dimensions: an obstacle blocks a footprint at every height
        within the search space, since the robot has to fit through the entire column
        of space above it, not just its floor-level silhouette -- but an obstacle
        outside the search space's own height range (e.g. the floor itself, sitting
        below where the robot's navigable space begins) does not block anything.

        Intersecting each obstacle with the full, three-dimensional search space
        before marginalizing onto x,y is what makes the height range matter: doing it
        the other way around -- marginalizing the obstacle first -- would drop its
        z-extent before ever comparing it to the search space's, so no obstacle could
        ever be excluded by height.

        :param bounding_boxes: The obstacle bounding boxes to subtract.
        :param search_space_event: The three-dimensional search space; the result is
            always a subset of its floor footprint.
        :return: The floor-plan free space as a disjoint, two-dimensional Event.
        """
        free_space = search_space_event.marginal(SpatialVariables.xy)
        for bounding_box in bounding_boxes:
            obstacle_in_search_space = (
                bounding_box.simple_event.as_composite_set() & search_space_event
            )
            if obstacle_in_search_space.is_empty():
                continue
            obstacle_in_search = obstacle_in_search_space.marginal(SpatialVariables.xy)
            if not obstacle_in_search.is_empty():
                free_space = free_space.subtract_disjoint(obstacle_in_search)
            if free_space.is_empty():
                break
        return free_space

    @classmethod
    def free_space_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection[VolumetricBoundingBox],
        semantic_obstacle_annotation: SemanticEnvironmentAnnotation,
        semantic_wall_annotation: Optional[Wall] = None,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
        obstacle_height_clearance: float = 0.01,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param search_space: The three-dimensional search space for the connectivity
            graph -- its height range bounds which obstacles count as blocking, since
            the robot has to fit through the entire space, not just the floor-level
            obstacles. The graph's own :attr:`search_space` is this volume's floor
            footprint.
        :param semantic_obstacle_annotation: The semantic annotation containing the
            obstacles.
        :param semantic_wall_annotation: An optional wall annotation to be considered
            as an obstacle.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :param obstacle_height_clearance: The amount every obstacle bounding box gets
            expanded by in z, regardless of ``bloat_obstacles``/``bloat_walls``.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        world = search_space.reference_frame._world
        floor_search_space = BoundingBoxCollection.from_event(
            cls.box_type(),
            search_space.reference_frame,
            search_space.event.marginal(SpatialVariables.xy),
        )

        nav_obstacles = semantic_obstacle_annotation.build_bloated_obstacle_collection(
            search_space,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
            obstacle_height_clearance,
        )

        if not nav_obstacles:
            return cls(world=world, search_space=floor_search_space)

        free_space = cls.free_space_from_bounding_boxes(
            nav_obstacles, search_space.event
        )

        # create a connectivity graph from the free space and calculate the edges
        result = cls(world=world, search_space=floor_search_space)
        free_space_boxes = BoundingBoxCollection.from_event(
            cls.box_type(), search_space.reference_frame, free_space
        )
        [result.add_node(bounding_box) for bounding_box in free_space_boxes]
        result.calculate_connectivity(tolerance)

        return result

    @classmethod
    def navigation_map_from_world(
        cls,
        world: World,
        tolerance=0.001,
        search_space: Optional[BoundingBoxCollection[VolumetricBoundingBox]] = None,
        bloat_obstacles: float = 0.0,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param world: The belief state.
        :param search_space: The three-dimensional search space for the connectivity
            graph -- three-dimensional, not planar, because its height range bounds
            which obstacles count as blocking: the robot has to fit through the entire
            space, not just the floor-level obstacles. The graph's own
            :attr:`~GraphOfConvexSets.search_space` is this volume's floor footprint.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :return: The connectivity graph.
        """
        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return cls.free_space_from_semantic_annotation(
            search_space,
            semantic_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
        )

    @classmethod
    def navigation_map_at_target(
        cls,
        target: Body,
        search_range_x: float = 2.0,
        search_range_y: float = 2.0,
        max_height: float = 2.0,
        bloat_obstacles: float = 0.02,
    ) -> Self:
        """
        Create a navigation map around a target.

        The navigation map is a graph of convex sets representing the navigable space
        around the target. The search space is constructed as a box around the target
        with the specified search ranges in the x and y directions.

        :param target: The target around which the navigation map is created.
        :param search_range_x: The search range in the x-direction.
        :param search_range_y: The search range in the y-direction.
        :param max_height: The maximum height of the navigation map from the floor.
        :param bloat_obstacles: The amount to bloat obstacles in the navigation map.
        :return: The navigation map.
        """
        search_space = BoundingBoxCollection.from_simple_event(
            VolumetricBoundingBox,
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

        return cls.navigation_map_from_world(
            world=target._world,
            search_space=search_space,
            bloat_obstacles=bloat_obstacles,
        )
