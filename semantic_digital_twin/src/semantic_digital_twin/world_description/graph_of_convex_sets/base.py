from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.operators.core_logical_operators import (
    OR,
    AND,
    chained_logic,
)
from random_events.interval import Interval, SimpleInterval, Bound
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from typing_extensions import List, Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)


@dataclass
class GraphOfConvexSets(ABC):
    """
    Abstract base for planning graphs whose nodes are convex sets of free space.

    A graph of convex sets (GCS) represents the navigable free space of a world as a
    collection of convex regions, connected by edges wherever two regions are adjacent
    or overlapping. Concrete subclasses differ in how they represent those regions and
    how they solve a shortest-path query over them.

    You can read more about GCS in :cite:t:`marcucci2021motion`.
    """

    world: World
    """
    The world that the graph is based on.
    """

    search_space: Optional[BoundingBoxCollection] = None
    """
    The bounding box of the search space.

    Pass ``None`` to default to the entire three-dimensional space; ``__post_init__``
    resolves that default, so this attribute is never ``None`` once the object exists.
    """

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = self._default_search_space()

    @abstractmethod
    def path_from_to(self, start: Point3, goal: Point3) -> Optional[List[Point3]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to, or None if no path
            exists.
        :raises PointOccupiedError: If ``start`` or ``goal`` lies inside an obstacle.
        """
        raise NotImplementedError

    def _default_search_space(self) -> BoundingBoxCollection:
        """
        :return: A search space spanning the entire three-dimensional space around
            ``self.world.root``.
        """
        return BoundingBoxCollection(
            shapes=[
                BoundingBox(
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
    def _obstacle_entities(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
    ) -> List[Body]:
        """
        Collect the obstacle bodies to consider from a semantic annotation.

        Filters out agent entities so the robot does not treat itself as an obstacle,
        and bodies without meaningful collision geometry.

        :param search_space: The search space; its reference frame is used to look up
            the owning world.
        :param semantic_obstacle_annotation: The annotation containing obstacle
            entities.
        :return: The obstacle bodies to consider.
        """
        world = search_space.reference_frame._world

        agents = world.get_semantic_annotations_by_type(Agent)
        agent_entities = set()
        for agent in agents:
            agent_entities.update(agent.kinematic_structure_entities)

        return [
            entity
            for entity in semantic_obstacle_annotation.kinematic_structure_entities
            if isinstance(entity, Body)
            and entity.has_collision()
            and entity not in agent_entities
        ]

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

        Applies independent bloat amounts to obstacles and walls.

        :param search_space: The search space; its reference frame is used as the
            origin.
        :param semantic_obstacle_annotation: The annotation containing obstacle
            entities.
        :param semantic_wall_annotation: An optional annotation containing wall
            entities.
        :param bloat_obstacles: Amount to expand each obstacle bounding box
            symmetrically in x and y.
        :param bloat_walls: Amount to expand wall bounding boxes in their thinner
            dimension.
        :return: A BoundingBoxCollection of the bloated obstacle and wall bounding
            boxes.
        """
        world_root = search_space.reference_frame

        entities_to_consider = cls._obstacle_entities(
            search_space, semantic_obstacle_annotation
        )

        collections = [
            entity.collision.as_bounding_box_collection_at_origin(
                HomogeneousTransformationMatrix(reference_frame=world_root)
            )
            for entity in entities_to_consider
        ]

        obstacle_bounding_boxes = BoundingBoxCollection([], world_root)
        for bounding_box_collection in collections:
            obstacle_bounding_boxes = obstacle_bounding_boxes.merge(
                bounding_box_collection
            )

        bloated_obstacles = BoundingBoxCollection(
            [
                bounding_box.bloat(bloat_obstacles, bloat_obstacles, 0.01)
                for bounding_box in obstacle_bounding_boxes
            ],
            world_root,
        )

        if semantic_wall_annotation is not None:
            bloated_walls: BoundingBoxCollection = BoundingBoxCollection(
                [
                    (
                        bounding_box.bloat(bloat_walls, 0, 0.01)
                        if bounding_box.width > bounding_box.depth
                        else bounding_box.bloat(0, bloat_walls, 0.01)
                    )
                    for bounding_box in semantic_wall_annotation.as_bounding_box_collection_at_origin(
                        HomogeneousTransformationMatrix(reference_frame=world_root)
                    )
                ],
                world_root,
            )
            bloated_obstacles.merge(bloated_walls)

        return bloated_obstacles


def translate_event_to(
    event: Event,
    position: Point3,
) -> Event:
    """
    Translates an event by a given position.

    A translation is a change in the position of an entity in space without altering its
    shape or orientation.

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


def translate_free_space_to_where_condition(
    free_space: Event,
    expression: SymbolicExpression,
    x_variable_name: str = "x",
    y_variable_name: str = "y",
) -> OR:
    """
    Translate the free space event generated by a GCS to a where condition describing
    the constraints of X and Y variables. This results in an OR statement containing a
    union over all simple events in the free space. The components of the OR statement
    are conjunctions of constraints on the X and Y variables extracted from the simple
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

    This reference frame is a body that ignores the roll and pitch but keeps the yaw and
    position.

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
