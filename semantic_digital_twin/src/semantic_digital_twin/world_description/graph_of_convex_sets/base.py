from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from typing_extensions import Generic, List, Optional, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

PointT = TypeVar("PointT", bound=Point)
"""
The point type a :class:`GraphOfConvexSets` subclass queries and returns paths in --
:class:`~semantic_digital_twin.spatial_types.Point3` for a graph that plans in three
dimensions, :class:`~semantic_digital_twin.spatial_types.Point2` for one that plans on
a single plane.
"""

SearchSpaceT = TypeVar("SearchSpaceT")
"""
The search-space representation a :class:`GraphOfConvexSets` subclass is built within --
a three-dimensional :class:`BoundingBoxCollection` or its planar counterpart.
"""


@dataclass
class GraphOfConvexSets(Generic[PointT, SearchSpaceT], SubClassSafeGeneric, ABC):
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

    search_space: Optional[SearchSpaceT] = None
    """
    The bounding box of the search space.

    Pass ``None`` to default to the entire search space :meth:`_default_search_space`
    describes; ``__post_init__`` resolves that default, so this attribute is never
    ``None`` once the object exists.
    """

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = self._default_search_space()

    @abstractmethod
    def path_from_to(self, start: PointT, goal: PointT) -> Optional[List[PointT]]:
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
