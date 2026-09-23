"""
Expose an attached world's native entities as live query domains and presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

from cramera.knowledge.presets import Preset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
from cramera.live.query import LiveQuerySource
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from semantic_digital_twin.world import World


# %% world query vocabulary
class WorldQueryName(StrEnum):
    """
    Domain names available to queries over native world entities.
    """

    BODY = "body"
    """
    Bodies belonging to the world.
    """
    ANNOTATION = "annotation"
    """
    All semantic annotations registered in the world.
    """
    HANDLE = "handle"
    """
    Semantic annotations identifying graspable handles.
    """
    SURFACE = "surface"
    """
    Semantic annotations with supporting surfaces.
    """
    ROBOT = "robot"
    """
    Robot annotations describing the world's robots.
    """
    ARM = "arm"
    """
    Arm annotations registered in the world.
    """


class WorldQueryLabel(StrEnum):
    """
    Display labels for the native world query source and its presets.
    """

    TITLE = "Live world"
    """
    Title of the attached world's query source.
    """
    BODIES = "show all scene bodies"
    """
    Label for listing the world's bodies.
    """
    ANNOTATIONS = "show all semantic annotations"
    """
    Label for listing every registered semantic annotation.
    """
    HANDLES = "show all handles"
    """
    Label for listing handle annotations.
    """
    SURFACES = "show all supporting surfaces"
    """
    Label for listing annotations with supporting surfaces.
    """
    ROBOTS = "show all robots"
    """
    Label for listing robot annotations.
    """
    ARMS = "show all robot arms"
    """
    Label for listing arm annotations.
    """


# %% native world queries
@dataclass
class WorldQuerySource(LiveQuerySource):
    """
    Query the current bodies and semantic annotations of an attached world.

    Domains retain the native entities so queries can inspect their current properties.
    """

    world: World
    """
    World supplying the current native entities for each request.
    """

    def title(self) -> str:
        """
        Identify the attached world as the source of query answers.

        :return: The display title for live world queries.
        """
        return WorldQueryLabel.TITLE

    def read_scope(self) -> AbstractContextManager[object]:
        """
        Provide the world's lock for consistent reads during a query.

        :return: The lock to hold until query evaluation and result rendering finish.
        """
        return self.world.state.world_lock

    def knowledge(self) -> list[QueryableKnowledge]:
        """
        Collect the world's current native entities into their query domains.

        Hold :meth:`read_scope` while reading and evaluating the returned knowledge.

        :return: Current-state knowledge with fresh collections of native entities.
        """
        return [
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[
                    QueryDomain(WorldQueryName.BODY, Body, list(self.world.bodies)),
                    QueryDomain(
                        WorldQueryName.ANNOTATION,
                        SemanticAnnotation,
                        list(self.world.semantic_annotations),
                    ),
                    QueryDomain(
                        WorldQueryName.HANDLE,
                        Handle,
                        list(self.world.get_semantic_annotations_by_type(Handle)),
                    ),
                    QueryDomain(
                        WorldQueryName.SURFACE,
                        HasSupportingSurface,
                        list(
                            self.world.get_semantic_annotations_by_type(
                                HasSupportingSurface
                            )
                        ),
                    ),
                    QueryDomain(
                        WorldQueryName.ROBOT,
                        AbstractRobot,
                        list(
                            self.world.get_semantic_annotations_by_type(AbstractRobot)
                        ),
                    ),
                    QueryDomain(
                        WorldQueryName.ARM,
                        Arm,
                        list(self.world.get_semantic_annotations_by_type(Arm)),
                    ),
                ],
            )
        ]

    def presets(self) -> list[Preset]:
        """
        Offer a query listing all entities in each native world domain.

        :return: Presets requiring a live source, ordered by domain.
        """
        return [
            Preset(label, f"an(entity({name}))", requires_live=True)
            for label, name in (
                (WorldQueryLabel.BODIES, WorldQueryName.BODY),
                (WorldQueryLabel.ANNOTATIONS, WorldQueryName.ANNOTATION),
                (WorldQueryLabel.HANDLES, WorldQueryName.HANDLE),
                (WorldQueryLabel.SURFACES, WorldQueryName.SURFACE),
                (WorldQueryLabel.ROBOTS, WorldQueryName.ROBOT),
                (WorldQueryLabel.ARMS, WorldQueryName.ARM),
            )
        ]
